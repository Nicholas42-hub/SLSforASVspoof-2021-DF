#!/usr/bin/env python3
"""
Analyze the semantic meaning of boundary discontinuities:
- Are boundary jumps correlated with prediction errors?
- Do they occur at attack transition points?
- Are they informative or just noise?
"""
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from scipy import stats
from collections import defaultdict

from model_window_topk import Model as WindowTopKModel
from data_utils_SSL import genSpoof_list

class BoundarySemanticAnalyzer:
    def __init__(self, window_size=8):
        self.window_size = window_size
    
    def detect_boundary_positions(self, num_frames):
        """Identify which frame pairs cross window boundaries"""
        boundaries = []
        for t in range(num_frames - 1):
            if (t + 1) % self.window_size == 0:
                boundaries.append(t)
        return boundaries
    
    def compute_discontinuity_at_frames(self, activations):
        """
        Compute Jaccard similarity for each adjacent frame pair
        
        Returns:
            jaccard_scores: [T-1] array of Jaccard similarities
            is_boundary: [T-1] boolean array indicating boundary transitions
        """
        active = (activations > 0).float()
        T = active.shape[0]
        
        jaccard_scores = []
        is_boundary = []
        
        for t in range(T - 1):
            cue_t = active[t]
            cue_t1 = active[t + 1]
            
            intersection = (cue_t * cue_t1).sum()
            union = ((cue_t + cue_t1) > 0).float().sum()
            
            if union > 0:
                jacc = (intersection / union).item()
            else:
                jacc = 0.0
            
            jaccard_scores.append(jacc)
            
            # Check if this is a boundary transition
            is_boundary.append((t + 1) % self.window_size == 0)
        
        return np.array(jaccard_scores), np.array(is_boundary)
    
    def correlate_boundaries_with_errors(
        self, activations, prediction, ground_truth, confidence
    ):
        """
        Check if boundary discontinuities correlate with prediction errors
        
        Returns:
            dict with correlation statistics
        """
        jaccard_scores, is_boundary = self.compute_discontinuity_at_frames(activations)
        
        # Compute discontinuity (1 - Jaccard)
        discontinuity = 1 - jaccard_scores
        
        # Separate boundary vs interior
        boundary_disc = discontinuity[is_boundary]
        interior_disc = discontinuity[~is_boundary]
        
        # Is this sample correctly predicted?
        is_correct = (prediction == ground_truth)
        
        # Mean discontinuity
        mean_boundary_disc = boundary_disc.mean() if len(boundary_disc) > 0 else 0
        mean_interior_disc = interior_disc.mean() if len(interior_disc) > 0 else 0
        
        return {
            'is_correct': is_correct,
            'confidence': confidence,
            'mean_boundary_discontinuity': mean_boundary_disc,
            'mean_interior_discontinuity': mean_interior_disc,
            'boundary_disc_values': boundary_disc,
            'interior_disc_values': interior_disc,
            'num_boundaries': len(boundary_disc),
        }
    
    def analyze_attack_transitions(self, activations, feature_ids):
        """
        Analyze if boundary jumps correspond to changes in attack patterns
        
        Strategy: Check if certain decision features show abrupt changes
        at boundaries, which might indicate detection of attack transitions
        """
        decision_activations = activations[:, feature_ids]
        T = decision_activations.shape[0]
        
        # Compute activation differences at boundaries vs interior
        boundary_changes = []
        interior_changes = []
        
        for t in range(T - 1):
            # L1 difference in decision feature activations
            change = torch.abs(decision_activations[t+1] - decision_activations[t]).mean().item()
            
            if (t + 1) % self.window_size == 0:
                boundary_changes.append(change)
            else:
                interior_changes.append(change)
        
        return {
            'mean_boundary_change': np.mean(boundary_changes) if boundary_changes else 0,
            'mean_interior_change': np.mean(interior_changes) if interior_changes else 0,
            'boundary_changes': boundary_changes,
            'interior_changes': interior_changes,
        }

def analyze_correlation_with_errors(
    model, eval_loader, decision_features, device, num_samples=1000
):
    """
    Main analysis: correlate boundary discontinuities with prediction errors
    """
    analyzer = BoundarySemanticAnalyzer(window_size=8)
    
    results = {
        'correct_predictions': [],
        'incorrect_predictions': [],
    }
    
    model.eval()
    processed = 0
    
    print("Analyzing correlation between boundaries and errors...")
    
    with torch.no_grad():
        for utt_id, audio_path, target in tqdm(eval_loader, total=num_samples):
            if processed >= num_samples:
                break
            
            try:
                # Load audio
                import torchaudio
                audio, _ = torchaudio.load(audio_path[0])
                audio = audio.to(device)
                
                if audio.shape[0] > 1:
                    audio = audio.mean(dim=0, keepdim=True)
                
                # Get prediction
                output = model(audio)
                prediction = torch.argmax(output, dim=1).item()
                confidence = torch.softmax(output, dim=1).max().item()
                ground_truth = target[0].item()
                
                # Get activations
                ssl_feats = model.ssl_model(audio)
                if isinstance(ssl_feats, tuple):
                    ssl_feats = ssl_feats[0]
                activations = model.sae.get_activations(ssl_feats)
                
                # Analyze
                result = analyzer.correlate_boundaries_with_errors(
                    activations, prediction, ground_truth, confidence
                )
                
                if result['is_correct']:
                    results['correct_predictions'].append(result)
                else:
                    results['incorrect_predictions'].append(result)
                
                processed += 1
                
            except Exception as e:
                print(f"Error processing {audio_path[0]}: {e}")
                continue
    
    return results

def visualize_boundary_error_correlation(results, save_path):
    """
    Visualize the relationship between boundary discontinuity and errors
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Extract data
    correct_boundary = [r['mean_boundary_discontinuity'] for r in results['correct_predictions']]
    correct_interior = [r['mean_interior_discontinuity'] for r in results['correct_predictions']]
    
    incorrect_boundary = [r['mean_boundary_discontinuity'] for r in results['incorrect_predictions']]
    incorrect_interior = [r['mean_interior_discontinuity'] for r in results['incorrect_predictions']]
    
    # Plot 1: Distribution comparison
    ax = axes[0, 0]
    ax.hist(correct_boundary, bins=30, alpha=0.6, label='Correct', color='green')
    ax.hist(incorrect_boundary, bins=30, alpha=0.6, label='Incorrect', color='red')
    ax.set_xlabel('Boundary Discontinuity')
    ax.set_ylabel('Frequency')
    ax.set_title('Boundary Discontinuity: Correct vs Incorrect Predictions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Interior comparison
    ax = axes[0, 1]
    ax.hist(correct_interior, bins=30, alpha=0.6, label='Correct', color='green')
    ax.hist(incorrect_interior, bins=30, alpha=0.6, label='Incorrect', color='red')
    ax.set_xlabel('Interior Discontinuity')
    ax.set_ylabel('Frequency')
    ax.set_title('Interior Discontinuity: Correct vs Incorrect Predictions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Scatter plot - boundary vs confidence
    ax = axes[1, 0]
    correct_conf = [r['confidence'] for r in results['correct_predictions']]
    incorrect_conf = [r['confidence'] for r in results['incorrect_predictions']]
    
    ax.scatter(correct_boundary, correct_conf, alpha=0.5, s=10, c='green', label='Correct')
    ax.scatter(incorrect_boundary, incorrect_conf, alpha=0.5, s=10, c='red', label='Incorrect')
    ax.set_xlabel('Boundary Discontinuity')
    ax.set_ylabel('Prediction Confidence')
    ax.set_title('Boundary Discontinuity vs Confidence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Statistical comparison
    ax = axes[1, 1]
    data_to_plot = [
        correct_boundary, incorrect_boundary,
        correct_interior, incorrect_interior
    ]
    labels = ['Correct\n(Boundary)', 'Incorrect\n(Boundary)',
              'Correct\n(Interior)', 'Incorrect\n(Interior)']
    
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
    for i, box in enumerate(bp['boxes']):
        color = 'lightgreen' if i % 2 == 0 else 'lightcoral'
        box.set_facecolor(color)
    
    ax.set_ylabel('Discontinuity')
    ax.set_title('Discontinuity Distribution by Prediction and Position')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualization saved: {save_path}")
    
    # Statistical tests
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS")
    print("="*60)
    
    # T-test: correct vs incorrect at boundaries
    t_stat, p_val = stats.ttest_ind(correct_boundary, incorrect_boundary)
    print(f"\nBoundary Discontinuity: Correct vs Incorrect")
    print(f"  Correct mean: {np.mean(correct_boundary):.4f} ± {np.std(correct_boundary):.4f}")
    print(f"  Incorrect mean: {np.mean(incorrect_boundary):.4f} ± {np.std(incorrect_boundary):.4f}")
    print(f"  t-statistic: {t_stat:.4f}, p-value: {p_val:.4e}")
    print(f"  {'⚠️  SIGNIFICANT' if p_val < 0.05 else '✓ Not significant'}")
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.std(correct_boundary)**2 + np.std(incorrect_boundary)**2) / 2)
    cohen_d = (np.mean(correct_boundary) - np.mean(incorrect_boundary)) / pooled_std
    print(f"  Cohen's d (effect size): {cohen_d:.4f}")
    
    if abs(cohen_d) < 0.2:
        interpretation = "negligible"
    elif abs(cohen_d) < 0.5:
        interpretation = "small"
    elif abs(cohen_d) < 0.8:
        interpretation = "medium"
    else:
        interpretation = "large"
    
    print(f"  Effect size interpretation: {interpretation}")

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load decision features
    result_path = Path("decision_analysis_2021LA_5k/decision_analysis_results.json")
    with open(result_path) as f:
        data = json.load(f)
    decision_features = data['attribution']['top_k_indices'][:50]
    
    # Load model
    print("Loading model...")
    model_path = Path("models/topk_sae_window_w8_LA_e40_bs14_lr1e-06_saeW0.1_dict4096_k128_window_topk_w8/best_checkpoint_eer_window_topk_w8.pth")
    
    class Args:
        pass
    args = Args()
    args.architecture = 'XLSR'
    args.num_classes = 2
    
    model = WindowTopKModel(
        args=args,
        device=device,
        cp_path='xlsr2_300m.pt',
        use_sae=True,
        use_sparse_features=True,
        sae_dict_size=4096,
        sae_k=128,
        sae_window_size=8,
        sae_weight=0.1
    )
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load data
    database_path = Path("/data/projects/punim2637/nnliang/Datasets/ASVspoof2021_LA_eval")
    _, _, eval_loader = genSpoof_list(
        dir_meta=database_path / 'keys/CM',
        is_train=False,
        is_eval=True
    )
    
    # Run analysis
    results = analyze_correlation_with_errors(
        model, eval_loader, decision_features, device, num_samples=2000
    )
    
    print(f"\nAnalyzed {len(results['correct_predictions'])} correct predictions")
    print(f"Analyzed {len(results['incorrect_predictions'])} incorrect predictions")
    
    # Visualize
    save_dir = Path("boundary_semantic_analysis")
    save_dir.mkdir(exist_ok=True)
    
    visualize_boundary_error_correlation(
        results, 
        save_dir / "boundary_error_correlation.png"
    )
    
    # Save results
    output_path = save_dir / "boundary_analysis_results.json"
    with open(output_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {
            'correct_predictions': [
                {k: (v.tolist() if isinstance(v, np.ndarray) else v) 
                 for k, v in r.items()}
                for r in results['correct_predictions']
            ],
            'incorrect_predictions': [
                {k: (v.tolist() if isinstance(v, np.ndarray) else v) 
                 for k, v in r.items()}
                for r in results['incorrect_predictions']
            ]
        }
        json.dump(serializable_results, f, indent=2)
    
    print(f"Results saved: {output_path}")

if __name__ == '__main__':
    main()
