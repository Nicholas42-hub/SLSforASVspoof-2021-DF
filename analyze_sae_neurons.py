#!/usr/bin/env python
"""
Analyze TopK SAE neurons to evaluate interpretability
- Check feature activation patterns
- Identify discriminative features between bonafide and spoof
- Visualize feature importance
"""

import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from data_utils_SSL import genSpoof_list, Dataset_ASVspoof2019_train
from model import Model
import argparse

# Add fairseq to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 
                                'fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1'))

def load_model(model_path, device, sae_k=128):
    """Load trained model"""
    print(f"Loading model from: {model_path}")
    
    # Check checkpoint to determine actual architecture
    checkpoint = torch.load(model_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Detect if using sparse features by checking classifier input size
    classifier_weight_key = None
    for key in state_dict.keys():
        if 'classifier.1.weight' in key:  # Linear layer after LayerNorm
            classifier_weight_key = key
            break
    
    if classifier_weight_key:
        # Shape is [out_features, in_features] for Linear layer
        classifier_input_dim = state_dict[classifier_weight_key].shape[1]
        use_sparse = (classifier_input_dim == 4096)
    else:
        use_sparse = True  # Default assumption
    
    print(f"  Detected architecture: use_sparse_features={use_sparse}, sae_k={sae_k}")
    
    # Create dummy args
    class DummyArgs:
        pass
    
    # Create model with correct config
    model = Model(
        args=DummyArgs(),
        device=device,
        use_sparse_features=use_sparse,
        sae_dict_size=4096,
        sae_k=sae_k,
        sae_weight=0.1,
        cp_path='xlsr2_300m.pt'
    )
    
    # Remove 'module.' prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)  # Move model to device
    model.eval()
    
    print(f"✓ Model loaded successfully")
    return model


def analyze_feature_statistics(model, dataloader, num_samples=200):
    """
    Analyze SAE feature activation statistics
    Ensures balanced sampling: 50% bonafide, 50% spoof
    """
    print(f"\n{'='*70}")
    print("ANALYZING FEATURE STATISTICS")
    print(f"{'='*70}")
    
    bonafide_features = []
    spoof_features = []
    
    # Calculate target counts for balanced sampling
    target_bonafide = num_samples // 2
    target_spoof = num_samples // 2
    
    print(f"Target: {target_bonafide} bonafide + {target_spoof} spoof = {num_samples} total")
    
    count = 0
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            # Check if we've collected enough samples
            if len(bonafide_features) >= target_bonafide and len(spoof_features) >= target_spoof:
                break
            
            batch_x = batch_x.to(model.device)
            batch_y = batch_y.to(model.device)
            
            # Forward pass with interpretability
            result = model(batch_x, return_sae_loss=False, return_interpretability=True)
            
            # Handle different return formats
            if isinstance(result, tuple):
                if len(result) == 2:
                    output, interp = result
                else:
                    print(f"⚠️  Unexpected return format: {len(result)} values")
                    continue
            else:
                print("⚠️  Model did not return interpretability info")
                return None
            
            if interp is None:
                print("⚠️  Model does not support interpretability analysis")
                return None
            
            # Separate by label (1=bonafide, 0=spoof) with balanced sampling
            for i in range(len(batch_y)):
                feat = interp['avg_activation'][i].cpu()
                if batch_y[i] == 1 and len(bonafide_features) < target_bonafide:
                    bonafide_features.append(feat)
                elif batch_y[i] == 0 and len(spoof_features) < target_spoof:
                    spoof_features.append(feat)
            
            count += len(batch_y)
            if count % 100 == 0:
                print(f"  Processed {count} batches... (bonafide: {len(bonafide_features)}, spoof: {len(spoof_features)})")
    
    if len(bonafide_features) == 0 or len(spoof_features) == 0:
        print("⚠️  Insufficient samples for analysis")
        return None
    
    # Stack features
    bonafide_features = torch.stack(bonafide_features)  # [N, 4096]
    spoof_features = torch.stack(spoof_features)        # [M, 4096]
    
    print(f"\n📊 Sample counts:")
    print(f"  Bonafide: {len(bonafide_features)}")
    print(f"  Spoof: {len(spoof_features)}")
    
    # Calculate statistics
    bonafide_mean = bonafide_features.mean(dim=0)
    spoof_mean = spoof_features.mean(dim=0)
    bonafide_std = bonafide_features.std(dim=0)
    spoof_std = spoof_features.std(dim=0)
    
    # Feature activation rates
    bonafide_active = (bonafide_features > 0).float().mean(dim=0)
    spoof_active = (spoof_features > 0).float().mean(dim=0)
    
    # Overall sparsity
    bonafide_sparsity = (bonafide_features > 0).float().mean()
    spoof_sparsity = (spoof_features > 0).float().mean()
    
    print(f"\n📈 Activation Statistics:")
    print(f"  Bonafide sparsity: {bonafide_sparsity*100:.2f}% (avg {bonafide_sparsity*4096:.1f}/4096 features active)")
    print(f"  Spoof sparsity: {spoof_sparsity*100:.2f}% (avg {spoof_sparsity*4096:.1f}/4096 features active)")
    
    # Find discriminative features
    diff = torch.abs(bonafide_mean - spoof_mean)
    top_discriminative = diff.topk(50)
    
    # Find class-specific features
    bonafide_specific = (bonafide_mean > spoof_mean * 3) & (bonafide_active > 0.3)
    spoof_specific = (spoof_mean > bonafide_mean * 3) & (spoof_active > 0.3)
    
    print(f"\n🎯 Discriminative Features:")
    print(f"  Top 50 discriminative features identified")
    print(f"  Mean discrimination score: {top_discriminative.values.mean():.4f}")
    print(f"  Max discrimination score: {top_discriminative.values.max():.4f}")
    print(f"\n  Bonafide-specific neurons: {bonafide_specific.sum().item()}")
    print(f"  Spoof-specific neurons: {spoof_specific.sum().item()}")
    
    # Check if features are meaningful
    print(f"\n✓ INTERPRETABILITY ASSESSMENT:")
    
    meaningfulness_score = 0
    
    # Criterion 1: Sufficient sparsity (not too dense, not too sparse)
    if 0.01 < bonafide_sparsity < 0.2 and 0.01 < spoof_sparsity < 0.2:
        print(f"  ✓ Good sparsity levels (bonafide: {bonafide_sparsity*100:.1f}%, spoof: {spoof_sparsity*100:.1f}%)")
        meaningfulness_score += 1
    else:
        print(f"  ✗ Sparsity out of ideal range (target: 1-20%)")
    
    # Criterion 2: Class-specific features exist
    if bonafide_specific.sum() > 10 and spoof_specific.sum() > 10:
        print(f"  ✓ Class-specific features found (bonafide: {bonafide_specific.sum()}, spoof: {spoof_specific.sum()})")
        meaningfulness_score += 1
    else:
        print(f"  ✗ Few class-specific features (need more)")
    
    # Criterion 3: Strong discrimination
    if top_discriminative.values.mean() > 0.1:
        print(f"  ✓ Strong discriminative power (score: {top_discriminative.values.mean():.4f})")
        meaningfulness_score += 1
    else:
        print(f"  ✗ Weak discriminative power (score: {top_discriminative.values.mean():.4f})")
    
    # Criterion 4: Feature diversity
    active_features = ((bonafide_active > 0.05) | (spoof_active > 0.05)).sum()
    if active_features > 500:
        print(f"  ✓ Good feature diversity ({active_features}/4096 features used)")
        meaningfulness_score += 1
    else:
        print(f"  ✗ Limited feature diversity ({active_features}/4096 features used)")
    
    print(f"\n{'='*70}")
    print(f"OVERALL SCORE: {meaningfulness_score}/4")
    if meaningfulness_score >= 3:
        print("✓ Model generates MEANINGFUL and INTERPRETABLE neurons")
    elif meaningfulness_score >= 2:
        print("⚠  Model generates PARTIALLY INTERPRETABLE neurons")
    else:
        print("✗ Model neurons have LIMITED INTERPRETABILITY")
    print(f"{'='*70}")
    
    return {
        'bonafide_mean': bonafide_mean,
        'spoof_mean': spoof_mean,
        'bonafide_std': bonafide_std,
        'spoof_std': spoof_std,
        'bonafide_active': bonafide_active,
        'spoof_active': spoof_active,
        'top_discriminative_idx': top_discriminative.indices,
        'top_discriminative_scores': top_discriminative.values,
        'bonafide_specific': bonafide_specific,
        'spoof_specific': spoof_specific,
        'meaningfulness_score': meaningfulness_score
    }


def visualize_features(stats, output_dir='analysis_results'):
    """
    Create visualization plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n📊 Creating visualizations...")
    
    # Plot 1: Feature activation heatmap (top discriminative)
    num_top_features = min(100, len(stats['top_discriminative_idx']))
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))
    
    top_idx = stats['top_discriminative_idx'][:num_top_features].numpy()
    
    bonafide_vals = stats['bonafide_mean'][top_idx].numpy()
    spoof_vals = stats['spoof_mean'][top_idx].numpy()
    
    axes[0].bar(range(num_top_features), bonafide_vals, color='green', alpha=0.7)
    axes[0].set_title(f'Top {num_top_features} Discriminative Features - Bonafide Activation', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Feature Index (sorted by discrimination)')
    axes[0].set_ylabel('Mean Activation')
    axes[0].grid(alpha=0.3)
    
    axes[1].bar(range(num_top_features), spoof_vals, color='red', alpha=0.7)
    axes[1].set_title(f'Top {num_top_features} Discriminative Features - Spoof Activation', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Feature Index (sorted by discrimination)')
    axes[1].set_ylabel('Mean Activation')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plot1_path = os.path.join(output_dir, 'feature_discrimination.png')
    plt.savefig(plot1_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {plot1_path}")
    
    # Plot 2: Activation frequency comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bonafide_freq = stats['bonafide_active'].numpy()
    spoof_freq = stats['spoof_active'].numpy()
    
    sorted_idx = np.argsort(bonafide_freq - spoof_freq)
    
    x = np.arange(len(sorted_idx))
    ax.scatter(x, bonafide_freq[sorted_idx], alpha=0.5, s=1, label='Bonafide', color='green')
    ax.scatter(x, spoof_freq[sorted_idx], alpha=0.5, s=1, label='Spoof', color='red')
    ax.set_title('Feature Activation Frequency: Bonafide vs Spoof', fontsize=14, fontweight='bold')
    ax.set_xlabel('Features (sorted by preference)')
    ax.set_ylabel('Activation Frequency')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plot2_path = os.path.join(output_dir, 'activation_frequency.png')
    plt.savefig(plot2_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {plot2_path}")
    
    # Plot 3: Discrimination score distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    
    scores = stats['top_discriminative_scores'].numpy()
    ax.hist(scores, bins=30, color='blue', alpha=0.7, edgecolor='black')
    ax.axvline(scores.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {scores.mean():.4f}')
    ax.set_title('Distribution of Top 50 Discriminative Scores', fontsize=14, fontweight='bold')
    ax.set_xlabel('Discrimination Score (|bonafide_mean - spoof_mean|)')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plot3_path = os.path.join(output_dir, 'discrimination_scores.png')
    plt.savefig(plot3_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {plot3_path}")
    
    print(f"\n✓ All visualizations saved to: {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description='Analyze TopK SAE neuron interpretability')
    parser.add_argument('--model_path', type=str, 
                        default='models/topk_sae_LA_e40_bs14_lr1e-06_saeW0.1_dict4096_k128_k128_sparse_4096dim/best_checkpoint_eer_k128_sparse_4096dim.pth',
                        help='Path to trained model checkpoint')
    parser.add_argument('--sae_k', type=int, default=128,
                        help='TopK K value (32 or 128)')
    parser.add_argument('--database_path', type=str,
                        default='/data/projects/punim2637/nnliang/Datasets/LA',
                        help='Path to ASVspoof2019 LA dataset')
    parser.add_argument('--num_samples', type=int, default=400,
                        help='Number of samples to analyze')
    parser.add_argument('--output_dir', type=str, default='analysis_results',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.model_path, device, sae_k=args.sae_k)
    
    # Load dataset
    print(f"\nLoading dataset...")
    
    # Use development set for analysis
    d_label_dev, file_dev = genSpoof_list(
        dir_meta=os.path.join(args.database_path, 'ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'),
        is_train=False,
        is_eval=False
    )
    
    print(f"  Development samples: {len(file_dev)}")
    
    # Create a simple args object for Dataset
    class SimpleArgs:
        def __init__(self):
            self.algo = 0  # No augmentation
            self.nBands = 5
            self.minF = 20
            self.maxF = 8000
            self.minBW = 100
            self.maxBW = 1000
            self.minCoeff = 10
            self.maxCoeff = 100
            self.minG = 0
            self.maxG = 0
            self.minBiasLinNonLin = 5
            self.maxBiasLinNonLin = 20
            self.N_f = 5
            self.P = 10
            self.g_sd = 2
            self.SNRmin = 10
            self.SNRmax = 40
    
    simple_args = SimpleArgs()
    
    dev_set = Dataset_ASVspoof2019_train(
        args=simple_args,
        list_IDs=file_dev,
        labels=d_label_dev,
        base_dir=os.path.join(args.database_path, 'ASVspoof2019_LA_dev/'),
        algo=0
    )
    
    dev_loader = DataLoader(
        dev_set,
        batch_size=8,
        shuffle=True,
        num_workers=4
    )
    
    # Analyze features
    stats = analyze_feature_statistics(model, dev_loader, num_samples=args.num_samples)
    
    if stats is not None:
        # Create visualizations
        visualize_features(stats, output_dir=args.output_dir)
        
        # Save numerical results
        results_path = os.path.join(args.output_dir, 'neuron_analysis.txt')
        with open(results_path, 'w') as f:
            f.write("TopK SAE Neuron Analysis Results\n")
            f.write("="*70 + "\n\n")
            f.write(f"Model: {args.model_path}\n")
            f.write(f"Samples analyzed: {args.num_samples}\n\n")
            f.write(f"Meaningfulness Score: {stats['meaningfulness_score']}/4\n\n")
            
            f.write("Top 20 Most Discriminative Features:\n")
            for i, (idx, score) in enumerate(zip(stats['top_discriminative_idx'][:20], 
                                                  stats['top_discriminative_scores'][:20])):
                f.write(f"  {i+1:2d}. Feature {idx:4d}: score={score:.4f}\n")
            
            f.write(f"\nBonafide-specific neurons: {stats['bonafide_specific'].sum().item()}\n")
            f.write(f"Spoof-specific neurons: {stats['spoof_specific'].sum().item()}\n")
        
        print(f"\n✓ Results saved to: {results_path}")


if __name__ == '__main__':
    main()
