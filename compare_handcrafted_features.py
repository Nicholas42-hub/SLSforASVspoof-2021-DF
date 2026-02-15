#!/usr/bin/env python3
"""
Compare SAE learned features with handcrafted features (MFCC, spectrogram)
in terms of temporal stability
"""
import numpy as np
import torch
import torchaudio
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import json

from model_window_topk import Model as WindowTopKModel
from data_utils_SSL import genSpoof_list

class TemporalStabilityAnalyzer:
    """Compute temporal stability metrics for any feature representation"""
    
    def compute_jaccard_similarity(self, features, threshold=0.0):
        """
        Compute frame-to-frame Jaccard similarity for sparse features
        
        Args:
            features: [T, D] tensor
            threshold: values > threshold are considered active
        """
        active = (features > threshold).float()
        T = active.shape[0]
        
        jaccard_scores = []
        for t in range(T - 1):
            feat_t = active[t]
            feat_t1 = active[t + 1]
            
            intersection = (feat_t * feat_t1).sum()
            union = ((feat_t + feat_t1) > 0).float().sum()
            
            if union > 0:
                jacc = (intersection / union).item()
            else:
                jacc = 0.0
            
            jaccard_scores.append(jacc)
        
        return np.mean(jaccard_scores) if jaccard_scores else 0.0
    
    def compute_cosine_similarity(self, features):
        """
        Compute frame-to-frame cosine similarity for dense features
        """
        features = features / (torch.norm(features, dim=-1, keepdim=True) + 1e-8)
        
        cos_sims = []
        for t in range(features.shape[0] - 1):
            cos_sim = (features[t] * features[t+1]).sum().item()
            cos_sims.append(cos_sim)
        
        return np.mean(cos_sims) if cos_sims else 0.0
    
    def compute_feature_lifetime(self, features, threshold=0.0):
        """
        Compute average lifetime of active features
        """
        active = (features > threshold).cpu().numpy()
        T, D = active.shape
        
        lifetimes = []
        
        for d in range(D):
            current_lifetime = 0
            for t in range(T):
                if active[t, d]:
                    current_lifetime += 1
                else:
                    if current_lifetime > 0:
                        lifetimes.append(current_lifetime)
                    current_lifetime = 0
            
            # Don't forget the last segment
            if current_lifetime > 0:
                lifetimes.append(current_lifetime)
        
        return np.median(lifetimes) if lifetimes else 0.0
    
    def compute_all_metrics(self, features, threshold=0.0, is_sparse=True):
        """Compute all temporal stability metrics"""
        if is_sparse:
            jaccard = self.compute_jaccard_similarity(features, threshold)
            cosine = self.compute_cosine_similarity(features)
        else:
            jaccard = None  # Not applicable for dense features
            cosine = self.compute_cosine_similarity(features)
        
        lifetime = self.compute_feature_lifetime(features, threshold)
        
        return {
            'jaccard_similarity': jaccard,
            'cosine_similarity': cosine,
            'feature_lifetime': lifetime,
        }

def extract_mfcc_features(audio, sr=16000, n_mfcc=40, hop_length=160):
    """
    Extract MFCC features matching SSL temporal resolution
    """
    # Compute MFCC
    mfcc = librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=n_mfcc, 
        hop_length=hop_length, n_fft=400
    )
    
    # Transpose to [T, D]
    mfcc = mfcc.T
    
    return torch.from_numpy(mfcc).float()

def extract_spectrogram_features(audio, sr=16000, n_mels=80, hop_length=160):
    """
    Extract log mel spectrogram features
    """
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=n_mels,
        hop_length=hop_length, n_fft=400, fmax=8000
    )
    
    # Convert to log scale
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Transpose to [T, D]
    log_mel = log_mel.T
    
    return torch.from_numpy(log_mel).float()

def compare_features_on_samples(
    model, eval_loader, device, num_samples=500
):
    """
    Compare SAE features, MFCC, and spectrogram on multiple samples
    """
    analyzer = TemporalStabilityAnalyzer()
    
    results = {
        'sae_features': [],
        'mfcc_features': [],
        'spectrogram_features': [],
    }
    
    model.eval()
    processed = 0
    
    print("Comparing SAE vs handcrafted features...")
    
    with torch.no_grad():
        for utt_id, audio_path, target in tqdm(eval_loader, total=num_samples):
            if processed >= num_samples:
                break
            
            try:
                # Load audio
                audio, sr = torchaudio.load(audio_path[0])
                audio_np = audio.squeeze().numpy()
                
                # 1. Extract SAE features
                audio_tensor = audio.to(device)
                if audio_tensor.shape[0] > 1:
                    audio_tensor = audio_tensor.mean(dim=0, keepdim=True)
                
                ssl_feats = model.ssl_model(audio_tensor)
                if isinstance(ssl_feats, tuple):
                    ssl_feats = ssl_feats[0]
                sae_activations = model.sae.get_activations(ssl_feats)
                
                # 2. Extract MFCC
                mfcc = extract_mfcc_features(audio_np, sr=sr)
                
                # 3. Extract spectrogram
                spectrogram = extract_spectrogram_features(audio_np, sr=sr)
                
                # Align lengths (take minimum)
                min_len = min(sae_activations.shape[0], mfcc.shape[0], spectrogram.shape[0])
                sae_activations = sae_activations[:min_len]
                mfcc = mfcc[:min_len]
                spectrogram = spectrogram[:min_len]
                
                # Compute metrics
                sae_metrics = analyzer.compute_all_metrics(
                    sae_activations, threshold=0.0, is_sparse=True
                )
                mfcc_metrics = analyzer.compute_all_metrics(
                    mfcc, threshold=-np.inf, is_sparse=False
                )
                spec_metrics = analyzer.compute_all_metrics(
                    spectrogram, threshold=-np.inf, is_sparse=False
                )
                
                results['sae_features'].append(sae_metrics)
                results['mfcc_features'].append(mfcc_metrics)
                results['spectrogram_features'].append(spec_metrics)
                
                processed += 1
                
            except Exception as e:
                print(f"Error processing {audio_path[0]}: {e}")
                continue
    
    return results

def visualize_comparison(results, save_path):
    """
    Create comprehensive comparison visualizations
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Extract data
    sae_cosine = [r['cosine_similarity'] for r in results['sae_features']]
    mfcc_cosine = [r['cosine_similarity'] for r in results['mfcc_features']]
    spec_cosine = [r['cosine_similarity'] for r in results['spectrogram_features']]
    
    sae_jaccard = [r['jaccard_similarity'] for r in results['sae_features'] if r['jaccard_similarity'] is not None]
    
    sae_lifetime = [r['feature_lifetime'] for r in results['sae_features']]
    mfcc_lifetime = [r['feature_lifetime'] for r in results['mfcc_features']]
    spec_lifetime = [r['feature_lifetime'] for r in results['spectrogram_features']]
    
    # Plot 1: Cosine Similarity comparison
    ax = axes[0]
    data_cosine = [sae_cosine, mfcc_cosine, spec_cosine]
    labels_cosine = ['SAE\n(Learned)', 'MFCC\n(Handcrafted)', 'Mel-Spec\n(Handcrafted)']
    
    bp = ax.boxplot(data_cosine, labels=labels_cosine, patch_artist=True)
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    for box, color in zip(bp['boxes'], colors):
        box.set_facecolor(color)
    
    ax.set_ylabel('Cosine Similarity (frame-to-frame)', fontsize=12)
    ax.set_title('Temporal Consistency Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.05])
    
    # Add mean values as text
    for i, data in enumerate(data_cosine):
        mean_val = np.mean(data)
        ax.text(i+1, 0.05, f'μ={mean_val:.3f}', 
                ha='center', fontsize=10, fontweight='bold')
    
    # Plot 2: Feature Lifetime comparison
    ax = axes[1]
    data_lifetime = [sae_lifetime, mfcc_lifetime, spec_lifetime]
    
    bp = ax.boxplot(data_lifetime, labels=labels_cosine, patch_artist=True)
    for box, color in zip(bp['boxes'], colors):
        box.set_facecolor(color)
    
    ax.set_ylabel('Feature Lifetime (frames)', fontsize=12)
    ax.set_title('Feature Persistence Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add mean values
    for i, data in enumerate(data_lifetime):
        mean_val = np.mean(data)
        ax.text(i+1, 1, f'μ={mean_val:.1f}', 
                ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nComparison visualization saved: {save_path}")
    
    # Print statistics
    print("\n" + "="*60)
    print("FEATURE COMPARISON STATISTICS")
    print("="*60)
    
    print("\n📊 Cosine Similarity (Temporal Consistency):")
    print(f"  SAE (Learned):      {np.mean(sae_cosine):.4f} ± {np.std(sae_cosine):.4f}")
    print(f"  MFCC (Handcrafted): {np.mean(mfcc_cosine):.4f} ± {np.std(mfcc_cosine):.4f}")
    print(f"  Mel-Spec:           {np.mean(spec_cosine):.4f} ± {np.std(spec_cosine):.4f}")
    
    print("\n📊 Jaccard Similarity (Sparse Features):")
    print(f"  SAE (Learned):      {np.mean(sae_jaccard):.4f} ± {np.std(sae_jaccard):.4f}")
    print(f"  (MFCC/Spectrogram are dense features, Jaccard not applicable)")
    
    print("\n📊 Feature Lifetime (Persistence):")
    print(f"  SAE (Learned):      {np.mean(sae_lifetime):.2f} ± {np.std(sae_lifetime):.2f} frames")
    print(f"  MFCC (Handcrafted): {np.mean(mfcc_lifetime):.2f} ± {np.std(mfcc_lifetime):.2f} frames")
    print(f"  Mel-Spec:           {np.mean(spec_lifetime):.2f} ± {np.std(spec_lifetime):.2f} frames")
    
    # Interpretation
    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    
    if np.mean(sae_cosine) > np.mean(mfcc_cosine):
        diff = (np.mean(sae_cosine) - np.mean(mfcc_cosine)) / np.mean(mfcc_cosine) * 100
        print(f"\n✅ SAE features are {diff:.1f}% MORE stable than MFCC")
        print("   → Learned representations capture more temporally consistent patterns")
    else:
        diff = (np.mean(mfcc_cosine) - np.mean(sae_cosine)) / np.mean(sae_cosine) * 100
        print(f"\n⚠️  MFCC features are {diff:.1f}% MORE stable than SAE")
        print("   → Handcrafted features may be smoother but less discriminative")
    
    if np.mean(sae_lifetime) > np.mean(mfcc_lifetime):
        print(f"\n✅ SAE features persist {np.mean(sae_lifetime)/np.mean(mfcc_lifetime):.2f}x longer than MFCC")
        print("   → SAE captures longer-term acoustic patterns")

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
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
    
    # Run comparison
    results = compare_features_on_samples(
        model, eval_loader, device, num_samples=500
    )
    
    # Visualize
    save_dir = Path("feature_comparison_analysis")
    save_dir.mkdir(exist_ok=True)
    
    visualize_comparison(
        results,
        save_dir / "sae_vs_handcrafted_comparison.png"
    )
    
    # Save results
    output_path = save_dir / "comparison_results.json"
    with open(output_path, 'w') as f:
        # Compute summary statistics
        summary = {
            'sae_features': {
                'mean_cosine': float(np.mean([r['cosine_similarity'] for r in results['sae_features']])),
                'mean_jaccard': float(np.mean([r['jaccard_similarity'] for r in results['sae_features'] if r['jaccard_similarity'] is not None])),
                'mean_lifetime': float(np.mean([r['feature_lifetime'] for r in results['sae_features']])),
            },
            'mfcc_features': {
                'mean_cosine': float(np.mean([r['cosine_similarity'] for r in results['mfcc_features']])),
                'mean_lifetime': float(np.mean([r['feature_lifetime'] for r in results['mfcc_features']])),
            },
            'spectrogram_features': {
                'mean_cosine': float(np.mean([r['cosine_similarity'] for r in results['spectrogram_features']])),
                'mean_lifetime': float(np.mean([r['feature_lifetime'] for r in results['spectrogram_features']])),
            }
        }
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved: {output_path}")

if __name__ == '__main__':
    main()
