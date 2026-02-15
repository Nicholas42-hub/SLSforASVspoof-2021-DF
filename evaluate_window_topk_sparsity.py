#!/usr/bin/env python3
"""
Comprehensive sparsity evaluation for Window TopK SAE model
Analyzes sparse feature embeddings and temporal patterns
"""

import torch
import numpy as np
import sys
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import json
from collections import defaultdict

# Add fairseq to path
sys.path.insert(0, 'fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1')

from model_window_topk import Model
from data_utils_SSL import genSpoof_list, Dataset_ASVspoof2019_train


def analyze_window_topk_sparsity(model, data_loader, device, num_batches=100):
    """
    Comprehensive sparsity analysis for Window TopK SAE
    
    Args:
        model: The trained model with window topk SAE
        data_loader: DataLoader for the dataset
        device: torch device
        num_batches: Number of batches to analyze
    
    Returns:
        dict: Comprehensive sparsity statistics
    """
    model.eval()
    
    # Storage for analysis
    all_sparse_features = []
    per_sample_stats = []
    per_window_stats = []
    feature_usage_count = None
    total_timesteps = 0
    
    print(f"\n{'='*80}")
    print(f"WINDOW TOPK SAE SPARSITY ANALYSIS")
    print(f"{'='*80}\n")
    
    # Get model parameters
    sae = model.module.sae if hasattr(model, 'module') else model.sae
    dict_size = sae.dict_size
    k_value = sae.k.item()
    window_size = sae.window_size
    
    print(f"📊 SAE Configuration:")
    print(f"   Dictionary size: {dict_size}")
    print(f"   TopK K value: {k_value}")
    print(f"   Window size: {window_size}")
    print(f"   Theoretical sparsity: {k_value}/{dict_size} = {k_value/dict_size*100:.2f}%\n")
    
    feature_usage_count = torch.zeros(dict_size, device=device)
    
    print(f"Analyzing {num_batches} batches...")
    
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(tqdm(data_loader)):
            if i >= num_batches:
                break
            
            batch_x = batch_x.to(device)
            batch_size = batch_x.size(0)
            
            # Extract SSL features
            if hasattr(model, 'module'):
                ssl_features = model.module.ssl_model.extract_feat(batch_x)  # [B, T, 1024]
            else:
                ssl_features = model.ssl_model.extract_feat(batch_x)
            
            B, T, C = ssl_features.shape
            total_timesteps += B * T
            
            # Encode with window topk SAE
            sparse_features = sae.encode(ssl_features)  # [B, T, dict_size]
            
            # Calculate per-sample statistics
            for b in range(B):
                sample_sparse = sparse_features[b]  # [T, dict_size]
                
                # Active features per timestep
                active_per_timestep = (sample_sparse > 0).sum(dim=-1)  # [T]
                
                # Per window analysis
                num_windows = T // window_size
                for w in range(num_windows):
                    window_start = w * window_size
                    window_end = window_start + window_size
                    window_sparse = sample_sparse[window_start:window_end]  # [window_size, dict_size]
                    
                    # Features active in this window
                    window_active_mask = (window_sparse > 0).any(dim=0)  # [dict_size]
                    num_active_in_window = window_active_mask.sum().item()
                    
                    per_window_stats.append({
                        'num_active_features': num_active_in_window,
                        'avg_activation_value': window_sparse[window_sparse > 0].mean().item() if window_sparse.sum() > 0 else 0,
                        'sparsity_ratio': num_active_in_window / dict_size
                    })
                
                per_sample_stats.append({
                    'avg_active_per_timestep': active_per_timestep.float().mean().item(),
                    'std_active_per_timestep': active_per_timestep.float().std().item(),
                    'min_active': active_per_timestep.min().item(),
                    'max_active': active_per_timestep.max().item(),
                    'total_unique_features': (sample_sparse > 0).any(dim=0).sum().item()
                })
            
            # Accumulate feature usage across dataset
            active_mask = (sparse_features > 0)  # [B, T, dict_size]
            feature_usage_count += active_mask.sum(dim=(0, 1))  # [dict_size]
            
            # Store some samples for visualization
            if i < 5:  # Store first 5 batches
                all_sparse_features.append(sparse_features.cpu().numpy())
    
    # ===========================
    # COMPUTE STATISTICS
    # ===========================
    
    stats = {}
    
    # 1. Per-timestep sparsity
    per_timestep_active = [s['avg_active_per_timestep'] for s in per_sample_stats]
    stats['avg_active_per_timestep'] = np.mean(per_timestep_active)
    stats['std_active_per_timestep'] = np.std(per_timestep_active)
    stats['actual_sparsity_percent'] = stats['avg_active_per_timestep'] / dict_size * 100
    
    # 2. Per-window sparsity
    window_active = [s['num_active_features'] for s in per_window_stats]
    stats['avg_active_per_window'] = np.mean(window_active)
    stats['std_active_per_window'] = np.std(window_active)
    stats['window_sparsity_percent'] = stats['avg_active_per_window'] / dict_size * 100
    
    # 3. Feature usage (dictionary utilization)
    feature_usage = feature_usage_count.cpu().numpy()
    stats['total_timesteps_analyzed'] = total_timesteps
    stats['features_ever_active'] = (feature_usage > 0).sum()
    stats['features_never_active'] = (feature_usage == 0).sum()
    stats['dict_utilization_percent'] = stats['features_ever_active'] / dict_size * 100
    
    # 4. Feature frequency distribution
    stats['feature_activation_mean'] = feature_usage.mean()
    stats['feature_activation_std'] = feature_usage.std()
    stats['feature_activation_median'] = np.median(feature_usage)
    stats['feature_activation_max'] = feature_usage.max()
    stats['feature_activation_min'] = feature_usage[feature_usage > 0].min() if (feature_usage > 0).any() else 0
    
    # 5. Temporal coherence (consistency across time)
    unique_features = [s['total_unique_features'] for s in per_sample_stats]
    stats['avg_unique_features_per_sample'] = np.mean(unique_features)
    stats['temporal_reuse_ratio'] = stats['avg_active_per_timestep'] / stats['avg_unique_features_per_sample'] if stats['avg_unique_features_per_sample'] > 0 else 0
    
    # 6. Activation value statistics
    window_activation_values = [s['avg_activation_value'] for s in per_window_stats if s['avg_activation_value'] > 0]
    if window_activation_values:
        stats['avg_activation_value'] = np.mean(window_activation_values)
        stats['std_activation_value'] = np.std(window_activation_values)
    
    # 7. Comparison with theoretical K
    stats['k_value'] = k_value
    stats['window_size'] = window_size
    stats['dict_size'] = dict_size
    stats['theoretical_vs_actual_ratio'] = stats['avg_active_per_window'] / k_value
    
    return stats, feature_usage, all_sparse_features, per_sample_stats, per_window_stats


def print_statistics(stats):
    """Print formatted statistics"""
    print(f"\n{'='*80}")
    print(f"SPARSITY ANALYSIS RESULTS")
    print(f"{'='*80}\n")
    
    print(f"📋 Model Configuration:")
    print(f"   Dictionary Size: {stats['dict_size']}")
    print(f"   TopK K: {stats['k_value']}")
    print(f"   Window Size: {stats['window_size']}")
    print(f"   Total timesteps analyzed: {stats['total_timesteps_analyzed']}\n")
    
    print(f"🎯 Sparsity Metrics:")
    print(f"   Avg active features per timestep: {stats['avg_active_per_timestep']:.2f} ± {stats['std_active_per_timestep']:.2f}")
    print(f"   Avg active features per window: {stats['avg_active_per_window']:.2f} ± {stats['std_active_per_window']:.2f}")
    print(f"   Actual sparsity: {stats['actual_sparsity_percent']:.2f}%")
    print(f"   Window sparsity: {stats['window_sparsity_percent']:.2f}%")
    print(f"   Theoretical/actual ratio: {stats['theoretical_vs_actual_ratio']:.2f}x\n")
    
    print(f"📚 Dictionary Utilization:")
    print(f"   Features ever active: {stats['features_ever_active']}/{stats['dict_size']} ({stats['dict_utilization_percent']:.2f}%)")
    print(f"   Features never active (dead): {stats['features_never_active']}/{stats['dict_size']} ({stats['features_never_active']/stats['dict_size']*100:.2f}%)\n")
    
    print(f"📊 Feature Activation Distribution:")
    print(f"   Mean activations per feature: {stats['feature_activation_mean']:.2f}")
    print(f"   Std activations per feature: {stats['feature_activation_std']:.2f}")
    print(f"   Median activations per feature: {stats['feature_activation_median']:.0f}")
    print(f"   Max activations (most used feature): {stats['feature_activation_max']:.0f}")
    if stats.get('feature_activation_min'):
        print(f"   Min activations (least used active): {stats['feature_activation_min']:.0f}\n")
    
    print(f"⏱️  Temporal Coherence:")
    print(f"   Avg unique features per sample: {stats['avg_unique_features_per_sample']:.2f}")
    print(f"   Temporal reuse ratio: {stats['temporal_reuse_ratio']:.2f}x")
    print(f"   (Higher ratio = features reused across timesteps)\n")
    
    if 'avg_activation_value' in stats:
        print(f"💪 Activation Strength:")
        print(f"   Avg activation value: {stats['avg_activation_value']:.4f} ± {stats['std_activation_value']:.4f}\n")
    
    print(f"{'='*80}\n")


def create_visualizations(stats, feature_usage, output_dir='analysis_results_window_topk'):
    """Create visualization plots"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Feature usage histogram
    plt.figure(figsize=(12, 6))
    plt.hist(feature_usage[feature_usage > 0], bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Number of Activations', fontsize=12)
    plt.ylabel('Number of Features', fontsize=12)
    plt.title(f'Feature Usage Distribution (Window TopK)\nDictionary Utilization: {stats["dict_utilization_percent"]:.2f}%', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_usage_histogram.png', dpi=150)
    plt.close()
    print(f"✅ Saved: {output_dir}/feature_usage_histogram.png")
    
    # 2. Feature usage log scale
    plt.figure(figsize=(12, 6))
    active_features = feature_usage[feature_usage > 0]
    plt.hist(np.log10(active_features + 1), bins=50, edgecolor='black', alpha=0.7, color='orange')
    plt.xlabel('Log10(Number of Activations + 1)', fontsize=12)
    plt.ylabel('Number of Features', fontsize=12)
    plt.title('Feature Usage Distribution (Log Scale)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_usage_log_histogram.png', dpi=150)
    plt.close()
    print(f"✅ Saved: {output_dir}/feature_usage_log_histogram.png")
    
    # 3. Top features
    top_n = 50
    top_indices = np.argsort(feature_usage)[-top_n:]
    top_values = feature_usage[top_indices]
    
    plt.figure(figsize=(14, 6))
    plt.bar(range(top_n), top_values, color='steelblue', edgecolor='black')
    plt.xlabel('Feature Rank', fontsize=12)
    plt.ylabel('Activation Count', fontsize=12)
    plt.title(f'Top {top_n} Most Activated Features', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/top_features.png', dpi=150)
    plt.close()
    print(f"✅ Saved: {output_dir}/top_features.png")
    
    # 4. Sparsity comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    categories = ['Theoretical\n(K value)', 'Actual Per\nTimestep', 'Actual Per\nWindow']
    values = [stats['k_value'], stats['avg_active_per_timestep'], stats['avg_active_per_window']]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    bars = ax.bar(categories, values, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Number of Active Features', fontsize=12)
    ax.set_title('Sparsity Comparison: Theoretical vs Actual', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/sparsity_comparison.png', dpi=150)
    plt.close()
    print(f"✅ Saved: {output_dir}/sparsity_comparison.png")


def main():
    # Configuration
    checkpoint_path = 'models/topk_sae_window_w8_LA_e40_bs14_lr1e-06_saeW0.1_dict4096_k128_window_topk_w8/best_checkpoint_eer_window_topk_w8.pth'
    database_path = 'database/database_asvspoof.pkl'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"\n📂 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Initialize model
    class Args:
        def __init__(self):
            self.num_epochs = 40
            self.batch_size = 14
            self.lr = 1e-6
    
    args = Args()
    
    model = Model(
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
    
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model = model.to(device)
    model.eval()
    print("✅ Model loaded successfully\n")
    
    # Load dataset
    print(f"📂 Loading dataset: {database_path}")
    d_label_trn, file_train = genSpoof_list(
        dir_meta='/data/projects/punim2637/nnliang/Datasets/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt',
        is_train=True,
        is_eval=False
    )
    
    train_set = Dataset_ASVspoof2019_train(file_train, d_label_trn)
    train_loader = DataLoader(train_set, batch_size=8, shuffle=False, num_workers=0)
    print(f"✅ Dataset loaded: {len(train_set)} samples\n")
    
    # Run analysis
    stats, feature_usage, sparse_samples, per_sample_stats, per_window_stats = analyze_window_topk_sparsity(
        model, train_loader, device, num_batches=100
    )
    
    # Print results
    print_statistics(stats)
    
    # Create visualizations
    create_visualizations(stats, feature_usage)
    
    # Save detailed results
    output_dir = 'analysis_results_window_topk'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save statistics
    with open(f'{output_dir}/sparsity_stats.json', 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        stats_serializable = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                             for k, v in stats.items()}
        json.dump(stats_serializable, f, indent=2)
    print(f"✅ Saved: {output_dir}/sparsity_stats.json")
    
    # Save feature usage
    np.save(f'{output_dir}/feature_usage.npy', feature_usage)
    print(f"✅ Saved: {output_dir}/feature_usage.npy")
    
    print(f"\n{'='*80}")
    print(f"✅ Analysis complete! Results saved to: {output_dir}/")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
