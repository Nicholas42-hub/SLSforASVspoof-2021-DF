"""
Compare Temporal Stability Across Different Models

Compares:
- Model 1: Per-timestep TopK (baseline, no window constraint)
- Model 2: Window-based TopK (window_size=8)

This addresses the key research question: 
"Does window-based TopK actually improve temporal stability compared to per-timestep TopK?"
"""

import argparse
import json
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from data_utils_SSL import genSpoof_list, Dataset_ASVspoof2019_train
from model_window_topk import Model


def load_model(model_path, window_size, device):
    """Load a model with specified window_size."""
    class Args:
        pass
    
    model_args = Args()
    model = Model(
        args=model_args,
        device=device,
        use_sae=True,
        use_sparse_features=True,
        sae_dict_size=4096,
        sae_k=128,
        sae_window_size=window_size,  # Key difference!
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Handle DataParallel
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    
    return model


def analyze_model(model, dataloader, num_samples, model_name):
    """Run all temporal analyses on a model."""
    print(f"\n{'='*80}")
    print(f"Analyzing: {model_name}")
    print('='*80)
    
    results = {}
    
    print(f"[{model_name}] 1/3 Temporal stability...", flush=True)
    results['temporal_stability'] = model.analyze_temporal_stability(
        dataloader, num_samples=num_samples
    )
    
    print(f"[{model_name}] 2/3 Feature identity...", flush=True)
    results['feature_identity'] = model.analyze_feature_identity_stability(
        dataloader, num_samples=min(num_samples, 50)
    )
    
    print(f"[{model_name}] 3/3 Failure modes...", flush=True)
    results['failure_modes'] = model.analyze_temporal_failure_modes(
        dataloader, num_samples=min(num_samples, 20), visualize_samples=5
    )
    
    print(f"[{model_name}] ✓ Complete", flush=True)
    
    return results


def plot_comparison(results_dict, output_dir):
    """Generate side-by-side comparison plots."""
    
    model_names = list(results_dict.keys())
    colors = ['#ff7f0e', '#1f77b4', '#2ca02c']
    
    # 1. Jaccard Similarity Comparison
    fig, axes = plt.subplots(1, len(model_names), figsize=(6*len(model_names), 5))
    if len(model_names) == 1:
        axes = [axes]
    
    for idx, (model_name, results) in enumerate(results_dict.items()):
        jaccard = results['temporal_stability']['jaccard_similarity_scores']
        axes[idx].hist(jaccard, bins=30, color=colors[idx], alpha=0.7, edgecolor='black')
        axes[idx].set_xlabel('Jaccard Similarity')
        axes[idx].set_ylabel('Count')
        axes[idx].set_title(f'{model_name}\nMean: {np.mean(jaccard):.3f}')
        axes[idx].axvline(np.mean(jaccard), color='red', linestyle='--')
    
    plt.suptitle('Temporal Coherence Comparison (Higher = Better)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_jaccard.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Feature Lifetime Comparison
    fig, axes = plt.subplots(1, len(model_names), figsize=(6*len(model_names), 5))
    if len(model_names) == 1:
        axes = [axes]
    
    for idx, (model_name, results) in enumerate(results_dict.items()):
        lifetimes = results['temporal_stability']['feature_lifetime_distribution']
        axes[idx].hist(lifetimes, bins=50, color=colors[idx], alpha=0.7, edgecolor='black')
        axes[idx].set_xlabel('Lifetime (timesteps)')
        axes[idx].set_ylabel('Count')
        median_lt = np.median(lifetimes)
        axes[idx].set_title(f'{model_name}\nMedian: {median_lt:.1f}')
        axes[idx].axvline(median_lt, color='red', linestyle='--')
    
    plt.suptitle('Feature Lifetime Distribution (Higher = More Persistent)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_lifetime.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Summary Bar Chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    metrics = ['Jaccard\nSimilarity', 'Index\nOverlap', 'Transient\nRatio', 
               'Flipping\nRate\n(normalized)', 'Median\nLifetime\n(normalized)']
    x = np.arange(len(metrics))
    width = 0.8 / len(model_names)
    
    for idx, (model_name, results) in enumerate(results_dict.items()):
        ts = results['temporal_stability']
        fi = results['feature_identity']
        fm = results['failure_modes']
        
        values = [
            ts['jaccard_similarity_mean'],
            fi['index_overlap_mean'],
            1 - ts['transient_feature_ratio'],  # Flip so higher is better
            1 - min(1.0, fm['flipping_rate_mean'] / 100),  # Normalize and flip
            min(1.0, ts['feature_lifetime_median'] / 50),  # Normalize
        ]
        
        ax.bar(x + idx*width - width*(len(model_names)-1)/2, values, width, 
               label=model_name, color=colors[idx], alpha=0.8)
    
    ax.set_ylabel('Score (Higher = Better Temporal Stability)')
    ax.set_title('Temporal Stability Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_summary.png'), dpi=300)
    plt.close()


def print_comparison_report(results_dict):
    """Print side-by-side comparison report."""
    
    print("\n" + "="*100)
    print("TEMPORAL STABILITY COMPARISON REPORT")
    print("="*100)
    
    model_names = list(results_dict.keys())
    
    # Table header
    print(f"\n{'Metric':<40}", end='')
    for name in model_names:
        print(f"{name:<25}", end='')
    print()
    print("-" * 100)
    
    # Jaccard Similarity
    print(f"{'Jaccard Similarity (↑)':<40}", end='')
    for name in model_names:
        val = results_dict[name]['temporal_stability']['jaccard_similarity_mean']
        print(f"{val:.3f}{' '*21}", end='')
    print()
    
    # Feature Lifetime
    print(f"{'Median Lifetime (↑)':<40}", end='')
    for name in model_names:
        val = results_dict[name]['temporal_stability']['feature_lifetime_median']
        print(f"{val:.1f} timesteps{' '*11}", end='')
    print()
    
    # Transient Ratio
    print(f"{'Transient Ratio (↓)':<40}", end='')
    for name in model_names:
        val = results_dict[name]['temporal_stability']['transient_feature_ratio']
        print(f"{val:.1%}{' '*19}", end='')
    print()
    
    # Index Overlap
    print(f"{'Index Overlap (↑)':<40}", end='')
    for name in model_names:
        val = results_dict[name]['feature_identity']['index_overlap_mean']
        print(f"{val:.3f}{' '*21}", end='')
    print()
    
    # Turnover Rate
    print(f"{'Turnover Rate (↓)':<40}", end='')
    for name in model_names:
        val = results_dict[name]['feature_identity']['index_turnover_mean']
        print(f"{val:.2%}{' '*19}", end='')
    print()
    
    # Flipping Rate
    print(f"{'Flipping Rate (↓)':<40}", end='')
    for name in model_names:
        val = results_dict[name]['failure_modes']['flipping_rate_mean']
        print(f"{val:.2f} changes/ts{' '*9}", end='')
    print()
    
    # Transient Spikes
    print(f"{'Transient Spikes (↓)':<40}", end='')
    for name in model_names:
        val = results_dict[name]['failure_modes']['transient_spike_ratio_mean']
        print(f"{val:.1%}{' '*19}", end='')
    print()
    
    print("\n" + "="*100)
    print("KEY FINDINGS")
    print("="*100)
    
    if len(model_names) >= 2:
        # Compare first two models
        m1, m2 = model_names[0], model_names[1]
        r1, r2 = results_dict[m1], results_dict[m2]
        
        jaccard_diff = r2['temporal_stability']['jaccard_similarity_mean'] - r1['temporal_stability']['jaccard_similarity_mean']
        lifetime_diff = r2['temporal_stability']['feature_lifetime_median'] - r1['temporal_stability']['feature_lifetime_median']
        transient_diff = r1['temporal_stability']['transient_feature_ratio'] - r2['temporal_stability']['transient_feature_ratio']
        
        print(f"\n{m2} vs {m1}:")
        print(f"  • Jaccard similarity: {'+' if jaccard_diff > 0 else ''}{jaccard_diff:.3f} ({jaccard_diff/r1['temporal_stability']['jaccard_similarity_mean']*100:+.1f}%)")
        print(f"  • Median lifetime: {'+' if lifetime_diff > 0 else ''}{lifetime_diff:.1f} timesteps ({lifetime_diff/r1['temporal_stability']['feature_lifetime_median']*100:+.1f}%)")
        print(f"  • Transient reduction: {transient_diff:.1%} fewer transient features")
        
        if jaccard_diff > 0.05:
            print(f"\n✓ {m2} shows SIGNIFICANTLY better temporal coherence")
        elif abs(jaccard_diff) < 0.02:
            print(f"\n~ Both models show SIMILAR temporal stability")
        else:
            print(f"\n✗ {m2} shows WORSE temporal coherence")
    
    print("\n" + "="*100)


def main():
    parser = argparse.ArgumentParser(description='Compare temporal stability across models')
    parser.add_argument('--model1_path', type=str, required=True, help='Path to first model (baseline)')
    parser.add_argument('--model1_name', type=str, default='Per-timestep TopK (w=1)', help='Name for first model')
    parser.add_argument('--model1_window', type=int, default=1, help='Window size for model 1')
    parser.add_argument('--model2_path', type=str, required=True, help='Path to second model')
    parser.add_argument('--model2_name', type=str, default='Window TopK (w=8)', help='Name for second model')
    parser.add_argument('--model2_window', type=int, default=8, help='Window size for model 2')
    parser.add_argument('--database_path', type=str, default='/data/projects/punim2637/nnliang/Datasets/LA')
    parser.add_argument('--protocols_path', type=str, 
                        default='/data/projects/punim2637/nnliang/Datasets/LA/ASVspoof2019_LA_cm_protocols')
    parser.add_argument('--output_dir', type=str, default='temporal_comparison')
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    
    # Load dataset
    print("Loading dataset...")
    class DataArgs:
        pass
    data_args = DataArgs()
    
    d_label_trn, file_train = genSpoof_list(
        dir_meta=os.path.join(args.protocols_path, 'ASVspoof2019.LA.cm.train.trn.txt'),
        is_train=True, is_eval=False
    )
    
    train_set = Dataset_ASVspoof2019_train(
        args=data_args, list_IDs=file_train, labels=d_label_trn,
        base_dir=os.path.join(args.database_path, 'ASVspoof2019_LA_train/'),
        algo=0
    )
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Analyze both models
    results_dict = {}
    
    print(f"\nLoading Model 1: {args.model1_name}...")
    model1 = load_model(args.model1_path, args.model1_window, device)
    results_dict[args.model1_name] = analyze_model(model1, train_loader, args.num_samples, args.model1_name)
    del model1
    torch.cuda.empty_cache()
    
    print(f"\nLoading Model 2: {args.model2_name}...")
    model2 = load_model(args.model2_path, args.model2_window, device)
    results_dict[args.model2_name] = analyze_model(model2, train_loader, args.num_samples, args.model2_name)
    del model2
    torch.cuda.empty_cache()
    
    # Save results
    def convert_to_serializable(obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    results_file = os.path.join(args.output_dir, 'comparison_results.json')
    with open(results_file, 'w') as f:
        json.dump(convert_to_serializable(results_dict), f, indent=2)
    
    # Generate comparison plots
    print("\nGenerating comparison visualizations...")
    plot_comparison(results_dict, args.output_dir)
    
    # Print comparison report
    print_comparison_report(results_dict)
    
    # Save report
    summary_file = os.path.join(args.output_dir, 'comparison_report.txt')
    import sys
    original_stdout = sys.stdout
    with open(summary_file, 'w') as f:
        sys.stdout = f
        print_comparison_report(results_dict)
        sys.stdout = original_stdout
    
    print(f"\n✓ Comparison complete! Results saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
