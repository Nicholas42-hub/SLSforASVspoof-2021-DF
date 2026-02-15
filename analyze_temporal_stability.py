"""
Temporal Stability Analysis for Window-based TopK SAE
Addresses Caren's request for diagnostic analysis of temporal failure modes.

Usage:
    python analyze_temporal_stability.py --model_path <path> --output_dir <dir>
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


def plot_temporal_diagnostics(results, output_dir):
    """Create comprehensive diagnostic visualizations."""
    
    # 1. Feature Lifetime Distribution
    plt.figure(figsize=(10, 6))
    lifetimes = results['temporal_stability']['feature_lifetime_distribution']
    plt.hist(lifetimes, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Feature Lifetime (timesteps)')
    plt.ylabel('Count')
    plt.title('Feature Lifetime Distribution\n(Shows prevalence of transient vs. persistent features)')
    plt.axvline(np.median(lifetimes), color='r', linestyle='--', label=f'Median: {np.median(lifetimes):.1f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_lifetime_distribution.png'), dpi=300)
    plt.close()
    
    # 2. Jaccard Similarity (Window-to-Window Coherence)
    plt.figure(figsize=(10, 6))
    jaccard_scores = results['temporal_stability']['jaccard_similarity_scores']
    plt.hist(jaccard_scores, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Jaccard Similarity (consecutive windows)')
    plt.ylabel('Count')
    plt.title('Temporal Coherence: Window-to-Window Feature Overlap\n(Lower = more feature instability)')
    plt.axvline(np.mean(jaccard_scores), color='r', linestyle='--', 
                label=f'Mean: {np.mean(jaccard_scores):.3f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'temporal_coherence.png'), dpi=300)
    plt.close()
    
    # 3. Feature Index Overlap Between Windows
    plt.figure(figsize=(10, 6))
    overlap_dist = results['feature_identity']['index_overlap_distribution']
    plt.hist(overlap_dist, bins=30, edgecolor='black', alpha=0.7, color='orange')
    plt.xlabel('Index Overlap Ratio')
    plt.ylabel('Count')
    plt.title('Feature Identity Stability\n(Ratio of shared feature indices in consecutive windows)')
    plt.axvline(np.mean(overlap_dist), color='r', linestyle='--', 
                label=f'Mean: {np.mean(overlap_dist):.3f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_identity_stability.png'), dpi=300)
    plt.close()
    
    # 4. Feature Flipping Rates
    plt.figure(figsize=(10, 6))
    flipping_rates = results['failure_modes']['flipping_rates']
    plt.hist(flipping_rates, bins=30, edgecolor='black', alpha=0.7, color='red')
    plt.xlabel('Features Changed per Timestep')
    plt.ylabel('Count')
    plt.title('Feature Flipping Rate\n(Number of feature indices changing per timestep)')
    plt.axvline(np.mean(flipping_rates), color='darkred', linestyle='--', 
                label=f'Mean: {np.mean(flipping_rates):.2f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_flipping_rate.png'), dpi=300)
    plt.close()
    
    # 5. Transient vs Persistent Features
    plt.figure(figsize=(8, 8))
    transient_ratio = results['temporal_stability']['transient_feature_ratio']
    persistent_ratio = 1 - transient_ratio
    plt.pie([transient_ratio, persistent_ratio], 
            labels=['Transient (<window_size)', 'Persistent (>=window_size)'],
            autopct='%1.1f%%', startangle=90, colors=['#ff9999', '#66b3ff'])
    plt.title('Transient vs. Persistent Feature Activations')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'transient_vs_persistent.png'), dpi=300)
    plt.close()
    
    # 6. Example Trace Visualization
    if results['failure_modes']['detailed_traces']:
        trace = results['failure_modes']['detailed_traces'][0]
        plt.figure(figsize=(14, 6))
        
        # Plot active feature indices over time
        timesteps = []
        feature_indices = []
        for t, indices in enumerate(trace['active_indices_per_timestep']):
            for idx in indices:
                timesteps.append(t)
                feature_indices.append(idx)
        
        plt.scatter(timesteps, feature_indices, alpha=0.5, s=10)
        plt.xlabel('Timestep')
        plt.ylabel('Feature Index')
        plt.title('Feature Activation Pattern Over Time (Example Sample)\n(Vertical scatter = feature instability)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'example_activation_trace.png'), dpi=300)
        plt.close()


def print_summary_report(results):
    """Print a comprehensive summary report."""
    
    print("\n" + "="*80)
    print("TEMPORAL STABILITY ANALYSIS REPORT")
    print("="*80)
    
    print("\n1. FEATURE LIFETIME ANALYSIS")
    print("-" * 80)
    ts = results['temporal_stability']
    print(f"   Mean Lifetime:     {ts['feature_lifetime_mean']:.2f} timesteps")
    print(f"   Median Lifetime:   {ts['feature_lifetime_median']:.2f} timesteps")
    print(f"   Std Dev:           {ts['feature_lifetime_std']:.2f}")
    print(f"   Transient Ratio:   {ts['transient_feature_ratio']:.2%}")
    print(f"   → Interpretation: {'HIGH instability - features are short-lived' if ts['feature_lifetime_mean'] < 5 else 'Moderate temporal persistence'}")
    
    print("\n2. TEMPORAL COHERENCE (Window-to-Window)")
    print("-" * 80)
    print(f"   Jaccard Similarity: {ts['jaccard_similarity_mean']:.3f}")
    print(f"   Feature Changes/Window: {ts['feature_flipping_rate_mean']:.2f}")
    print(f"   → Interpretation: {'LOW coherence - features change frequently' if ts['jaccard_similarity_mean'] < 0.5 else 'Moderate temporal coherence'}")
    
    print("\n3. FEATURE IDENTITY STABILITY")
    print("-" * 80)
    fi = results['feature_identity']
    print(f"   Index Overlap:     {fi['index_overlap_mean']:.3f} ± {fi['index_overlap_std']:.3f}")
    print(f"   Turnover Rate:     {fi['index_turnover_mean']:.2%}")
    print(f"   Expected k:        {fi['expected_k']}")
    print(f"   Actual Active:     {fi['active_features_per_window_mean']:.1f} ± {fi['active_features_per_window_std']:.1f}")
    print(f"   → Interpretation: {'HIGH index flipping' if fi['index_overlap_mean'] < 0.5 else 'Stable feature indices'}")
    
    print("\n4. FAILURE MODES")
    print("-" * 80)
    fm = results['failure_modes']
    print(f"   Flipping Rate:     {fm['flipping_rate_mean']:.2f} ± {fm['flipping_rate_std']:.2f} changes/timestep")
    print(f"   Transient Spikes:  {fm['transient_spike_ratio_mean']:.2%} of activations")
    print(f"   Repr. Variance:    {fm['representation_variance_mean']:.2f}")
    print(f"   Samples Analyzed:  {fm['num_samples_analyzed']}")
    
    print("\n5. KEY FINDINGS FOR PAPER")
    print("="*80)
    
    # Generate concrete findings based on metrics
    findings = []
    
    if ts['feature_lifetime_mean'] < 5:
        findings.append(f"✗ Short feature lifetimes (avg {ts['feature_lifetime_mean']:.1f} timesteps) indicate temporal instability")
    
    if ts['transient_feature_ratio'] > 0.5:
        findings.append(f"✗ {ts['transient_feature_ratio']:.1%} of features are transient (short-lived)")
    
    if ts['jaccard_similarity_mean'] < 0.5:
        findings.append(f"✗ Low temporal coherence (Jaccard={ts['jaccard_similarity_mean']:.3f}) shows features change rapidly")
    
    if fi['index_overlap_mean'] < 0.5:
        findings.append(f"✗ Feature index instability (overlap={fi['index_overlap_mean']:.3f}) suggests semantic inconsistency")
    
    if fm['flipping_rate_mean'] > 50:
        findings.append(f"✗ High flipping rate ({fm['flipping_rate_mean']:.0f} changes/timestep) indicates feature identity instability")
    
    for i, finding in enumerate(findings, 1):
        print(f"{i}. {finding}")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description='Analyze temporal stability of window-based TopK SAE')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--database_path', type=str, default='Datasets/LA', help='Path to ASVspoof dataset')
    parser.add_argument('--protocols_path', type=str, default='Datasets/LA/ASVspoof2019_LA_cm_protocols', 
                        help='Path to protocol files')
    parser.add_argument('--output_dir', type=str, default='temporal_stability_analysis', 
                        help='Output directory for results')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples to analyze')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading model...")
    device = torch.device(args.device)
    
    # Load model (adjust parameters based on your trained model)
    class Args:
        def __init__(self):
            pass
    
    model_args = Args()
    model = Model(
        args=model_args,
        device=device,
        use_sae=True,
        use_sparse_features=True,
        sae_dict_size=4096,
        sae_k=128,
        sae_window_size=8,
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Handle DataParallel wrapper (remove 'module.' prefix if present)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k  # remove 'module.' prefix
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    
    model.eval()
    
    print("Loading dataset...")
    
    # Create args object for dataset
    class DataArgs:
        def __init__(self):
            pass
    data_args = DataArgs()
    
    # Load evaluation data
    d_label_trn, file_train = genSpoof_list(
        dir_meta=os.path.join(args.protocols_path, 'ASVspoof2019.LA.cm.train.trn.txt'),
        is_train=True,
        is_eval=False
    )
    
    train_set = Dataset_ASVspoof2019_train(
        args=data_args,
        list_IDs=file_train,
        labels=d_label_trn,
        base_dir=os.path.join(args.database_path, 'ASVspoof2019_LA_train/'),
        algo=0  # No augmentation
    )
    
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    print(f"\nAnalyzing temporal stability on {args.num_samples} samples...")
    print("This will take a few minutes...\n")
    import sys
    sys.stdout.flush()
    
    # Run analyses
    results = {}
    
    print("1/3 Analyzing temporal stability...", flush=True)
    results['temporal_stability'] = model.analyze_temporal_stability(
        train_loader, 
        num_samples=args.num_samples
    )
    print("✓ Temporal stability analysis complete", flush=True)
    
    print("2/3 Analyzing feature identity stability...", flush=True)
    results['feature_identity'] = model.analyze_feature_identity_stability(
        train_loader,
        num_samples=min(args.num_samples, 50)
    )
    print("✓ Feature identity analysis complete", flush=True)
    
    print("3/3 Analyzing failure modes...", flush=True)
    results['failure_modes'] = model.analyze_temporal_failure_modes(
        train_loader,
        num_samples=min(args.num_samples, 20),
        visualize_samples=5
    )
    print("✓ Failure mode analysis complete", flush=True)
    
    # Save results
    results_file = os.path.join(args.output_dir, 'temporal_stability_results.json')
    
    # Convert tensors to lists for JSON serialization
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
    
    results_serializable = convert_to_serializable(results)
    
    with open(results_file, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_temporal_diagnostics(results, args.output_dir)
    print(f"Visualizations saved to {args.output_dir}/")
    
    # Print summary report
    print_summary_report(results)
    
    # Save summary report to file
    summary_file = os.path.join(args.output_dir, 'summary_report.txt')
    import sys
    original_stdout = sys.stdout
    with open(summary_file, 'w') as f:
        sys.stdout = f
        print_summary_report(results)
        sys.stdout = original_stdout
    
    print(f"\nSummary report saved to {summary_file}")
    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()
