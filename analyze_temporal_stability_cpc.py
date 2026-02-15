"""
Temporal Stability Analysis for CPC-based Window TopK SAE
Adapted from analyze_temporal_stability.py to work with model_cpc.py

Usage:
    python analyze_temporal_stability_cpc.py --model_path <path> --output_dir <dir>
"""

import argparse
import json
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from data_utils_SSL import genSpoof_list, Dataset_ASVspoof2019_train
from model_cpc import Model, AutoEncoderTopK


def analyze_temporal_stability_manual(model, dataloader, num_samples=100):
    """
    Manually analyze temporal stability by extracting SAE features.
    Since CPC model doesn't have built-in temporal analysis, we implement it here.
    """
    model.eval()
    device = next(model.parameters()).device
    
    all_lifetimes = []
    all_jaccard_scores = []
    all_feature_changes = []
    
    sample_count = 0
    
    with torch.no_grad():
        for batch_x, _ in dataloader:
            if sample_count >= num_samples:
                break
                
            batch_x = batch_x.to(device)
            
            # Forward pass through SSL model
            ssl_features = model.ssl_model.extract_feat(batch_x)  # [B, T, D]
            
            # Get SAE sparse features
            if model.use_sae:
                sae_features, sae_loss = model.sae(ssl_features)  # [B, T, dict_size]
            else:
                continue
            
            # Analyze each sample in batch
            for sample_idx in range(sae_features.shape[0]):
                if sample_count >= num_samples:
                    break
                    
                feature_matrix = sae_features[sample_idx]  # [T, dict_size]
                T = feature_matrix.shape[0]
                
                # Get top-k indices per timestep
                topk_indices = []
                for t in range(T):
                    _, indices = torch.topk(feature_matrix[t], k=model.sae.k, dim=0)
                    topk_indices.append(set(indices.cpu().tolist()))
                
                # Compute Jaccard similarity between consecutive timesteps
                for t in range(T - 1):
                    current_set = topk_indices[t]
                    next_set = topk_indices[t + 1]
                    intersection = len(current_set & next_set)
                    union = len(current_set | next_set)
                    jaccard = intersection / union if union > 0 else 0
                    all_jaccard_scores.append(jaccard)
                    
                    # Count feature changes
                    changes = len(current_set ^ next_set)  # symmetric difference
                    all_feature_changes.append(changes)
                
                # Compute feature lifetimes
                # Track how long each feature stays active
                feature_lifetimes_dict = {}
                
                for t, indices_set in enumerate(topk_indices):
                    for idx in indices_set:
                        if idx not in feature_lifetimes_dict:
                            feature_lifetimes_dict[idx] = {'start': t, 'segments': []}
                        
                        # Check if this is continuation or new segment
                        if feature_lifetimes_dict[idx]['segments'] and \
                           feature_lifetimes_dict[idx]['segments'][-1]['end'] == t - 1:
                            # Continue current segment
                            feature_lifetimes_dict[idx]['segments'][-1]['end'] = t
                        else:
                            # Start new segment
                            feature_lifetimes_dict[idx]['segments'].append({'start': t, 'end': t})
                
                # Extract all segment lifetimes
                for idx, data in feature_lifetimes_dict.items():
                    for segment in data['segments']:
                        lifetime = segment['end'] - segment['start'] + 1
                        all_lifetimes.append(lifetime)
                
                sample_count += 1
    
    # Compute statistics
    all_lifetimes = np.array(all_lifetimes)
    all_jaccard_scores = np.array(all_jaccard_scores)
    all_feature_changes = np.array(all_feature_changes)
    
    transient_count = np.sum(all_lifetimes == 1)
    transient_ratio = transient_count / len(all_lifetimes) if len(all_lifetimes) > 0 else 0
    
    results = {
        'feature_lifetime_distribution': all_lifetimes.tolist(),
        'feature_lifetime_mean': float(np.mean(all_lifetimes)),
        'feature_lifetime_median': float(np.median(all_lifetimes)),
        'feature_lifetime_std': float(np.std(all_lifetimes)),
        'transient_feature_ratio': float(transient_ratio),
        'jaccard_similarity_scores': all_jaccard_scores.tolist(),
        'jaccard_similarity_mean': float(np.mean(all_jaccard_scores)),
        'jaccard_similarity_std': float(np.std(all_jaccard_scores)),
        'feature_flipping_rate_mean': float(np.mean(all_feature_changes)),
        'feature_flipping_rate_std': float(np.std(all_feature_changes)),
    }
    
    return results


def analyze_feature_identity_stability(model, dataloader, num_samples=50):
    """Analyze whether the same feature indices are selected over time."""
    model.eval()
    device = next(model.parameters()).device
    
    all_index_overlaps = []
    all_turnover_rates = []
    all_active_counts = []
    
    sample_count = 0
    
    with torch.no_grad():
        for batch_x, _ in dataloader:
            if sample_count >= num_samples:
                break
                
            batch_x = batch_x.to(device)
            ssl_features = model.ssl_model.extract_feat(batch_x)
            
            if model.use_sae:
                sae_features, _ = model.sae(ssl_features)
            else:
                continue
            
            for sample_idx in range(sae_features.shape[0]):
                if sample_count >= num_samples:
                    break
                    
                feature_matrix = sae_features[sample_idx]
                T = feature_matrix.shape[0]
                
                topk_indices = []
                for t in range(T):
                    _, indices = torch.topk(feature_matrix[t], k=model.sae.k, dim=0)
                    topk_indices.append(set(indices.cpu().tolist()))
                
                for t in range(T - 1):
                    current = topk_indices[t]
                    next_t = topk_indices[t + 1]
                    
                    overlap = len(current & next_t)
                    k_val = int(model.sae.k) if hasattr(model.sae.k, 'item') else model.sae.k
                    overlap_ratio = overlap / k_val
                    all_index_overlaps.append(float(overlap_ratio))
                    
                    turnover = (k_val - overlap) / k_val
                    all_turnover_rates.append(float(turnover))
                    
                    all_active_counts.append(int(len(current)))
                
                sample_count += 1
    
    # Convert to numpy arrays
    all_index_overlaps = np.array(all_index_overlaps)
    all_turnover_rates = np.array(all_turnover_rates)
    all_active_counts = np.array(all_active_counts)
    
    k_val = int(model.sae.k) if hasattr(model.sae.k, 'item') else int(model.sae.k)
    
    return {
        'index_overlap_distribution': all_index_overlaps.tolist(),
        'index_overlap_mean': float(np.mean(all_index_overlaps)),
        'index_overlap_std': float(np.std(all_index_overlaps)),
        'index_turnover_mean': float(np.mean(all_turnover_rates)),
        'expected_k': k_val,
        'active_features_per_window_mean': float(np.mean(all_active_counts)),
        'active_features_per_window_std': float(np.std(all_active_counts)),
    }


def analyze_temporal_failure_modes(model, dataloader, num_samples=20):
    """Analyze specific failure patterns."""
    model.eval()
    device = next(model.parameters()).device
    
    all_flipping_rates = []
    all_transient_spikes = []
    all_repr_variances = []
    
    sample_count = 0
    
    with torch.no_grad():
        for batch_x, _ in dataloader:
            if sample_count >= num_samples:
                break
                
            batch_x = batch_x.to(device)
            ssl_features = model.ssl_model.extract_feat(batch_x)
            
            if model.use_sae:
                sae_features, _ = model.sae(ssl_features)
            else:
                continue
            
            for sample_idx in range(sae_features.shape[0]):
                if sample_count >= num_samples:
                    break
                    
                feature_matrix = sae_features[sample_idx]
                T = feature_matrix.shape[0]
                
                # Get binary activation pattern
                topk_indices_list = []
                for t in range(T):
                    _, indices = torch.topk(feature_matrix[t], k=model.sae.k, dim=0)
                    topk_indices_list.append(set(indices.cpu().tolist()))
                
                # Compute flipping rate
                flipping_counts = []
                for t in range(T - 1):
                    changes = len(topk_indices_list[t] ^ topk_indices_list[t + 1])
                    flipping_counts.append(changes)
                
                all_flipping_rates.extend(flipping_counts)
                
                # Compute transient spikes ratio
                feature_segments = {}
                for t, indices_set in enumerate(topk_indices_list):
                    for idx in indices_set:
                        if idx not in feature_segments:
                            feature_segments[idx] = []
                        
                        if feature_segments[idx] and feature_segments[idx][-1]['end'] == t - 1:
                            feature_segments[idx][-1]['end'] = t
                        else:
                            feature_segments[idx].append({'start': t, 'end': t})
                
                total_activations = 0
                transient_activations = 0
                for segments_list in feature_segments.values():
                    for seg in segments_list:
                        total_activations += 1
                        if seg['end'] - seg['start'] == 0:  # lifetime = 1
                            transient_activations += 1
                
                if total_activations > 0:
                    transient_ratio = transient_activations / total_activations
                    all_transient_spikes.append(transient_ratio)
                
                # Representation variance
                # Measure variance of the sparse representation across time
                active_features = feature_matrix > 0
                variance = torch.var(active_features.float(), dim=0).mean()
                all_repr_variances.append(variance.item())
                
                sample_count += 1
    
    return {
        'flipping_rates': all_flipping_rates,
        'flipping_rate_mean': float(np.mean(all_flipping_rates)),
        'flipping_rate_std': float(np.std(all_flipping_rates)),
        'transient_spike_ratio_mean': float(np.mean(all_transient_spikes)),
        'representation_variance_mean': float(np.mean(all_repr_variances)),
        'num_samples_analyzed': sample_count,
        'detailed_traces': []  # Could be expanded if needed
    }


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
            labels=['Transient (1 frame)', 'Persistent (>1 frame)'],
            autopct='%1.1f%%', startangle=90, colors=['#ff9999', '#66b3ff'])
    plt.title('Transient vs. Persistent Feature Activations')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'transient_vs_persistent.png'), dpi=300)
    plt.close()


def print_summary_report(results):
    """Print a comprehensive summary report."""
    
    print("\n" + "="*80)
    print("TEMPORAL STABILITY ANALYSIS REPORT - CPC MODEL")
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
    
    if findings:
        for i, finding in enumerate(findings, 1):
            print(f"{i}. {finding}")
    else:
        print("✓ Model shows good temporal stability across all metrics")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description='Analyze temporal stability of CPC-based window TopK SAE')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained CPC model')
    parser.add_argument('--database_path', type=str, default='/data/projects/punim2637/nnliang/Datasets/LA', help='Path to ASVspoof dataset')
    parser.add_argument('--protocols_path', type=str, default='/data/projects/punim2637/nnliang/Datasets/LA/ASVspoof2019_LA_cm_protocols', 
                        help='Path to protocol files')
    parser.add_argument('--output_dir', type=str, default='temporal_stability_analysis/cpc', 
                        help='Output directory for results')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples to analyze')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading CPC model...")
    device = torch.device(args.device)
    
    # Load model
    class DummyArgs:
        pass
    
    model_args = DummyArgs()
    model = Model(
        args=model_args,
        device=device,
        use_sae=True,
        sae_dict_size=4096,
        sae_k=128,
        sae_window_size=8,
        use_cpc=True,
        cpc_weight=0.5
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Handle DataParallel wrapper
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    
    print("Loading dataset...")
    
    class DataArgs:
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
    results['temporal_stability'] = analyze_temporal_stability_manual(
        model, 
        train_loader, 
        num_samples=args.num_samples
    )
    print("✓ Temporal stability analysis complete", flush=True)
    
    print("2/3 Analyzing feature identity stability...", flush=True)
    results['feature_identity'] = analyze_feature_identity_stability(
        model,
        train_loader,
        num_samples=min(args.num_samples, 50)
    )
    print("✓ Feature identity analysis complete", flush=True)
    
    print("3/3 Analyzing failure modes...", flush=True)
    results['failure_modes'] = analyze_temporal_failure_modes(
        model,
        train_loader,
        num_samples=min(args.num_samples, 20)
    )
    print("✓ Failure mode analysis complete", flush=True)
    
    # Save results
    results_file = os.path.join(args.output_dir, 'temporal_stability_results.json')
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_temporal_diagnostics(results, args.output_dir)
    print(f"Visualizations saved to {args.output_dir}/")
    
    # Print summary report
    print_summary_report(results)
    
    # Save summary report to file
    summary_file = os.path.join(args.output_dir, 'summary_report.txt')
    original_stdout = sys.stdout
    with open(summary_file, 'w') as f:
        sys.stdout = f
        print_summary_report(results)
        sys.stdout = original_stdout
    
    print(f"\nSummary report saved to {summary_file}")
    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()
