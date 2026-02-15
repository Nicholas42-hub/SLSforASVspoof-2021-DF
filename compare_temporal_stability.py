"""
Compare temporal stability between Window TopK and CPC models.

Usage:
    python compare_temporal_stability.py --window_dir <dir> --cpc_dir <dir> --output <file>
"""

import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np


def load_results(results_file):
    """Load temporal stability results from JSON."""
    with open(results_file, 'r') as f:
        return json.load(f)


def create_comparison_table(window_results, cpc_results):
    """Create a markdown comparison table."""
    
    w_ts = window_results['temporal_stability']
    c_ts = cpc_results['temporal_stability']
    
    w_fi = window_results['feature_identity']
    c_fi = cpc_results['feature_identity']
    
    w_fm = window_results['failure_modes']
    c_fm = cpc_results['failure_modes']
    
    table = f"""
# Temporal Stability Comparison: Window TopK vs CPC

## Summary Metrics

| Metric | Window TopK | CPC Model | Change |
|--------|-------------|-----------|--------|
| **Feature Lifetime (mean)** | {w_ts['feature_lifetime_mean']:.2f} frames | {c_ts['feature_lifetime_mean']:.2f} frames | {((c_ts['feature_lifetime_mean'] - w_ts['feature_lifetime_mean']) / w_ts['feature_lifetime_mean'] * 100):+.1f}% |
| **Feature Lifetime (median)** | {w_ts['feature_lifetime_median']:.2f} frames | {c_ts['feature_lifetime_median']:.2f} frames | {((c_ts['feature_lifetime_median'] - w_ts['feature_lifetime_median']) / w_ts['feature_lifetime_median'] * 100):+.1f}% |
| **Transient Ratio** | {w_ts['transient_feature_ratio']:.2%} | {c_ts['transient_feature_ratio']:.2%} | {((c_ts['transient_feature_ratio'] - w_ts['transient_feature_ratio']) * 100):+.1f}pp |
| **Jaccard Similarity** | {w_ts['jaccard_similarity_mean']:.3f} | {c_ts['jaccard_similarity_mean']:.3f} | {((c_ts['jaccard_similarity_mean'] - w_ts['jaccard_similarity_mean']) / w_ts['jaccard_similarity_mean'] * 100):+.1f}% |
| **Index Overlap** | {w_fi['index_overlap_mean']:.3f} | {c_fi['index_overlap_mean']:.3f} | {((c_fi['index_overlap_mean'] - w_fi['index_overlap_mean']) / w_fi['index_overlap_mean'] * 100):+.1f}% |
| **Turnover Rate** | {w_fi['index_turnover_mean']:.2%} | {c_fi['index_turnover_mean']:.2%} | {((c_fi['index_turnover_mean'] - w_fi['index_turnover_mean']) * 100):+.1f}pp |
| **Flipping Rate** | {w_fm['flipping_rate_mean']:.2f} changes/frame | {c_fm['flipping_rate_mean']:.2f} changes/frame | {((c_fm['flipping_rate_mean'] - w_fm['flipping_rate_mean']) / w_fm['flipping_rate_mean'] * 100):+.1f}% |
| **Transient Spikes** | {w_fm['transient_spike_ratio_mean']:.2%} | {c_fm['transient_spike_ratio_mean']:.2%} | {((c_fm['transient_spike_ratio_mean'] - w_fm['transient_spike_ratio_mean']) * 100):+.1f}pp |

## Interpretation

### Feature Persistence
- **Window TopK**: Features persist for {w_ts['feature_lifetime_median']:.0f} frames (median), with {w_ts['transient_feature_ratio']:.1%} transient
- **CPC Model**: Features persist for {c_ts['feature_lifetime_median']:.0f} frames (median), with {c_ts['transient_feature_ratio']:.1%} transient
- **Winner**: {'CPC' if c_ts['feature_lifetime_median'] > w_ts['feature_lifetime_median'] else 'Window TopK'} (longer persistence = better)

### Temporal Coherence
- **Window TopK**: {w_ts['jaccard_similarity_mean']:.1%} overlap between consecutive frames
- **CPC Model**: {c_ts['jaccard_similarity_mean']:.1%} overlap between consecutive frames
- **Winner**: {'CPC' if c_ts['jaccard_similarity_mean'] > w_ts['jaccard_similarity_mean'] else 'Window TopK'} (higher coherence = better)

### Feature Stability
- **Window TopK**: {w_fm['flipping_rate_mean']:.1f} features change per frame
- **CPC Model**: {c_fm['flipping_rate_mean']:.1f} features change per frame
- **Winner**: {'CPC' if c_fm['flipping_rate_mean'] < w_fm['flipping_rate_mean'] else 'Window TopK'} (fewer changes = better)

## Key Findings

"""
    
    # Determine winner overall
    cpc_wins = 0
    window_wins = 0
    
    if c_ts['feature_lifetime_median'] > w_ts['feature_lifetime_median']:
        cpc_wins += 1
        table += "1. **CPC shows longer feature lifetimes**, indicating more persistent representations\n"
    else:
        window_wins += 1
        table += "1. **Window TopK shows longer feature lifetimes**, indicating more persistent representations\n"
    
    if c_ts['jaccard_similarity_mean'] > w_ts['jaccard_similarity_mean']:
        cpc_wins += 1
        table += "2. **CPC has better temporal coherence**, with higher frame-to-frame overlap\n"
    else:
        window_wins += 1
        table += "2. **Window TopK has better temporal coherence**, with higher frame-to-frame overlap\n"
    
    if c_fm['flipping_rate_mean'] < w_fm['flipping_rate_mean']:
        cpc_wins += 1
        table += "3. **CPC has fewer feature changes**, suggesting more stable feature selection\n"
    else:
        window_wins += 1
        table += "3. **Window TopK has fewer feature changes**, suggesting more stable feature selection\n"
    
    if c_ts['transient_feature_ratio'] < w_ts['transient_feature_ratio']:
        cpc_wins += 1
        table += "4. **CPC has fewer transient spikes**, with more meaningful activations\n"
    else:
        window_wins += 1
        table += "4. **Window TopK has fewer transient spikes**, with more meaningful activations\n"
    
    table += f"\n**Overall**: "
    if cpc_wins > window_wins:
        table += f"CPC model shows superior temporal stability ({cpc_wins} vs {window_wins} metrics)\n"
    elif window_wins > cpc_wins:
        table += f"Window TopK shows superior temporal stability ({window_wins} vs {cpc_wins} metrics)\n"
    else:
        table += "Both models show similar temporal stability\n"
    
    return table


def create_comparison_plots(window_results, cpc_results, output_dir):
    """Create side-by-side comparison plots."""
    
    # 1. Feature Lifetime Distribution Comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    w_lifetimes = window_results['temporal_stability']['feature_lifetime_distribution']
    c_lifetimes = cpc_results['temporal_stability']['feature_lifetime_distribution']
    
    axes[0].hist(w_lifetimes, bins=50, edgecolor='black', alpha=0.7, color='blue')
    axes[0].axvline(np.median(w_lifetimes), color='r', linestyle='--', 
                    label=f'Median: {np.median(w_lifetimes):.1f}')
    axes[0].set_xlabel('Feature Lifetime (frames)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Window TopK')
    axes[0].legend()
    
    axes[1].hist(c_lifetimes, bins=50, edgecolor='black', alpha=0.7, color='green')
    axes[1].axvline(np.median(c_lifetimes), color='r', linestyle='--', 
                    label=f'Median: {np.median(c_lifetimes):.1f}')
    axes[1].set_xlabel('Feature Lifetime (frames)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('CPC Model')
    axes[1].legend()
    
    plt.suptitle('Feature Lifetime Distribution Comparison')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_lifetime.png'), dpi=300)
    plt.close()
    
    # 2. Jaccard Similarity Comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    w_jaccard = window_results['temporal_stability']['jaccard_similarity_scores']
    c_jaccard = cpc_results['temporal_stability']['jaccard_similarity_scores']
    
    axes[0].hist(w_jaccard, bins=30, edgecolor='black', alpha=0.7, color='blue')
    axes[0].axvline(np.mean(w_jaccard), color='r', linestyle='--', 
                    label=f'Mean: {np.mean(w_jaccard):.3f}')
    axes[0].set_xlabel('Jaccard Similarity')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Window TopK')
    axes[0].legend()
    
    axes[1].hist(c_jaccard, bins=30, edgecolor='black', alpha=0.7, color='green')
    axes[1].axvline(np.mean(c_jaccard), color='r', linestyle='--', 
                    label=f'Mean: {np.mean(c_jaccard):.3f}')
    axes[1].set_xlabel('Jaccard Similarity')
    axes[1].set_ylabel('Count')
    axes[1].set_title('CPC Model')
    axes[1].legend()
    
    plt.suptitle('Temporal Coherence Comparison')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_jaccard.png'), dpi=300)
    plt.close()
    
    # 3. Bar Chart Comparison
    metrics = ['Median\nLifetime', 'Jaccard\nSimilarity', 'Flipping\nRate', 'Transient\nRatio']
    
    w_ts = window_results['temporal_stability']
    c_ts = cpc_results['temporal_stability']
    w_fm = window_results['failure_modes']
    c_fm = cpc_results['failure_modes']
    
    window_values = [
        w_ts['feature_lifetime_median'],
        w_ts['jaccard_similarity_mean'] * 100,  # Scale to 0-100
        w_fm['flipping_rate_mean'],
        w_ts['transient_feature_ratio'] * 100
    ]
    
    cpc_values = [
        c_ts['feature_lifetime_median'],
        c_ts['jaccard_similarity_mean'] * 100,
        c_fm['flipping_rate_mean'],
        c_ts['transient_feature_ratio'] * 100
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, window_values, width, label='Window TopK', color='blue', alpha=0.7)
    bars2 = ax.bar(x + width/2, cpc_values, width, label='CPC Model', color='green', alpha=0.7)
    
    ax.set_ylabel('Value')
    ax.set_title('Temporal Stability Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_bar_chart.png'), dpi=300)
    plt.close()
    
    print(f"Comparison plots saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description='Compare temporal stability between models')
    parser.add_argument('--window_dir', type=str, 
                       default='temporal_stability_analysis/top_k_window',
                       help='Directory with Window TopK results')
    parser.add_argument('--cpc_dir', type=str,
                       default='temporal_stability_analysis/cpc',
                       help='Directory with CPC results')
    parser.add_argument('--output', type=str,
                       default='temporal_comparison',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    print("Loading results...")
    
    # Load results
    window_file = os.path.join(args.window_dir, 'temporal_stability_results.json')
    cpc_file = os.path.join(args.cpc_dir, 'temporal_stability_results.json')
    
    if not os.path.exists(window_file):
        print(f"Error: Window TopK results not found at {window_file}")
        return
    
    if not os.path.exists(cpc_file):
        print(f"Error: CPC results not found at {cpc_file}")
        return
    
    window_results = load_results(window_file)
    cpc_results = load_results(cpc_file)
    
    print("Creating comparison table...")
    table = create_comparison_table(window_results, cpc_results)
    
    # Save table
    table_file = os.path.join(args.output, 'comparison_report.md')
    with open(table_file, 'w') as f:
        f.write(table)
    
    print(f"Comparison table saved to {table_file}")
    
    # Also save as txt
    txt_file = os.path.join(args.output, 'comparison_report.txt')
    with open(txt_file, 'w') as f:
        f.write(table)
    
    print(f"Text report saved to {txt_file}")
    
    print("\nCreating comparison plots...")
    create_comparison_plots(window_results, cpc_results, args.output)
    
    print("\n" + "="*80)
    print(table)
    print("="*80)
    
    print("\nComparison complete!")


if __name__ == '__main__':
    main()
