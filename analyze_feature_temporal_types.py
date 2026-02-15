"""
Analyze whether different features have different temporal characteristics.
Test hypothesis: discriminative features are transient (short-term), not persistent.

This explains why overlapping windows fail: they smooth ALL features,
destroying discriminative transient features while only benefiting persistent features.
"""

import torch
import torch.nn as nn
import numpy as np
import os
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from model_window_topk import Model as WindowTopKModel
import json
from tqdm import tqdm
from collections import defaultdict


def pad(x, max_len=64600):
    """Pad or truncate audio to fixed length"""
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


class Dataset_ASVspoof2021_eval_torchaudio(Dataset):
    def __init__(self, list_IDs, base_dir):
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut = 64600
    
    def __len__(self):
        return len(self.list_IDs)
    
    def __getitem__(self, index):
        utt_id = self.list_IDs[index]
        flac_path = os.path.join(self.base_dir, 'flac', f'{utt_id}.flac')
        
        try:
            waveform, sample_rate = torchaudio.load(flac_path)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
            X = waveform.squeeze().numpy()
            X_pad = pad(X, self.cut)
            x_inp = Tensor(X_pad)
            return x_inp, utt_id
        except Exception as e:
            print(f"Warning: Failed to load {utt_id}: {e}")
            x_inp = Tensor(np.zeros(self.cut))
            return x_inp, utt_id

def compute_feature_lifetimes(active_mask):
    """
    Compute lifetime (consecutive activation duration) for each feature activation.
    
    Args:
        active_mask: Boolean tensor [T, D] indicating which features are active
    
    Returns:
        List of lifetimes for each activation instance
    """
    T, D = active_mask.shape
    lifetimes = []
    
    for d in range(D):
        feature_active = active_mask[:, d]
        
        # Find consecutive activation runs
        current_run = 0
        for t in range(T):
            if feature_active[t]:
                current_run += 1
            else:
                if current_run > 0:
                    lifetimes.append(current_run)
                    current_run = 0
        
        # Handle last run
        if current_run > 0:
            lifetimes.append(current_run)
    
    return lifetimes


def analyze_feature_temporal_characteristics(
    model,
    dataloader,
    device,
    top_k_features,
    num_samples=2000,
    window_size=8
):
    """
    Analyze temporal characteristics of top-k discriminative features vs all features.
    
    Key hypothesis:
    - Discriminative features (top-k) should be more TRANSIENT (short-lived)
    - Non-discriminative features should be more PERSISTENT (long-lived)
    """
    model.eval()
    
    # Statistics collectors
    all_feature_lifetimes = defaultdict(list)  # feature_id -> list of lifetimes
    all_feature_activations = defaultdict(int)  # feature_id -> total activations
    
    count = 0
    with torch.no_grad():
        for batch_x, _ in tqdm(dataloader, desc="Analyzing temporal characteristics"):
            if count >= num_samples:
                break
            
            batch_x = batch_x.to(device)
            
            # Get sparse features
            output = model(batch_x, return_sae_loss=False, return_interpretability=True)
            sparse_features = model.module.last_sparse_features  # [B, T, D]
            
            # For each sample in batch
            for b in range(sparse_features.shape[0]):
                if count >= num_samples:
                    break
                
                features = sparse_features[b]  # [T, D]
                active_mask = (features > 0)  # [T, D]
                T, D = active_mask.shape
                
                # For each feature dimension
                for d in range(D):
                    feature_active = active_mask[:, d]
                    
                    # Find consecutive activation runs
                    current_run = 0
                    for t in range(T):
                        if feature_active[t]:
                            current_run += 1
                        else:
                            if current_run > 0:
                                all_feature_lifetimes[d].append(current_run)
                                all_feature_activations[d] += 1
                                current_run = 0
                    
                    # Handle last run
                    if current_run > 0:
                        all_feature_lifetimes[d].append(current_run)
                        all_feature_activations[d] += 1
                
                count += 1
    
    # Compute statistics for each feature
    feature_stats = {}
    for feature_id in range(4096):  # Assuming dict_size=4096
        if feature_id in all_feature_lifetimes and len(all_feature_lifetimes[feature_id]) > 0:
            lifetimes = all_feature_lifetimes[feature_id]
            feature_stats[feature_id] = {
                'mean_lifetime': np.mean(lifetimes),
                'median_lifetime': np.median(lifetimes),
                'std_lifetime': np.std(lifetimes),
                'total_activations': all_feature_activations[feature_id],
                'num_lifetime_instances': len(lifetimes),
                'transient_ratio': sum(1 for lt in lifetimes if lt < window_size) / len(lifetimes),
                'persistent_ratio': sum(1 for lt in lifetimes if lt >= window_size) / len(lifetimes),
            }
    
    # Separate top-k discriminative features vs others
    top_k_set = set(top_k_features)
    
    discriminative_stats = []
    non_discriminative_stats = []
    
    for feature_id, stats in feature_stats.items():
        if feature_id in top_k_set:
            discriminative_stats.append(stats)
        else:
            non_discriminative_stats.append(stats)
    
    # Aggregate statistics
    def aggregate_stats(stats_list):
        if not stats_list:
            return {}
        
        return {
            'mean_lifetime_avg': np.mean([s['mean_lifetime'] for s in stats_list]),
            'mean_lifetime_std': np.std([s['mean_lifetime'] for s in stats_list]),
            'median_lifetime_avg': np.mean([s['median_lifetime'] for s in stats_list]),
            'transient_ratio_avg': np.mean([s['transient_ratio'] for s in stats_list]),
            'persistent_ratio_avg': np.mean([s['persistent_ratio'] for s in stats_list]),
            'total_activations_avg': np.mean([s['total_activations'] for s in stats_list]),
            'num_features': len(stats_list),
        }
    
    discriminative_agg = aggregate_stats(discriminative_stats)
    non_discriminative_agg = aggregate_stats(non_discriminative_stats)
    
    # Statistical test: t-test on mean lifetimes
    from scipy import stats as scipy_stats
    discriminative_lifetimes = [s['mean_lifetime'] for s in discriminative_stats]
    non_discriminative_lifetimes = [s['mean_lifetime'] for s in non_discriminative_stats]
    
    if len(discriminative_lifetimes) > 0 and len(non_discriminative_lifetimes) > 0:
        t_stat, p_value = scipy_stats.ttest_ind(discriminative_lifetimes, non_discriminative_lifetimes)
        
        # Effect size (Cohen's d)
        mean_diff = np.mean(discriminative_lifetimes) - np.mean(non_discriminative_lifetimes)
        pooled_std = np.sqrt((np.std(discriminative_lifetimes)**2 + np.std(non_discriminative_lifetimes)**2) / 2)
        cohens_d = mean_diff / (pooled_std + 1e-8)
    else:
        t_stat, p_value, cohens_d = None, None, None
    
    results = {
        'hypothesis': 'Discriminative features should be more transient (short-lived) than non-discriminative features',
        'num_samples_analyzed': count,
        'window_size': window_size,
        'top_k_features_count': len(top_k_features),
        
        'discriminative_features': {
            'description': 'Top-k features that drive classification decisions',
            'statistics': discriminative_agg,
        },
        
        'non_discriminative_features': {
            'description': 'Other features with lower attribution scores',
            'statistics': non_discriminative_agg,
        },
        
        'comparison': {
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'interpretation': {
                'lifetime_difference': discriminative_agg.get('mean_lifetime_avg', 0) - non_discriminative_agg.get('mean_lifetime_avg', 0),
                'transient_ratio_difference': discriminative_agg.get('transient_ratio_avg', 0) - non_discriminative_agg.get('transient_ratio_avg', 0),
                'conclusion': None  # Will be filled based on results
            }
        },
        
        'implication_for_overlapping_windows': {
            'if_discriminative_more_transient': 'Overlapping windows smooth transient features, destroying discriminative power',
            'if_discriminative_more_persistent': 'Overlapping windows should help, but experiments show degradation - other factors involved',
        },
        
        'detailed_feature_stats': {
            f'feature_{fid}': stats 
            for fid, stats in list(feature_stats.items())[:100]  # Top 100 for brevity
        }
    }
    
    # Add interpretation
    if p_value is not None and p_value < 0.05:
        if discriminative_agg['mean_lifetime_avg'] < non_discriminative_agg['mean_lifetime_avg']:
            results['comparison']['interpretation']['conclusion'] = (
                "VALIDATED: Discriminative features are significantly MORE TRANSIENT than non-discriminative features. "
                f"Mean lifetime: {discriminative_agg['mean_lifetime_avg']:.2f} vs {non_discriminative_agg['mean_lifetime_avg']:.2f}. "
                f"This explains why overlapping windows (smoothing) degrades EER from 2.94% to 7.22%."
            )
        else:
            results['comparison']['interpretation']['conclusion'] = (
                "REJECTED: Discriminative features are significantly MORE PERSISTENT than non-discriminative features. "
                f"Mean lifetime: {discriminative_agg['mean_lifetime_avg']:.2f} vs {non_discriminative_agg['mean_lifetime_avg']:.2f}. "
                "This suggests other factors explain overlapping windows failure."
            )
    else:
        results['comparison']['interpretation']['conclusion'] = (
            "INCONCLUSIVE: No significant difference in temporal characteristics between discriminative and non-discriminative features. "
            f"p-value: {p_value if p_value is not None else 'N/A'}"
        )
    
    return results


if __name__ == '__main__':
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model_path = 'models/topk_sae_window_w8_LA_e40_bs14_lr1e-06_saeW0.1_dict4096_k128_window_topk_w8/best_checkpoint_eer_window_topk_w8.pth'
    print(f"Loading model from {model_path}")
    
    class DummyArgs:
        pass
    
    model = WindowTopKModel(
        args=DummyArgs(),
        device=device,
        use_sae=True,
        sae_dict_size=4096,
        sae_k=128,
        sae_window_size=8,
        sae_weight=0.1
    )
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    model = nn.DataParallel(model).to(device)
    model.eval()
    
    # Load top-k discriminative features from previous analysis
    print("Loading top-k discriminative features...")
    with open('decision_analysis_2021LA_5k/decision_analysis_results.json', 'r') as f:
        decision_results = json.load(f)
    
    top_k_features = decision_results['attribution']['top_k_indices'][:50]
    print(f"Top-50 discriminative features: {top_k_features[:10]}... (showing first 10)")
    
    # Load dataset
    print("Loading ASVspoof 2021 LA evaluation dataset...")
    database_path = '/data/projects/punim2637/nnliang/Datasets/ASVspoof2021_LA_eval/'
    
    # Get file list from flac directory
    flac_dir = os.path.join(database_path, 'flac')
    all_files = sorted([f.replace('.flac', '') for f in os.listdir(flac_dir) if f.endswith('.flac')])
    print(f'Found {len(all_files)} files')
    
    eval_dataset = Dataset_ASVspoof2021_eval_torchaudio(
        list_IDs=all_files,
        base_dir=database_path
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0
    )
    
    # Run analysis
    print("\n" + "="*80)
    print("HYPOTHESIS: Discriminative features are MORE TRANSIENT (short-lived)")
    print("If TRUE: Overlapping windows smooth them -> EER degradation")
    print("If FALSE: Need to find other explanation for overlapping failure")
    print("="*80 + "\n")
    
    results = analyze_feature_temporal_characteristics(
        model=model,
        dataloader=eval_loader,
        device=device,
        top_k_features=top_k_features,
        num_samples=2000,
        window_size=8
    )
    
    # Save results
    output_path = 'feature_temporal_types_analysis.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to {output_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("ANALYSIS SUMMARY")
    print("="*80)
    
    disc_stats = results['discriminative_features']['statistics']
    non_disc_stats = results['non_discriminative_features']['statistics']
    
    print(f"\nDiscriminative Features (Top-50):")
    print(f"  Mean Lifetime: {disc_stats['mean_lifetime_avg']:.2f} timesteps")
    print(f"  Transient Ratio: {disc_stats['transient_ratio_avg']*100:.1f}% (lifetime < {results['window_size']})")
    print(f"  Persistent Ratio: {disc_stats['persistent_ratio_avg']*100:.1f}% (lifetime >= {results['window_size']})")
    
    print(f"\nNon-Discriminative Features:")
    print(f"  Mean Lifetime: {non_disc_stats['mean_lifetime_avg']:.2f} timesteps")
    print(f"  Transient Ratio: {non_disc_stats['transient_ratio_avg']*100:.1f}% (lifetime < {results['window_size']})")
    print(f"  Persistent Ratio: {non_disc_stats['persistent_ratio_avg']*100:.1f}% (lifetime >= {results['window_size']})")
    
    print(f"\nStatistical Test:")
    print(f"  t-statistic: {results['comparison']['t_statistic']:.4f}")
    print(f"  p-value: {results['comparison']['p_value']:.6f}")
    print(f"  Cohen's d: {results['comparison']['cohens_d']:.4f}")
    
    print(f"\n🎯 CONCLUSION:")
    print(f"  {results['comparison']['interpretation']['conclusion']}")
    
    print("\n" + "="*80)
    print("IMPLICATION:")
    if disc_stats['mean_lifetime_avg'] < non_disc_stats['mean_lifetime_avg']:
        print("  ❌ Overlapping windows SHOULD NOT be used - they smooth discriminative transient features")
        print("  ✅ Feature-specific smoothing: Only smooth non-discriminative persistent features")
        print("  ✅ Preserve discriminative transient features unchanged")
    else:
        print("  ⚠️ Temporal characteristics alone don't explain overlapping failure")
        print("  🔍 Other factors: train-test mismatch, feature interaction, semantic drift")
    print("="*80 + "\n")
