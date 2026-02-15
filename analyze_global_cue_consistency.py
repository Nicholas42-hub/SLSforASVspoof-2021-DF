#!/usr/bin/env python3
"""
Global Cue Consistency Analysis
目的：验证是否每个 sample 在整个 utterance 中都使用相同的 decision cues

与 analyze_decision_relevance.py 的区别：
- 那个只测量相邻帧的 overlap (local consistency)
- 这个测量全局的 cue overlap (global consistency)
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path

# Import existing modules
from model_window_topk import Model as WindowTopKModel


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

def compute_jaccard_similarity(set1, set2):
    """计算两个集合的 Jaccard similarity"""
    if len(set1) == 0 and len(set2) == 0:
        return 1.0
    if len(set1) == 0 or len(set2) == 0:
        return 0.0
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0


def compute_global_metrics(cue_sets, utt_id):
    """
    Compute global cue consistency metrics for a single sample
    
    Returns:
        dict with global metrics
    """
    T = len(cue_sets)
    
    # 1. Global Jaccard: 所有帧对的平均相似度
    global_jaccards = []
    for i in range(T):
        for j in range(i+1, T):
            jac = compute_jaccard_similarity(cue_sets[i], cue_sets[j])
            global_jaccards.append(jac)
    
    global_jaccard_mean = np.mean(global_jaccards) if global_jaccards else 0.0
    global_jaccard_std = np.std(global_jaccards) if global_jaccards else 0.0
    
    # 2. Temporal Drift: 首帧 vs 末帧
    temporal_drift = compute_jaccard_similarity(cue_sets[0], cue_sets[-1])
    
    # 3. Core Cue Stability: 交集 / 并集
    intersection = set.intersection(*cue_sets) if cue_sets else set()
    union = set.union(*cue_sets) if cue_sets else set()
    core_stability = len(intersection) / len(union) if union else 0.0
    
    # 4. First-frame Consistency: 每帧与首帧的平均
    first_frame_jaccards = [
        compute_jaccard_similarity(cue_sets[0], cue_sets[t])
        for t in range(1, T)
    ]
    first_frame_mean = np.mean(first_frame_jaccards) if first_frame_jaccards else 0.0
    first_frame_std = np.std(first_frame_jaccards) if first_frame_jaccards else 0.0
    
    # 5. Adjacent-frame (for comparison)
    adjacent_jaccards = [
        compute_jaccard_similarity(cue_sets[t], cue_sets[t+1])
        for t in range(T-1)
    ]
    adjacent_mean = np.mean(adjacent_jaccards) if adjacent_jaccards else 0.0
    
    # 6. Core cue analysis
    core_cues = list(intersection)
    variable_cues = list(union - intersection)
    
    results = {
        'utt_id': utt_id,
        'num_frames': T,
        'avg_active_cues': float(np.mean([len(s) for s in cue_sets])),
        
        # Global consistency metrics
        'global_jaccard_mean': float(global_jaccard_mean),
        'global_jaccard_std': float(global_jaccard_std),
        
        # Temporal drift
        'temporal_drift': float(temporal_drift),
        
        # Core cue stability
        'core_stability': float(core_stability),
        'num_core_cues': len(core_cues),
        'num_variable_cues': len(variable_cues),
        'num_union_cues': len(union),
        
        # First-frame consistency
        'first_frame_consistency_mean': float(first_frame_mean),
        'first_frame_consistency_std': float(first_frame_std),
        
        # Adjacent (for comparison)
        'adjacent_jaccard_mean': float(adjacent_mean),
        
        # Core cue indices
        'core_cues': core_cues,
    }
    
    return results


class GlobalCueConsistencyAnalyzer:
    """分析全局 cue consistency - SIMPLIFIED VERSION"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, 
                       default='models/topk_sae_window_w8_LA_e40_bs14_lr1e-06_saeW0.1_dict4096_k128_window_topk_w8/best_checkpoint_eer_window_topk_w8.pth',
                       help='Path to trained model')
    parser.add_argument('--num_samples', type=int, default=500,
                       help='Number of samples to analyze')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size (must be 1 for this analysis)')
    parser.add_argument('--output_dir', type=str, 
                       default='global_cue_consistency_analysis',
                       help='Output directory')
    parser.add_argument('--dataset', type=str, default='2021_LA',
                       choices=['2021_LA', '2019_LA'],
                       help='Dataset to analyze')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed info for each sample')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model (same as analyze_feature_temporal_2000.py)
    print(f"Loading model from {args.model_path}")
    
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
    
    checkpoint = torch.load(args.model_path, map_location=device)
    model = nn.DataParallel(model).to(device)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    print("Model loaded successfully")
    
    # Load dataset - directly from flac directory like analyze_feature_temporal_2000.py
    print("Loading dataset...")
    database_path = '/data/projects/punim2637/nnliang/Datasets/ASVspoof2021_LA_eval/'
    
    # Get files directly from flac directory
    flac_dir = os.path.join(database_path, 'flac')
    all_files = sorted([f.replace('.flac', '') for f in os.listdir(flac_dir) if f.endswith('.flac')])
    print(f"Total files in dataset: {len(all_files)}")
    
    # Limit number of samples
    if args.num_samples > 0:
        file_list = all_files[:args.num_samples]
    else:
        file_list = all_files
    print(f"Analyzing {len(file_list)} samples")
    
    # Phase 1: Feature Attribution (using existing results to save time)
    print("\n=== Phase 1: Loading pre-computed top-50 decision features ===")
    with open('decision_analysis_2021LA_5k/decision_analysis_results.json', 'r') as f:
        decision_results = json.load(f)
    top_k_indices = np.array(decision_results['attribution']['top_k_indices'][:50])
    print(f"Top-50 decision features loaded: {len(top_k_indices)} features")
    
    # Phase 2: Global Cue Consistency Analysis using batched data loading
    print("\n=== Phase 2: Global Cue Consistency Analysis ===")
    
    # Create dataset and dataloader
    eval_dataset = Dataset_ASVspoof2021_eval_torchaudio(list_IDs=file_list, base_dir=database_path)
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    analyzer = GlobalCueConsistencyAnalyzer(model, device)
    
    all_results = []
    count = 0
    
    with torch.no_grad():
        for batch_x, batch_ids in tqdm(eval_loader, desc="Analyzing global consistency"):
            if count >= args.num_samples:
                break
            
            utt_id = batch_ids[0]
            batch_x = batch_x.to(device)
            
            # Get SAE activations through the model
            output = model(batch_x, return_sae_loss=False, return_interpretability=True)
            sparse_features = model.module.last_sparse_features  # [B, T, 4096]
            
            # Extract for this sample
            sae_activations = sparse_features[0]  # [T, 4096]
            T = sae_activations.shape[0]
            
            # Extract decision cues (top-k features)
            decision_activations = sae_activations[:, top_k_indices]  # [T, 50]
            active_cues = (decision_activations > 0).cpu().numpy()  # [T, 50]
            
            # Convert to sets for each frame
            cue_sets = []
            for t in range(T):
                active_indices = set(np.where(active_cues[t])[0].tolist())
                cue_sets.append(active_indices)
            
            # Compute global metrics
            result = compute_global_metrics(cue_sets, utt_id)
            all_results.append(result)
            
            count += 1
    
    print(f"\n✓ Analyzed {count} samples")
    
    # Aggregate results
    print("\n=== Aggregated Results ===")
    
    def aggregate_metrics(results):
        metrics = {}
        for key in ['global_jaccard_mean', 'temporal_drift', 'core_stability', 
                    'first_frame_consistency_mean', 'adjacent_jaccard_mean',
                    'num_core_cues', 'num_variable_cues', 'num_union_cues']:
            values = [r[key] for r in results if key in r]
            if values:
                metrics[f'{key}_mean'] = float(np.mean(values))
                metrics[f'{key}_std'] = float(np.std(values))
        return metrics
    
    aggregated = aggregate_metrics(all_results)
    
    # Print comparison
    print(f"\n{'Metric':<40} {'Value':<20}")
    print("=" * 60)
    
    key_metrics = [
        ('global_jaccard_mean', 'Global Jaccard (all pairs)'),
        ('adjacent_jaccard_mean', 'Adjacent Jaccard (consecutive)'),
        ('first_frame_consistency_mean', 'First-frame Consistency'),
        ('temporal_drift', 'Temporal Drift (first vs last)'),
        ('core_stability', 'Core Stability (∩/∪)'),
        ('num_core_cues', 'Num Core Cues (always active)'),
        ('num_variable_cues', 'Num Variable Cues'),
    ]
    
    for key, name in key_metrics:
        val = aggregated.get(f'{key}_mean', 0)
        std = aggregated.get(f'{key}_std', 0)
        print(f"{name:<40} {val:.4f} ± {std:.4f}")
    
    # Save results
    output_file = os.path.join(args.output_dir, 'global_cue_consistency_results.json')
    results_dict = {
        'config': vars(args),
        'top_k_indices': top_k_indices.tolist(),
        'samples': all_results,
        'aggregated': aggregated,
    }
    
    with open(output_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    # Key insights
    print("\n=== Key Insights ===")
    print(f"1. Adjacent Jaccard vs Global Jaccard:")
    print(f"   {aggregated['adjacent_jaccard_mean_mean']:.4f} (adjacent) vs {aggregated['global_jaccard_mean_mean']:.4f} (global)")
    print(f"   → Gap = {abs(aggregated['adjacent_jaccard_mean_mean'] - aggregated['global_jaccard_mean_mean']):.4f}")
    
    print(f"\n2. Temporal Drift (first vs last frame):")
    print(f"   {aggregated['temporal_drift_mean']:.4f}")
    print(f"   → {'High consistency' if aggregated['temporal_drift_mean'] > 0.9 else 'Some drift detected'}")
    
    print(f"\n3. Core Cues (always active across all frames):")
    print(f"   {aggregated['num_core_cues_mean']:.1f} / 50")
    print(f"   → These are truly persistent decision cues")
    
    print(f"\n4. First-frame Consistency:")
    print(f"   {aggregated['first_frame_consistency_mean_mean']:.4f}")
    print(f"   → Average similarity of each frame to the first frame")


if __name__ == '__main__':
    main()
