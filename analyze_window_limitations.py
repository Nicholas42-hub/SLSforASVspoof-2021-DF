"""
Analyze Limitations of Window-based TopK SAE

Systematically evaluates the remaining problems after window-based temporal regularization:
1. Window boundary discontinuity
2. Fixed temporal scale limitations
3. Semantic drift of features
4. Over-smoothing of discriminative transients

Usage:
    python analyze_window_limitations.py --model_path <path> --output_dir <dir>
"""

import argparse
import json
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader

# Add fairseq to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1'))

from data_utils_SSL import genSpoof_list, Dataset_ASVspoof2019_train
from model_window_topk import Model


def main():
    parser = argparse.ArgumentParser(description='Analyze window-based TopK limitations')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--database_path', type=str, default='/data/projects/punim2637/nnliang/Datasets/LA')
    parser.add_argument('--protocols_path', type=str, 
                        default='/data/projects/punim2637/nnliang/Datasets/LA/ASVspoof2019_LA_cm_protocols')
    parser.add_argument('--output_dir', type=str, default='window_limitations_analysis')
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading model...")
    device = torch.device(args.device)
    
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
        sae_window_size=8,
    ).to(device)
    
    checkpoint = torch.load(args.model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
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
    
    d_label_trn, file_train = genSpoof_list(
        dir_meta=os.path.join(args.protocols_path, 'ASVspoof2019.LA.cm.train.trn.txt'),
        is_train=True, is_eval=False
    )
    
    # Convert dict to list of labels matching file order
    labels_list = [d_label_trn[fname] for fname in file_train]
    
    # Create balanced subset with both bonafide and spoof samples
    # Note: genSpoof_list returns 1 for bonafide, 0 for spoof
    bonafide_indices = [i for i, label in enumerate(labels_list) if label == 1]
    spoof_indices = [i for i, label in enumerate(labels_list) if label == 0]
    
    print(f"  Total dataset: {len(labels_list)} samples (bonafide={len(bonafide_indices)}, spoof={len(spoof_indices)})")
    
    # Sample equal numbers of each class
    num_per_class = min(len(bonafide_indices), len(spoof_indices), args.num_samples // 2)
    
    if num_per_class == 0:
        raise ValueError("Not enough samples of both classes in the dataset!")
    
    selected_bonafide = np.random.choice(bonafide_indices, num_per_class, replace=False)
    selected_spoof = np.random.choice(spoof_indices, num_per_class, replace=False)
    selected_indices = np.concatenate([selected_bonafide, selected_spoof])
    np.random.shuffle(selected_indices)
    
    # Filter dataset
    file_train_balanced = [file_train[i] for i in selected_indices]
    d_label_trn_balanced = {fname: d_label_trn[fname] for fname in file_train_balanced}
    
    print(f"  Selected {len(file_train_balanced)} samples: bonafide={num_per_class}, spoof={num_per_class}")
    
    train_set = Dataset_ASVspoof2019_train(
        args=data_args, list_IDs=file_train_balanced, labels=d_label_trn_balanced,
        base_dir=os.path.join(args.database_path, 'ASVspoof2019_LA_train/'),
        algo=0
    )
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    print("\n" + "="*80)
    print("ANALYZING WINDOW-BASED TOPK LIMITATIONS")
    print("="*80)
    
    results = {}
    
    # 1. Window Boundary Discontinuity
    print("\n[1/4] Analyzing window boundary discontinuity...", flush=True)
    results['boundary_analysis'] = model.analyze_window_boundary_discontinuity(
        train_loader, num_samples=args.num_samples
    )
    
    print(f"\n  Results:")
    print(f"    Boundary Jaccard: {results['boundary_analysis']['boundary_jaccard_mean']:.3f}")
    print(f"    Within Jaccard:   {results['boundary_analysis']['within_jaccard_mean']:.3f}")
    print(f"    Discontinuity:    {results['boundary_analysis']['discontinuity_score']:.3f}")
    
    if results['boundary_analysis']['discontinuity_score'] > 0.05:
        print(f"    ✗ PROBLEM: Significant boundary discontinuity detected!")
    else:
        print(f"    ✓ Boundary transitions are smooth")
    
    # 2. Multi-scale Analysis
    print("\n[2/4] Analyzing multi-scale temporal structure...", flush=True)
    results['scale_analysis'] = model.analyze_multi_scale_temporal_structure(
        train_loader, num_samples=min(args.num_samples, 50)
    )
    
    print(f"\n  Results:")
    for scale, metrics in results['scale_analysis']['scale_analysis'].items():
        print(f"    {scale}: Jaccard={metrics['mean_jaccard']:.3f}")
    print(f"    Optimal window: {results['scale_analysis']['optimal_window_size']}")
    print(f"    Current window: {results['scale_analysis']['current_window_size']}")
    
    if results['scale_analysis']['optimal_window_size'] != results['scale_analysis']['current_window_size']:
        print(f"    ⚠ Fixed window size may not be optimal!")
    else:
        print(f"    ✓ Current window size appears optimal")
    
    # 3. Semantic Drift
    print("\n[3/4] Analyzing semantic drift...", flush=True)
    results['semantic_drift'] = model.analyze_semantic_drift(
        train_loader, num_samples=min(args.num_samples, 50)
    )
    
    print(f"\n  Results:")
    print(f"    Semantic consistency: {results['semantic_drift']['semantic_consistency_mean']:.3f}")
    print(f"    Features analyzed: {results['semantic_drift']['num_features_analyzed']}")
    
    if results['semantic_drift']['semantic_consistency_mean'] < 0.3:
        print(f"    ✗ PROBLEM: Features show semantic drift!")
    elif results['semantic_drift']['semantic_consistency_mean'] < 0.5:
        print(f"    ⚠ Moderate semantic consistency")
    else:
        print(f"    ✓ Features have stable semantics")
    
    # 4. Discriminative Transients
    print("\n[4/4] Analyzing discriminative transients (using improved method)...", flush=True)
    results['transient_analysis'] = model.analyze_discriminative_transients(
        train_loader, num_samples=args.num_samples
    )
    
    print(f"\n  Results:")
    print(f"    Transient discriminative power (AUC): {results['transient_analysis']['transient_discriminative_power']:.3f}")
    print(f"    Persistent discriminative power (AUC): {results['transient_analysis']['persistent_discriminative_power']:.3f}")
    print(f"    Ratio (transient/persistent): {results['transient_analysis']['ratio']:.3f}")
    
    if 'transient_accuracy' in results['transient_analysis']:
        print(f"    Transient accuracy: {results['transient_analysis']['transient_accuracy']:.3f}")
        print(f"    Persistent accuracy: {results['transient_analysis']['persistent_accuracy']:.3f}")
    
    if 'error' in results['transient_analysis']:
        print(f"    ⚠ WARNING: {results['transient_analysis']['error']}")
    elif results['transient_analysis']['ratio'] > 0.8:
        print(f"    ✗ PROBLEM: Transients are highly discriminative, window may be over-smoothing!")
    elif results['transient_analysis']['ratio'] > 0.5:
        print(f"    ⚠ Transients have significant discriminative power")
    elif results['transient_analysis']['ratio'] > 0.3:
        print(f"    ℹ Transients have moderate discriminative power")
    else:
        print(f"    ✓ Persistent features dominate, window smoothing is appropriate")
    
    # Save results
    def convert_to_serializable(obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    results_file = os.path.join(args.output_dir, 'limitations_analysis.json')
    with open(results_file, 'w') as f:
        json.dump(convert_to_serializable(results), f, indent=2)
    
    print(f"\n{'='*80}")
    print("SUMMARY REPORT")
    print("="*80)
    
    problems_found = []
    
    if results['boundary_analysis']['discontinuity_score'] > 0.05:
        problems_found.append("Window boundary discontinuity")
    
    if results['scale_analysis']['optimal_window_size'] != results['scale_analysis']['current_window_size']:
        problems_found.append("Suboptimal fixed window size")
    
    if results['semantic_drift']['semantic_consistency_mean'] < 0.3:
        problems_found.append("Severe semantic drift")
    elif results['semantic_drift']['semantic_consistency_mean'] < 0.5:
        problems_found.append("Moderate semantic drift")
    
    if 'error' not in results['transient_analysis']:
        if results['transient_analysis']['ratio'] > 0.8:
            problems_found.append("Severe over-smoothing of discriminative transients")
        elif results['transient_analysis']['ratio'] > 0.5:
            problems_found.append("Potential over-smoothing of discriminative transients")
    
    if problems_found:
        print("\nLIMITATIONS IDENTIFIED:")
        for i, problem in enumerate(problems_found, 1):
            print(f"  {i}. {problem}")
        print("\nThese limitations motivate further research:")
        print("  - CPC loss for semantic consistency")
        print("  - Adaptive/learnable window sizes")
        print("  - Separate treatment of discriminative transients")
    else:
        print("\n✓ No major limitations detected!")
        print("Window-based TopK appears to work well for this task.")
    
    print(f"\nResults saved to {results_file}")
    print("="*80)


if __name__ == '__main__':
    main()
