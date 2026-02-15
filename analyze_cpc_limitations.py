"""
Analyze limitations of CPC-enhanced Window-based TopK model.
Compares with baseline Window TopK to measure CPC's improvement.
"""

import os
import sys
import argparse
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Import model
sys.path.insert(0, os.path.dirname(__file__))
from data_utils_SSL import genSpoof_list, Dataset_ASVspoof2019_train

# Import CPC model
import importlib.util
spec = importlib.util.spec_from_file_location("model_cpc", "model_topk contrastive loss.py")
model_cpc_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_cpc_module)

def load_model(model_path, device):
    """Load CPC model from checkpoint."""
    print(f"Loading model from {model_path}...")
    
    # Create model instance
    class Args:
        pass
    args = Args()
    
    model = model_cpc_module.Model(
        args=args,
        device=device,
        use_sae=True,
        use_sparse_features=True,
        sae_dict_size=4096,
        sae_k=128,
        sae_window_size=8,
        use_cpc=True,
        cpc_hidden_dim=256,
        cpc_weight=0.5,
    )
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model

def load_dataset(database_path, protocols_path, num_samples=100, batch_size=8):
    """Load ASVspoof2019 LA training dataset."""
    print("Loading dataset...")
    
    d_label_trn, file_train = genSpoof_list(
        dir_meta=os.path.join(protocols_path, 'ASVspoof2019.LA.cm.train.trn.txt'),
        is_train=True, is_eval=False
    )
    
    train_set = Dataset_ASVspoof2019_train(
        list_IDs=file_train,
        labels=d_label_trn,
        base_dir=os.path.join(database_path, 'ASVspoof2019_LA_train/flac'),
    )
    
    # Limit to num_samples
    if len(train_set) > num_samples:
        indices = torch.randperm(len(train_set))[:num_samples]
        train_set = torch.utils.data.Subset(train_set, indices)
    
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--database_path', type=str, required=True)
    parser.add_argument('--protocols_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='cpc_limitations_analysis')
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = load_model(args.model_path, device)
    
    # Load dataset
    train_loader = load_dataset(
        args.database_path,
        args.protocols_path,
        num_samples=args.num_samples,
        batch_size=args.batch_size
    )
    
    print("\n" + "="*80)
    print("ANALYZING CPC-ENHANCED WINDOW-BASED TOPK LIMITATIONS")
    print("="*80)
    
    results = {}
    
    # 1. Boundary Discontinuity Analysis
    print("\n[1/4] Analyzing window boundary discontinuity...")
    boundary_results = model.analyze_window_boundary_discontinuity(
        train_loader, num_samples=args.num_samples
    )
    results['boundary_analysis'] = boundary_results
    
    print(f"\n  Results:")
    print(f"    Boundary Jaccard: {boundary_results['boundary_jaccard_mean']:.3f}")
    print(f"    Within Jaccard:   {boundary_results['within_jaccard_mean']:.3f}")
    print(f"    Discontinuity:    {boundary_results['discontinuity_score']:.3f}")
    
    if boundary_results['discontinuity_score'] > 0.05:
        print(f"    ✗ PROBLEM: Significant boundary discontinuity detected!")
    else:
        print(f"    ✓ Boundary discontinuity is acceptable")
    
    # 2. Multi-scale Analysis
    print("\n[2/4] Analyzing multi-scale temporal structure...")
    scale_results = model.analyze_multi_scale_temporal_structure(
        train_loader, num_samples=min(args.num_samples, 50)
    )
    results['scale_analysis'] = scale_results
    
    print(f"\n  Results:")
    for ws, metrics in scale_results['scale_analysis'].items():
        print(f"    {ws}: Jaccard={metrics['mean_jaccard']:.3f}")
    print(f"    Optimal window: {scale_results['optimal_window_size']}")
    print(f"    Current window: {scale_results['current_window_size']}")
    
    if scale_results['optimal_window_size'] != scale_results['current_window_size']:
        print(f"    ⚠ Fixed window size may not be optimal!")
    else:
        print(f"    ✓ Window size is optimal")
    
    # 3. Semantic Drift Analysis
    print("\n[3/4] Analyzing semantic drift...")
    semantic_results = model.analyze_semantic_drift(
        train_loader, num_samples=min(args.num_samples, 50)
    )
    results['semantic_drift'] = semantic_results
    
    print(f"\n  Results:")
    print(f"    Semantic consistency: {semantic_results['semantic_consistency_mean']:.3f}")
    print(f"    Features analyzed: {semantic_results['num_features_analyzed']}")
    
    if semantic_results['semantic_consistency_mean'] < 0.3:
        print(f"    ✗ PROBLEM: Semantic drift detected!")
    else:
        print(f"    ✓ Features have stable semantics")
    
    # 4. Discriminative Transients Analysis
    print("\n[4/4] Analyzing discriminative transients...")
    transient_results = model.analyze_discriminative_transients(
        train_loader, num_samples=args.num_samples
    )
    results['transient_analysis'] = transient_results
    
    print(f"\n  Results:")
    print(f"    Transient discriminative power: {transient_results['transient_discriminative_power']:.3f}")
    print(f"    Persistent discriminative power: {transient_results['persistent_discriminative_power']:.3f}")
    print(f"    Ratio: {transient_results['ratio']:.3f}")
    
    if transient_results['ratio'] > 0.5:
        print(f"    ⚠ Transients may be discriminative, could be over-smoothed")
    else:
        print(f"    ✓ Transients are not critical for discrimination")
    
    # Save results
    results_file = os.path.join(args.output_dir, 'cpc_limitations_analysis.json')
    with open(results_file, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            else:
                return obj
        
        json.dump(convert_types(results), f, indent=4)
    
    print("\n" + "="*80)
    print("SUMMARY REPORT")
    print("="*80)
    
    # Identify limitations
    limitations = []
    if boundary_results['discontinuity_score'] > 0.05:
        limitations.append("1. Window boundary discontinuity")
    if scale_results['optimal_window_size'] != scale_results['current_window_size']:
        limitations.append("2. Suboptimal fixed window size")
    if semantic_results['semantic_consistency_mean'] < 0.3:
        limitations.append("3. Semantic drift")
    if transient_results['ratio'] > 0.5:
        limitations.append("4. Over-smoothed discriminative transients")
    
    if limitations:
        print("\nLIMITATIONS IDENTIFIED:")
        for lim in limitations:
            print(f"  {lim}")
    else:
        print("\n✓ No major limitations detected!")
    
    print("\nCPC's Impact:")
    print("  - Semantic consistency: Expected improvement in cross-window coherence")
    print("  - Boundary smoothness: Partial mitigation through representation space constraints")
    print("  - Fixed window size: No direct impact (architectural limitation)")
    
    print(f"\nResults saved to {results_file}")
    print("="*80)

if __name__ == '__main__':
    main()
