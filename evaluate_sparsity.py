#!/usr/bin/env python3
"""
Evaluate the sparsity patterns of TopK SAE models
"""

import torch
import numpy as np
import sys
import os
from tqdm import tqdm
from torch.utils.data import DataLoader

# Add fairseq to path
sys.path.insert(0, 'fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1')

from model_window_topk import Model
from data_utils_SSL import genSpoof_list, Dataset_ASVspoof2019_train


def analyze_sparsity(model, data_loader, device, num_batches=50):
    """
    Analyze sparsity patterns of the SAE
    
    Args:
        model: The trained model
        data_loader: DataLoader for the dataset
        device: torch device
        num_batches: Number of batches to analyze
    
    Returns:
        dict: Sparsity statistics
    """
    model.eval()
    
    all_activations = []
    all_feature_counts = []
    total_samples = 0
    
    print(f"\nAnalyzing sparsity over {num_batches} batches...")
    
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(tqdm(data_loader)):
            if i >= num_batches:
                break
            
            batch_x = batch_x.to(device)
            batch_size = batch_x.size(0)
            total_samples += batch_size
            
            # Forward pass to get sparse features
            output, sae_loss = model(batch_x, return_sae_loss=True)
            
            # Get the sparse features from the model
            if hasattr(model.module, 'last_sparse_features') and model.module.last_sparse_features is not None:
                sparse_features = model.module.last_sparse_features  # [B, T, dict_size]
                
                # Count active features per sample
                active_mask = (sparse_features > 0).float()  # [B, T, dict_size]
                
                # Average over time dimension
                avg_active = active_mask.mean(dim=1)  # [B, dict_size]
                
                # Store activations
                all_activations.append(avg_active.cpu().numpy())
                
                # Count features per sample (average over time)
                num_active = active_mask.sum(dim=-1).mean(dim=1)  # [B]
                all_feature_counts.extend(num_active.cpu().numpy().tolist())
    
    # Concatenate all activations
    all_activations = np.concatenate(all_activations, axis=0)  # [N, dict_size]
    
    # Calculate statistics
    stats = {}
    
    # Per-sample statistics
    stats['avg_active_features'] = np.mean(all_feature_counts)
    stats['std_active_features'] = np.std(all_feature_counts)
    stats['min_active_features'] = np.min(all_feature_counts)
    stats['max_active_features'] = np.max(all_feature_counts)
    
    # Dictionary utilization
    dict_size = all_activations.shape[1]
    feature_usage = (all_activations > 0).sum(axis=0)  # How many samples activate each feature
    
    stats['dict_size'] = dict_size
    stats['features_ever_active'] = (feature_usage > 0).sum()
    stats['features_never_active'] = (feature_usage == 0).sum()
    stats['dict_utilization'] = stats['features_ever_active'] / dict_size * 100
    
    # Feature frequency distribution
    stats['feature_activation_mean'] = feature_usage.mean()
    stats['feature_activation_std'] = feature_usage.std()
    stats['feature_activation_max'] = feature_usage.max()
    
    # Sparsity percentage
    stats['sparsity_percent'] = stats['avg_active_features'] / dict_size * 100
    stats['theoretical_k'] = model.module.sae.k if hasattr(model.module, 'sae') else None
    
    # Top activated features
    top_features_idx = np.argsort(feature_usage)[::-1][:20]
    stats['top_20_features'] = [(int(idx), int(feature_usage[idx])) for idx in top_features_idx]
    
    # Dead features (never activated)
    dead_features = np.where(feature_usage == 0)[0]
    stats['dead_features_count'] = len(dead_features)
    stats['dead_features_percent'] = len(dead_features) / dict_size * 100
    
    return stats


def print_sparsity_report(stats, model_name):
    """Print a formatted sparsity report"""
    print("\n" + "="*80)
    print(f"SPARSITY ANALYSIS: {model_name}")
    print("="*80)
    
    print(f"\n📊 Dictionary Configuration:")
    print(f"   Dictionary Size: {stats['dict_size']}")
    print(f"   Theoretical K: {stats['theoretical_k']}")
    print(f"   Theoretical Sparsity: {stats['theoretical_k']/stats['dict_size']*100:.2f}%")
    
    print(f"\n🔍 Activation Statistics (per sample, averaged over time):")
    print(f"   Average Active Features: {stats['avg_active_features']:.2f} ± {stats['std_active_features']:.2f}")
    print(f"   Min/Max Active Features: {stats['min_active_features']:.0f} / {stats['max_active_features']:.0f}")
    print(f"   Actual Sparsity: {stats['sparsity_percent']:.2f}%")
    
    print(f"\n📈 Dictionary Utilization:")
    print(f"   Features Ever Active: {stats['features_ever_active']}/{stats['dict_size']} ({stats['dict_utilization']:.2f}%)")
    print(f"   Dead Features: {stats['dead_features_count']} ({stats['dead_features_percent']:.2f}%)")
    
    print(f"\n🎯 Feature Activation Frequency:")
    print(f"   Mean: {stats['feature_activation_mean']:.2f} samples")
    print(f"   Std: {stats['feature_activation_std']:.2f}")
    print(f"   Max: {stats['feature_activation_max']} samples")
    
    print(f"\n🏆 Top 20 Most Activated Features:")
    for i, (feature_idx, count) in enumerate(stats['top_20_features'][:10], 1):
        print(f"   {i:2d}. Feature {feature_idx:4d}: activated in {count} samples")
    
    print("\n" + "="*80 + "\n")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Configuration
    database_path = '/data/projects/punim2637/nnliang/Datasets/LA/'
    
    models_to_evaluate = [
        {
            'name': 'Window TopK K=128, W=8 (Theoretical 3.12% sparsity)',
            'checkpoint': 'models/topk_sae_window_w8_LA_e40_bs14_lr1e-06_saeW0.1_dict4096_k128_window_topk_w8/best_checkpoint_eer_window_topk_w8.pth',
            'sae_k': 128,
            'window_size': 8,
        }
    ]
    
    # Create dummy args for dataset
    class Args:
        def __init__(self):
            self.nBands = 5
            self.minF = 20
            self.maxF = 8000
            self.minBW = 100
            self.maxBW = 1000
            self.minCoeff = 10
            self.maxCoeff = 100
            self.minG = 0
            self.maxG = 0
            self.minBiasLinNonLin = 5
            self.maxBiasLinNonLin = 20
            self.N_f = 5
            self.P = 10
            self.g_sd = 2
            self.SNRmin = 10
            self.SNRmax = 40
    
    args = Args()
    
    # Load validation dataset
    print("Loading validation dataset...")
    d_label_dev, file_dev = genSpoof_list(
        dir_meta=os.path.join(database_path, 'ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'),
        is_train=False,
        is_eval=False
    )
    
    print("Creating validation data loader...")
    dev_set = Dataset_ASVspoof2019_train(
        args=args,
        list_IDs=file_dev,
        labels=d_label_dev,
        base_dir=os.path.join(database_path, 'ASVspoof2019_LA_dev/'),
        algo=0  # No augmentation
    )
    
    dev_loader = DataLoader(
        dev_set,
        batch_size=8,  # Smaller batch for memory efficiency
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=4
    )
    
    # Evaluate each model
    for model_config in models_to_evaluate:
        print(f"\n{'='*80}")
        print(f"Loading model: {model_config['name']}")
        print(f"Checkpoint: {model_config['checkpoint']}")
        print(f"{'='*80}")
        
        if not os.path.exists(model_config['checkpoint']):
            print(f"❌ Checkpoint not found: {model_config['checkpoint']}")
            continue
        
        # Initialize model
        class DummyArgs:
            pass
        
        dummy_args = DummyArgs()
        
        model = Model(
            args=dummy_args,
            device=device,
            cp_path='xlsr2_300m.pt',
            use_sae=True,
            sae_dict_size=4096,
            sae_k=model_config['sae_k'],
            sae_window_size=model_config.get('window_size', 1),
            use_sparse_features=True
        )
        
        # Load checkpoint
        checkpoint = torch.load(model_config['checkpoint'], map_location=device)
        
        # Move to device first
        model = model.to(device)
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        
        # Load state dict (it should match now with DataParallel)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Analyze sparsity
        stats = analyze_sparsity(model, dev_loader, device, num_batches=100)
        
        # Print report
        print_sparsity_report(stats, model_config['name'])
        
        # Clean up
        del model
        torch.cuda.empty_cache()
    
    print("✅ Sparsity evaluation completed!")


if __name__ == '__main__':
    main()
