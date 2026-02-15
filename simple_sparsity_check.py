#!/usr/bin/env python3
"""
Simple sparsity check by analyzing SAE activations on a small dataset
"""

import torch
import numpy as np
import sys
import os
from tqdm import tqdm

# Add fairseq to path
sys.path.insert(0, 'fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1')

from model import Model, AutoEncoderTopK


def analyze_topk_sae_weights(checkpoint_path, model_name, k_value):
    """Analyze SAE without running full forward pass"""
    print(f"\n{'='*80}")
    print(f"SPARSITY ANALYSIS: {model_name}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*80}\n")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract SAE parameters
    encoder_weight = None
    decoder_weight = None
    encoder_bias = None
    b_dec = None
    
    for key in checkpoint['model_state_dict'].keys():
        if 'sae.encoder.weight' in key:
            encoder_weight = checkpoint['model_state_dict'][key]
        elif 'sae.decoder.weight' in key:
            decoder_weight = checkpoint['model_state_dict'][key]
        elif 'sae.encoder.bias' in key:
            encoder_bias = checkpoint['model_state_dict'][key]
        elif 'sae.b_dec' in key:
            b_dec = checkpoint['model_state_dict'][key]
    
    if encoder_weight is None:
        print("❌ No SAE found in checkpoint")
        return
    
    print("📊 SAE Architecture:")
    print(f"   Encoder: {encoder_weight.shape[1]} → {encoder_weight.shape[0]}")
    print(f"   Decoder: {decoder_weight.shape[1]} → {decoder_weight.shape[0]}")
    print(f"   TopK K value: {k_value}")
    print(f"   Theoretical sparsity: {k_value}/{encoder_weight.shape[0]} = {k_value/encoder_weight.shape[0]*100:.2f}%")
    
    # Analyze encoder weights
    print(f"\n🔍 Encoder Weight Statistics:")
    encoder_norm = torch.norm(encoder_weight, dim=1)
    print(f"   Weight norm - Mean: {encoder_norm.mean():.4f}, Std: {encoder_norm.std():.4f}")
    print(f"   Weight norm - Min: {encoder_norm.min():.4f}, Max: {encoder_norm.max():.4f}")
    
    # Analyze decoder weights
    print(f"\n🔍 Decoder Weight Statistics:")
    decoder_norm = torch.norm(decoder_weight, dim=0)
    print(f"   Weight norm - Mean: {decoder_norm.mean():.4f}, Std: {decoder_norm.std():.4f}")
    print(f"   Weight norm - Min: {decoder_norm.min():.4f}, Max: {decoder_norm.max():.4f}")
    
    # Check for dead features (very low norm)
    threshold = decoder_norm.mean() * 0.01  # 1% of mean
    dead_features = (decoder_norm < threshold).sum().item()
    print(f"\n⚠️  Potentially Dead Features (norm < {threshold:.6f}):")
    print(f"   Count: {dead_features}/{decoder_norm.shape[0]} ({dead_features/decoder_norm.shape[0]*100:.2f}%)")
    
    # Simulate sparse activation on random input
    print(f"\n🧪 Simulating Sparse Activation on Random Input:")
    test_input = torch.randn(100, 1024)  # 100 samples, 1024 dim
    
    # Compute pre-activations
    pre_activation = test_input @ encoder_weight.t() + encoder_bias
    
    # Apply TopK
    topk_values, topk_indices = torch.topk(pre_activation, k_value, dim=-1)
    
    # Create sparse activations
    sparse = torch.zeros_like(pre_activation)
    sparse.scatter_(-1, topk_indices, topk_values)
    
    # Statistics
    active_per_sample = (sparse > 0).sum(dim=-1).float()
    print(f"   Active features per sample: {active_per_sample.mean():.2f} ± {active_per_sample.std():.2f}")
    print(f"   Min/Max active: {active_per_sample.min():.0f} / {active_per_sample.max():.0f}")
    
    # Feature utilization across samples
    feature_usage = (sparse > 0).sum(dim=0)
    print(f"\n📈 Feature Utilization (across 100 random samples):")
    print(f"   Features used at least once: {(feature_usage > 0).sum()}/{sparse.shape[1]}")
    print(f"   Average usage per feature: {feature_usage.float().mean():.2f} samples")
    print(f"   Most used feature: {feature_usage.max()} times")
    print(f"   Never used features: {(feature_usage == 0).sum()}")
    
    print("\n" + "="*80 + "\n")


def main():
    print("\n🔬 TopK SAE Sparsity Analysis")
    print("="*80)
    
    models = [
        ('K=32 (Theoretical 0.78% sparsity)', 
         'models/topk_sae_LA_e40_bs14_lr1e-06_saeW0.1_dict4096_k32_k32_sparse_4096dim/best_checkpoint_eer_k32_sparse_4096dim.pth',
         32),
        ('K=128 (Theoretical 3.12% sparsity)',
         'models/topk_sae_LA_e40_bs14_lr1e-06_saeW0.1_dict4096_k128_k128_sparse_4096dim/best_checkpoint_eer_k128_sparse_4096dim.pth',
         128),
    ]
    
    for model_name, checkpoint_path, k_value in models:
        if not os.path.exists(checkpoint_path):
            print(f"\n❌ Checkpoint not found: {checkpoint_path}")
            continue
        
        analyze_topk_sae_weights(checkpoint_path, model_name, k_value)
    
    print("✅ Analysis complete!")


if __name__ == '__main__':
    main()
