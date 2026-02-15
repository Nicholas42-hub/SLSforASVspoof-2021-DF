#!/usr/bin/env python
"""
Quick test: Check if current model can provide interpretability analysis
"""

import os
import sys
import torch
import numpy as np

# Add fairseq to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 
                                'fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1'))

from model import Model

def test_interpretability():
    print("="*70)
    print("TopK SAE Interpretability Test")
    print("="*70)
    
    model_path = 'models/topk_sae_LA_e40_bs14_lr1e-06_saeW0.1_dict4096_k128_topk_sae_test/best_checkpoint_eer_topk_sae_test.pth'
    
    # Load checkpoint
    print(f"\n1. Loading checkpoint...")
    checkpoint = torch.load(model_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Detect architecture
    classifier_weight_key = None
    for key in state_dict.keys():
        if 'classifier.1.weight' in key:
            classifier_weight_key = key
            break
    
    if classifier_weight_key:
        classifier_input_dim = state_dict[classifier_weight_key].shape[1]
        use_sparse = (classifier_input_dim == 4096)
    else:
        use_sparse = False
    
    print(f"   Classifier input dim: {classifier_input_dim}")
    print(f"   use_sparse_features: {use_sparse}")
    
    # Check SAE weights
    has_sae = False
    sae_k = None
    for key in state_dict.keys():
        if 'sae.k' in key:
            has_sae = True
            sae_k = state_dict[key].item() if hasattr(state_dict[key], 'item') else state_dict[key]
            break
    
    print(f"   Has SAE: {has_sae}")
    if has_sae and sae_k is not None:
        print(f"   SAE k (top-k): {sae_k}")
    
    print(f"\n2. Creating model...")
    device = torch.device('cpu')
    model = Model(
        args=None,
        device=device,
        use_sparse_features=use_sparse,
        sae_dict_size=4096,
        sae_k=128,
        sae_weight=0.1,
        cp_path='xlsr2_300m.pt'
    )
    
    # Load weights
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    print(f"   ✓ Model loaded")
    
    print(f"\n3. Testing interpretability methods...")
    
    # Create dummy input (batch_size=2, length=64600)
    dummy_audio = torch.randn(2, 64600)
    
    print(f"   Input shape: {dummy_audio.shape}")
    
    # Test forward pass with interpretability
    with torch.no_grad():
        try:
            output, interp = model(dummy_audio, return_interpretability=True)
            
            if interp is None:
                print(f"   ✗ Model returned None for interpretability info")
                print(f"\n{'='*70}")
                print(f"RESULT: Model CANNOT provide interpretability")
                print(f"   Reason: use_sparse_features=False (using reconstructed features)")
                print(f"   Solution: Retrain with use_sparse_features=True")
                print(f"{'='*70}")
                return False
            
            print(f"   ✓ Interpretability info returned")
            print(f"\n4. Analyzing interpretability info...")
            
            print(f"   Available keys: {list(interp.keys())}")
            
            # Check each component
            if 'avg_activation' in interp:
                avg_act = interp['avg_activation']
                print(f"   ✓ avg_activation: shape={avg_act.shape}, range=[{avg_act.min():.4f}, {avg_act.max():.4f}]")
            
            if 'sparse_features' in interp:
                sparse = interp['sparse_features']
                sparsity = (sparse > 0).float().mean()
                print(f"   ✓ sparse_features: shape={sparse.shape}, sparsity={sparsity*100:.2f}%")
            
            if 'top20_features' in interp:
                top20 = interp['top20_features']
                print(f"   ✓ top20_features: shape={top20.shape}")
                print(f"      Sample top 5 features: {top20[0, :5].tolist()}")
            
            if 'sparsity' in interp:
                sp = interp['sparsity']
                print(f"   ✓ sparsity: {sp[0]*100:.2f}% (sample 1), {sp[1]*100:.2f}% (sample 2)")
            
            # Evaluate quality
            print(f"\n5. Evaluating neuron quality...")
            
            if 'sparse_features' in interp:
                sparse_feat = interp['sparse_features']
                B, T, D = sparse_feat.shape
                
                # Check activation diversity
                active_features = (sparse_feat > 0).any(dim=0).any(dim=0)
                num_active = active_features.sum().item()
                diversity_ratio = num_active / D
                
                print(f"   Active features: {num_active}/{D} ({diversity_ratio*100:.1f}%)")
                
                # Check sparsity
                sparsity_val = (sparse_feat > 0).float().mean().item()
                print(f"   Overall sparsity: {sparsity_val*100:.2f}%")
                
                # Quality assessment
                print(f"\n{'='*70}")
                print(f"INTERPRETABILITY ASSESSMENT:")
                print(f"{'='*70}")
                
                score = 0
                
                if sparsity_val > 0.005 and sparsity_val < 0.25:
                    print(f"✓ Good sparsity level ({sparsity_val*100:.2f}%)")
                    score += 1
                else:
                    print(f"⚠ Sparsity not ideal ({sparsity_val*100:.2f}%, target: 0.5-25%)")
                
                if diversity_ratio > 0.3:
                    print(f"✓ Good feature diversity ({diversity_ratio*100:.1f}%)")
                    score += 1
                else:
                    print(f"⚠ Limited feature diversity ({diversity_ratio*100:.1f}%)")
                
                if D == 4096 and use_sparse:
                    print(f"✓ Using sparse features (dict_size=4096)")
                    score += 1
                else:
                    print(f"⚠ Not using sparse features or incorrect dict size")
                
                print(f"\nScore: {score}/3")
                
                if score >= 2:
                    print(f"\n✓ Model CAN generate meaningful interpretable neurons!")
                    print(f"  You can use analyze_sae_neurons.py for detailed analysis")
                else:
                    print(f"\n⚠ Model has LIMITED interpretability")
                    print(f"  Consider retraining with better hyperparameters")
                
                print(f"{'='*70}")
                return score >= 2
            
        except Exception as e:
            print(f"   ✗ Error during forward pass: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    return True


if __name__ == '__main__':
    test_interpretability()
