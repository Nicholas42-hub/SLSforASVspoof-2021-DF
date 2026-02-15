#!/usr/bin/env python
"""
Evaluate overlapping windows temporal stability on ASVspoof 2021 LA evaluation set
"""
import argparse
import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import importlib.util

# Add fairseq path
sys.path.insert(0, "/data/gpfs/projects/punim2637/nnliang/SLSforASVspoof-2021-DF/fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1")


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
        self.cut = 64600  # ~4 sec audio at 16kHz
    
    def __len__(self):
        return len(self.list_IDs)
    
    def __getitem__(self, index):
        utt_id = self.list_IDs[index]
        flac_path = os.path.join(self.base_dir, 'flac', f'{utt_id}.flac')
        
        try:
            # Load with torchaudio
            waveform, sample_rate = torchaudio.load(flac_path)
            
            # Convert to mono if needed
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # Resample if needed
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
            
            # Convert to numpy and pad
            X = waveform.squeeze().numpy()
            X_pad = pad(X, self.cut)
            x_inp = Tensor(X_pad)
            
            return x_inp, utt_id
        
        except Exception as e:
            # If file is corrupted, return zero tensor and still include the ID
            print(f"Warning: Failed to load {utt_id}: {e}")
            x_inp = Tensor(np.zeros(self.cut))
            return x_inp, utt_id


def compute_temporal_stability(features, window_size=8):
    """Compute temporal stability metrics with 50% overlap"""
    B, T, D = features.shape
    stride = window_size // 2
    
    jaccard_scores = []
    for t in range(T - 1):
        features_t = features[:, t, :]
        features_t1 = features[:, t + 1, :]
        intersection = (features_t * features_t1).sum(dim=1)
        union = ((features_t + features_t1) > 0).float().sum(dim=1)
        jaccard = (intersection / (union + 1e-8)).mean().item()
        jaccard_scores.append(jaccard)
    
    boundary_positions = set(range(stride-1, T-1, stride))
    interior_scores = [s for i, s in enumerate(jaccard_scores) if i not in boundary_positions]
    boundary_scores = [s for i, s in enumerate(jaccard_scores) if i in boundary_positions]
    
    interior_mean = np.mean(interior_scores) if interior_scores else 0
    boundary_mean = np.mean(boundary_scores) if boundary_scores else 0
    
    return {
        'interior_stability': interior_mean,
        'boundary_stability': boundary_mean,
        'discontinuity': (interior_mean - boundary_mean) / (interior_mean + 1e-8)
    }


def evaluate_2021_LA(model, eval_loader, device, num_samples=5000):
    """Evaluate model and compute temporal stability"""
    model.eval()
    
    all_stability = []
    valid_samples = 0
    
    with torch.no_grad():
        for batch_idx, (batch_x, utt_list) in enumerate(tqdm(eval_loader)):
            if valid_samples >= num_samples:
                break
                
            batch_x = batch_x.to(device)
            
            if torch.all(batch_x == 0):
                continue
            
            try:
                # Get SAE features
                _, _, sae_features = model(batch_x, return_sae_features=True)
                active_features = (sae_features != 0).float()
                stability = compute_temporal_stability(active_features, window_size=8)
                all_stability.append(stability)
                valid_samples += batch_x.size(0)
            except Exception as e:
                print(f"\nError in batch {batch_idx}: {e}")
                continue
    
    interior_stab = np.mean([s['interior_stability'] for s in all_stability])
    boundary_stab = np.mean([s['boundary_stability'] for s in all_stability])
    discontinuity = np.mean([s['discontinuity'] for s in all_stability])
    
    return {
        'num_samples': valid_samples,
        'temporal_stability': {
            'interior': float(interior_stab),
            'boundary': float(boundary_stab),
            'discontinuity_pct': float(discontinuity * 100)
        }
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate overlapping windows temporal stability')
    parser.add_argument('--model_path', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--database_path', type=str, required=True, help='Path to ASVspoof2021_LA_eval')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--output_file', type=str, default='overlap_eval_5k_results.json', help='Output JSON file')
    parser.add_argument('--num_samples', type=int, default=5000, help='Number of samples to evaluate')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Import window_topk model
    from model_window_topk import Model
    
    print(f'\nLoading model from: {args.model_path}')
    checkpoint = torch.load(args.model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Create dummy args
    class DummyArgs:
        pass
    
    print(f'\nCreating Window TopK model with overlapping windows')
    model = Model(
        args=DummyArgs(),
        device=device,
        cp_path='xlsr2_300m.pt',
        use_sae=True,
        use_sparse_features=True,
        sae_dict_size=4096,
        sae_k=128,
        sae_window_size=8,
        sae_weight=0.1
    )
    
    model = nn.DataParallel(model).to(device)
    model.load_state_dict(state_dict, strict=False)
    print(f'✓ Model loaded')
    
    # Load evaluation data
    print(f'\nLoading evaluation data from: {args.database_path}')
        model_window_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_window_module)
        
        print(f'\nCreating Window TopK model:')
        print(f'  - sae_k={args.sae_k}')
        print(f'  - window_size={args.window_size}')
        print(f'  - use_sparse_features={use_sparse_features}')
        
        model = model_window_module.Model(
            args=dummy_args,
            device=device,
            cp_path='xlsr2_300m.pt',
            use_sae=True,
            use_sparse_features=use_sparse_features,
            sae_dict_size=4096,
            sae_k=args.sae_k,
            sae_weight=0.1,
            window_size=args.window_size
        )
    
    elif args.model_type == 'cpc':
        # Import CPC model
        spec = importlib.util.spec_from_file_location("model_cpc", "model_topk contrastive loss.py")
        model_cpc_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_cpc_module)
        
        print(f'\nCreating CPC model:')
        print(f'  - sae_k={args.sae_k}')
        print(f'  - window_size={args.window_size}')
        print(f'  - use_sparse_features={use_sparse_features}')
        print(f'  - use_cpc=True')
        print(f'  - cpc_hidden_dim={args.cpc_hidden_dim}')
        print' + '='*80)
    print('Starting evaluation...')
    print('='*80)
    file_ids, scores = evaluate_2021_LA(model, eval_loader, device, args.model_typ
        model = model_cpc_module.Model(
            args=dummy_args,
            device=device,
            cp_path='xlsr2_300m.pt',
            use_sae=True,
            use_sparse_features=use_sparse_features,
            sae_dict_size=4096,
            sae_k=args.sae_k,
            sae_weight=0.1,
            sae_window_size=args.window_size,  # CPC model uses sae_window_size
            use_cpc=True,
            cpc_hidden_dim=args.cpc_hidden_dim,
            cpc_weight=args.cpc_weight
        )
    
    model = nn.DataParallel(model).to(device)
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(state_dict, strict=False)
    
        sae_weight=0.1
    )
    model = nn.DataParallel(model).to(device)
    
    # Load state dict
    model.load_state_dict(state_dict, strict=False)
    print(f'✓ Model loaded from epoch {checkpoint.get("epoch", "unknown")}')
    if 'best_val_eer' in checkpoint:
        print(f'  Best validation EER: {checkpoint["best_val_eer"]:.2f}%')
    
    # Load evaluation data
    print(f'\nLoading evaluation data from: {args.database_path}')
    
    # Get all FLAC files
    flac_dir = os.path.join(args.database_path, 'flac')
    if not os.path.exists(flac_dir):
        print(f'Error: FLAC directory not found: {flac_dir}')
        return
    
    flac_files = [f.replace('.flac', '') for f in os.listdir(flac_dir) if f.endswith('.flac')]
    print(f'Found {len(flac_files)} files')
    
    eval_set = Dataset_ASVspoof2021_eval_torchaudio(
        list_IDs=flac_files,
        base_dir=args.database_path
    )
    
    eval_loader = DataLoader(
        eval_set,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=4  # Reduced for stability
    )
    
    # Evaluate
    print('\nEvaluating...')
    file_ids, scores = evaluate_2021_LA(model, eval_loader, device)
    
    # Save scores
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    print(f'\nSaving scores to: {args.output_file}')
    
    with open(args.output_file, 'w') as f:
        for file_id, score in zip(file_ids, scores):
            f.write(f'{file_id} {score:.6f}\n')
    
    print(f'✓ Saved {len(scores)} scores')
    print(f'  Score range: [{min(scores):.4f}, {max(scores):.4f}]')
    print(f'  Mean score: {np.mean(scores):.4f}')
    print('\nEvaluation completed!')


if __name__ == '__main__':
    main()
