#!/usr/bin/env python3
"""
Evaluate overlapping windows temporal stability on ASVspoof 2021 LA
"""
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from model_window_topk import Model
import os
import sys

sys.path.insert(0, "/data/gpfs/projects/punim2637/nnliang/SLSforASVspoof-2021-DF/fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1")


def pad(x, max_len=64600):
    """Pad or truncate audio"""
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
        except:
            x_inp = Tensor(np.zeros(self.cut))
            return x_inp, utt_id


def compute_temporal_stability(features, window_size=8):
    """Compute temporal stability with 50% overlap boundaries"""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--database_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_samples', type=int, default=5000)
    parser.add_argument('--output', type=str, default='overlap_eval_5k_results.json')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load model
    print(f'\nLoading model: {args.model_path}')
    checkpoint = torch.load(args.model_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    class DummyArgs:
        pass
    
    model = Model(
        args=DummyArgs(),
        device=device,
        use_sae=True,
        sae_dict_size=4096,
        sae_k=128,
        sae_window_size=8
    )
    model = nn.DataParallel(model).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print('✓ Model loaded')
    
    # Load data
    print(f'\nLoading data: {args.database_path}')
    flac_dir = os.path.join(args.database_path, 'flac')
    flac_files = [f.replace('.flac', '') for f in os.listdir(flac_dir) if f.endswith('.flac')]
    print(f'Found {len(flac_files)} files')
    
    eval_set = Dataset_ASVspoof2021_eval_torchaudio(flac_files, args.database_path)
    eval_loader = DataLoader(eval_set, batch_size=args.batch_size, shuffle=False, 
                            pin_memory=True, num_workers=4)
    
    # Evaluate
    print(f'\n{"="*80}')
    print(f'Evaluating temporal stability on {args.num_samples} samples...')
    print(f'{"="*80}\n')
    
    all_stability = []
    valid_samples = 0
    
    with torch.no_grad():
        for batch_x, _ in tqdm(eval_loader):
            if valid_samples >= args.num_samples:
                break
                
            if torch.all(batch_x == 0):
                continue
            
            try:
                batch_x = batch_x.to(device)
                # Forward pass with interpretability info
                output = model(batch_x, return_sae_loss=False, return_interpretability=True)
                
                # Access sparse features from model's stored attribute
                # Shape: [B, T, dict_size]
                sae_features = model.module.last_sparse_features if hasattr(model, 'module') else model.last_sparse_features
                
                if sae_features is not None:
                    active_features = (sae_features > 0).float()
                    stability = compute_temporal_stability(active_features, window_size=8)
                    all_stability.append(stability)
                    valid_samples += sae_features.size(0)
            except Exception as e:
                continue
    
    # Results
    interior_stab = np.mean([s['interior_stability'] for s in all_stability])
    boundary_stab = np.mean([s['boundary_stability'] for s in all_stability])
    discontinuity = np.mean([s['discontinuity'] for s in all_stability])
    
    results = {
        'num_samples': valid_samples,
        'temporal_stability': {
            'interior': float(interior_stab),
            'boundary': float(boundary_stab),
            'discontinuity_pct': float(discontinuity * 100)
        }
    }
    
    # Print
    print(f'\n{"="*80}')
    print('RESULTS')
    print(f'{"="*80}')
    print(f'\n📊 Samples analyzed: {valid_samples}')
    print(f'\n📈 Temporal Stability (50% overlap):')
    print(f'  Interior: {interior_stab:.4f}')
    print(f'  Boundary: {boundary_stab:.4f}')
    print(f'  Discontinuity: {discontinuity*100:.1f}%')
    print(f'\n💡 Comparison:')
    print(f'  Expected (non-overlap): ~25%')
    print(f'  Expected (overlap): ~6-8%')
    print(f'  Observed: {discontinuity*100:.1f}%')
    
    # Save
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n✓ Saved: {args.output}')
    print(f'{"="*80}\n')


if __name__ == '__main__':
    main()
