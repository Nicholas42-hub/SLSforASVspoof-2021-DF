#!/usr/bin/env python3
"""
Evaluate overlapping windows on ASVspoof 2021 LA dataset
Compare boundary discontinuity with original model
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
from model_window_topk import Model
from tqdm import tqdm

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
            # Load with torchaudio (more robust than librosa)
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
            # If file is corrupted, return zero tensor (will be filtered out)
            x_inp = Tensor(np.zeros(self.cut))
            return x_inp, utt_id

def compute_temporal_stability(features, window_size=8):
    """
    Compute temporal stability metrics
    
    Args:
        features: (B, T, D) tensor of binary feature activations
        window_size: window size for boundary detection
        
    Returns:
        dict with stability metrics
    """
    B, T, D = features.shape
    
    # Compute Jaccard similarity between consecutive timesteps
    jaccard_scores = []
    for t in range(T - 1):
        features_t = features[:, t, :]  # (B, D)
        features_t1 = features[:, t + 1, :]
        
        intersection = (features_t * features_t1).sum(dim=1)
        union = ((features_t + features_t1) > 0).float().sum(dim=1)
        
        jaccard = (intersection / (union + 1e-8)).mean().item()
        jaccard_scores.append(jaccard)
    
    # Identify boundary positions (stride = window_size // 2 for 50% overlap)
    stride = window_size // 2
    boundary_positions = set()
    pos = stride - 1
    while pos < T - 1:
        boundary_positions.add(pos)
        pos += stride
    
    # Separate interior vs boundary
    interior_scores = []
    boundary_scores = []
    
    for i, score in enumerate(jaccard_scores):
        if i in boundary_positions:
            boundary_scores.append(score)
        else:
            interior_scores.append(score)
    
    interior_mean = np.mean(interior_scores) if interior_scores else 0
    boundary_mean = np.mean(boundary_scores) if boundary_scores else 0
    
    return {
        'interior_stability': interior_mean,
        'boundary_stability': boundary_mean,
        'discontinuity': (interior_mean - boundary_mean) / (interior_mean + 1e-8),
        'all_jaccard': jaccard_scores,
        'boundary_positions': sorted(list(boundary_positions))
    }

def evaluate_model(model, device, eval_loader, num_samples=1000):
    """
    Evaluate model and compute stability metrics
    """
    model.eval()
    
    all_stability = []
    valid_samples = 0
    
    print(f"\n{'='*80}")
    print(f"Evaluating on up to {num_samples} samples...")
    print(f"{'='*80}\n")
    
    with torch.no_grad():
        for batch_idx, (batch_x, utt_ids) in enumerate(tqdm(eval_loader)):
            if valid_samples >= num_samples:
                break
            
            batch_x = batch_x.to(device)
            
            # Skip batches with all-zero tensors (corrupted files)
            if torch.all(batch_x == 0):
                continue
            
            try:
                # Forward pass
                score, _, sae_features = model(batch_x, return_sae_features=True)
            except Exception as e:
                print(f"\nSkipping batch {batch_idx} due to error: {e}")
                continue
            
            valid_samples += batch_x.size(0)
            
            # Get binary activation pattern (B, T, dict_size)
            active_features = (sae_features != 0).float()
            
            # Compute stability for this batch
            stability = compute_temporal_stability(active_features, window_size=8)
            all_stability.append(stability)
    
    # Aggregate stability metrics
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
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--database_path', type=str, default='/data/projects/punim2637/nnliang/Datasets/ASVspoof2021_LA_eval')
    parser.add_argument('--protocol_path', type=str, default='/data/projects/punim2637/nnliang/Datasets/keys/LA/CM/trial_metadata.txt')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--output', type=str, default='overlap_eval_results.json')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from: {args.model_path}")
    
    # Create dummy args
    class Args:
        def __init__(self):
            pass
    
    model = Model(
        args=Args(),
        device=device,
        use_sae=True,
        sae_dict_size=4096,
        sae_k=128,
        sae_window_size=8
    )
    
    # Load checkpoint
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Handle DataParallel state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Remove 'module.' prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('module.', '')
        new_state_dict[new_key] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    model = model.to(device)
    
    print("✓ Model loaded successfully")
    
    # Prepare dataset
    print(f"\nPreparing dataset from: {args.database_path}")
    
    # Get all FLAC files directly
    flac_dir = os.path.join(args.database_path, 'flac')
    if not os.path.exists(flac_dir):
        print(f'Error: FLAC directory not found: {flac_dir}')
        return
    
    file_train = [f.replace('.flac', '') for f in os.listdir(flac_dir) if f.endswith('.flac')]
    
    print(f"Found {len(file_train)} files")
    
    eval_set = Dataset_ASVspoof2021_eval(
        list_IDs=file_train,
        base_dir=args.database_path
    )
    
    eval_loader = DataLoader(
        eval_set,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,  # Drop last incomplete batch to avoid issues
        pin_memory=True,
        num_workers=0
    )
    _torchaudio(
        list_IDs=file_train,
        base_dir=args.database_path
    )
    
    eval_loader = DataLoader(
        eval_set,
        batch_size=args.batch_size,
        shuffle=False,
        drop_lSamples analyzedue,
        num_workers=4
    print(f"\n📈 Temporal Stability (with 50% overlap):")
    print(f"  Interior stability: {results['temporal_stability']['interior']:.4f}")
    print(f"  Boundary stability: {results['temporal_stability']['boundary']:.4f}")
    print(f"  Discontinuity: {results['temporal_stability']['discontinuity_pct']:.1f}%")
    
    # Comparison with expected values
    print(f"\n💡 Comparison:")
    print(f"  Expected (non-overlapping): ~25% discontinuity")
    print(f"  Expected (overlapping): ~6-8% discontinuity")
    print(f"  Observed: {results['temporal_stability']['discontinuity_pct']:.1f}%")
    
    disc = results['temporal_stability']['discontinuity_pct']
    if disc < 10:
        print(f"  ✅ EXCELLENT: Achieved target (<10%)")
    elif disc < 15:
        print(f"  ✅ GOOD: Significant improvement")
    else:
        print(f"  ⚠️  HIGHER THAN EXPECTED: May need tuning")
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {args.output}")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    main()
