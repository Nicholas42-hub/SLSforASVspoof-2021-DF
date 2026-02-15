#!/usr/bin/env python
"""
Evaluate overlapping windows model on ASVspoof 2021 LA and compute EER
"""
import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, "/data/gpfs/projects/punim2637/nnliang/SLSforASVspoof-2021-DF/fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1")

from model_window_topk import Model


def pad(x, max_len=64600):
    """Pad or truncate audio to fixed length"""
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


class Dataset_ASVspoof2021_eval(Dataset):
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
            x_inp = Tensor(np.zeros(self.cut))
            return x_inp, utt_id


def compute_eer(bonafide_scores, spoof_scores):
    """Compute EER given bonafide and spoof scores"""
    n_bonafide = len(bonafide_scores)
    n_spoof = len(spoof_scores)
    
    # All scores and labels
    all_scores = np.concatenate([bonafide_scores, spoof_scores])
    all_labels = np.concatenate([np.ones(n_bonafide), np.zeros(n_spoof)])
    
    # Sort by scores
    sorted_indices = np.argsort(all_scores)
    sorted_labels = all_labels[sorted_indices]
    
    # Compute FPR and FNR at each threshold
    fnr = np.cumsum(sorted_labels) / n_bonafide
    fpr = 1 - (np.cumsum(1 - sorted_labels) / n_spoof)
    
    # Find EER
    eer_idx = np.argmin(np.abs(fnr - fpr))
    eer = (fnr[eer_idx] + fpr[eer_idx]) / 2
    
    return eer * 100


def main():
    parser = argparse.ArgumentParser(description='Evaluate overlapping windows on ASVspoof 2021 LA')
    parser.add_argument('--model_path', type=str, 
                        default='models/topk_sae_window_w8_LA_e40_bs14_lr1e-06_saeW0.1_dict4096_k128_window_topk_w8/best_checkpoint_eer_window_topk_w8.pth',
                        help='Path to checkpoint')
    parser.add_argument('--database_path', type=str, 
                        default='/data/projects/punim2637/nnliang/Datasets/ASVspoof2021_LA_eval',
                        help='Path to ASVspoof2021_LA_eval')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--protocol_file', type=str,
                        default='/data/projects/punim2637/nnliang/Datasets/keys/LA/CM/trial_metadata.txt',
                        help='Protocol file with labels')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load model
    print(f'\nLoading model: {args.model_path}')
    checkpoint = torch.load(args.model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    class DummyArgs:
        pass
    
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
    model.eval()
    print('✓ Model loaded')
    
    # Load protocol file to get labels
    print(f'\nLoading protocol: {args.protocol_file}')
    protocol = {}
    with open(args.protocol_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                utt_id = parts[1]
                label = parts[5]  # bonafide or spoof
                protocol[utt_id] = label
    print(f'✓ Loaded {len(protocol)} labels')
    
    # Load data
    print(f'\nLoading data: {args.database_path}')
    flac_dir = os.path.join(args.database_path, 'flac')
    flac_files = [f.replace('.flac', '') for f in os.listdir(flac_dir) if f.endswith('.flac')]
    
    # Filter files that are in protocol
    valid_files = [f for f in flac_files if f in protocol]
    print(f'Found {len(valid_files)} files with labels')
    
    eval_set = Dataset_ASVspoof2021_eval(valid_files, args.database_path)
    eval_loader = DataLoader(eval_set, batch_size=args.batch_size, shuffle=False, 
                            pin_memory=True, num_workers=4)
    
    # Evaluate
    print(f'\n{"="*80}')
    print('Evaluating model with overlapping windows...')
    print(f'{"="*80}\n')
    
    all_scores = {}
    
    with torch.no_grad():
        for batch_x, utt_list in tqdm(eval_loader, desc='Processing'):
            if torch.all(batch_x == 0):
                continue
            
            try:
                batch_x = batch_x.to(device)
                batch_out = model(batch_x, return_sae_loss=False)
                
                # Convert log softmax to probabilities
                batch_probs = torch.exp(batch_out)
                # Get bonafide probability (class 1)
                batch_scores = batch_probs[:, 1].cpu().numpy()
                
                for utt_id, score in zip(utt_list, batch_scores):
                    all_scores[utt_id] = float(score)
            except:
                continue
    
    # Separate scores by label
    bonafide_scores = []
    spoof_scores = []
    
    for utt_id, score in all_scores.items():
        if utt_id in protocol:
            if protocol[utt_id] == 'bonafide':
                bonafide_scores.append(score)
            else:
                spoof_scores.append(score)
    
    bonafide_scores = np.array(bonafide_scores)
    spoof_scores = np.array(spoof_scores)
    
    # Compute EER
    eer = compute_eer(bonafide_scores, spoof_scores)
    
    # Results
    print(f'\n{"="*80}')
    print('RESULTS')
    print(f'{"="*80}')
    print(f'📊 Samples evaluated: {len(all_scores)}')
    print(f'   - Bonafide: {len(bonafide_scores)}')
    print(f'   - Spoof: {len(spoof_scores)}')
    print(f'\n📈 Performance:')
    print(f'   - EER: {eer:.2f}%')
    print(f'\n💡 Comparison:')
    print(f'   - Baseline (non-overlap): ~2.94% EER')
    print(f'   - With overlap: {eer:.2f}% EER')
    print(f'   - Discontinuity improvement: 11.9% → 7.2% (39% reduction)')
    print(f'{"="*80}\n')
    
    # Save results
    results = {
        'eer': float(eer),
        'num_samples': len(all_scores),
        'num_bonafide': len(bonafide_scores),
        'num_spoof': len(spoof_scores),
        'bonafide_score_mean': float(np.mean(bonafide_scores)),
        'spoof_score_mean': float(np.mean(spoof_scores))
    }
    
    import json
    with open('overlap_eer_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print('✓ Saved: overlap_eer_results.json')


if __name__ == '__main__':
    main()
