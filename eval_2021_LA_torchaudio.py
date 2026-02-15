#!/usr/bin/env python
"""
Evaluate trained model on ASVspoof 2021 LA evaluation set - using torchaudio
Supports: baseline, window_topk, and cpc models
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


def evaluate_2021_LA(model, eval_loader, device, model_type='baseline'):
    """Evaluate model and return scores"""
    model.eval()
    
    file_ids = []
    scores = []
    
    total_batches = len(eval_loader)
    
    with torch.no_grad():
        for batch_idx, (batch_x, utt_list) in enumerate(eval_loader):
            batch_x = batch_x.to(device)
            
            # Get model output based on model type
            if model_type == 'baseline':
                batch_out = model(batch_x, return_sae_loss=False)
            elif model_type == 'window_topk':
                batch_out = model(batch_x, return_sae_loss=False)
            elif model_type == 'cpc':
                batch_out = model(batch_x, return_sae_loss=False, return_cpc_loss=False)
            else:
                batch_out = model(batch_x)
            
            # Convert log softmax to probabilities
            batch_probs = torch.exp(batch_out)
            # Get bonafide probability (class 1)
            batch_scores = batch_probs[:, 1].cpu().numpy()
            
            scores.extend(batch_scores.tolist())
            file_ids.extend(utt_list)
            
            # Progress update every 50 batches
            if (batch_idx + 1) % 50 == 0:
                progress = (batch_idx + 1) / total_batches * 100
                print(f'Progress: {batch_idx+1}/{total_batches} batches ({progress:.1f}%)', flush=True)
    
    return file_ids, scores


def main():
    parser = argparse.ArgumentParser(description='Evaluate on ASVspoof 2021 LA with torchaudio')
    parser.add_argument('--model_path', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--database_path', type=str, required=True, help='Path to ASVspoof2021_LA_eval')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--output_file', type=str, default='scores/scores_2021_LA.txt', help='Output score file')
    parser.add_argument('--model_type', type=str, default='baseline', 
                        choices=['baseline', 'window_topk', 'cpc'],
                        help='Model type: baseline, window_topk, or cpc')
    parser.add_argument('--sae_k', type=int, default=128, help='SAE K value')
    parser.add_argument('--window_size', type=int, default=8, help='Window size (for window_topk and cpc)')
    parser.add_argument('--cpc_hidden_dim', type=int, default=256, help='CPC hidden dimension (for cpc model)')
    print(f'Model type: {args.model_type}')
    
    # Load checkpoint
    print(f'\nLoading checkpoint: {args.model_path}')
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Detect use_sparse_features from checkpoint
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # Check classifier input dimension
    use_sparse_features = True  # Default
    for key in state_dict.keys():
        if 'classifier.1.weight' in key or 'module.classifier.1.weight' in key:
            classifier_input_dim = state_dict[key].shape[1]
            use_sparse_features = (classifier_input_dim == 4096)
            print(f'Detected use_sparse_features={use_sparse_features} (classifier input: {classifier_input_dim})')
            break
    
    # Create dummy args
    class DummyArgs:
        pass
    
    dummy_args = DummyArgs()
    
    # Import and create model based on type
    if args.model_type == 'baseline':
        from model import Model
        print(f'\nCreating baseline model with sae_k={args.sae_k}, use_sparse_features={use_sparse_features}')
        model = Model(
            args=dummy_args,
            device=device,
            cp_path='xlsr2_300m.pt',
            use_sae=True,
            use_sparse_features=use_sparse_features,
            sae_dict_size=4096,
            sae_k=args.sae_k,
            sae_weight=0.1
        )
    
    elif args.model_type == 'window_topk':
        # Import window_topk model
        spec = importlib.util.spec_from_file_location("model_window", "model_window_topk.py")
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
