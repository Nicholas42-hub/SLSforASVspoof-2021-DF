#!/usr/bin/env python
"""
Evaluate window topk model on ASVspoof 2021 LA evaluation set
"""
import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add fairseq path
sys.path.insert(0, "/data/gpfs/projects/punim2637/nnliang/SLSforASVspoof-2021-DF/fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1")

from data_utils_SSL import Dataset_ASVspoof2021_eval
from model_window_topk import Model

def evaluate_2021_LA(model, eval_loader, device):
    """Evaluate model and return scores"""
    model.eval()
    
    file_ids = []
    scores = []
    
    with torch.no_grad():
        for batch_x, utt_list in tqdm(eval_loader, desc='Evaluating'):
            batch_x = batch_x.to(device)
            
            # Get model output
            batch_out = model(batch_x, return_sae_loss=False)
            
            # Convert log softmax to probabilities
            batch_probs = torch.exp(batch_out)
            # Get bonafide probability (class 1)
            batch_scores = batch_probs[:, 1].cpu().numpy()
            
            scores.extend(batch_scores.tolist())
            file_ids.extend(utt_list)
    
    return file_ids, scores

def main():
    parser = argparse.ArgumentParser(description='Evaluate window topk on ASVspoof 2021 LA')
    parser.add_argument('--model_path', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--database_path', type=str, required=True, help='Path to ASVspoof2021_LA_eval')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--output_file', type=str, default='scores/scores_2021_LA_window.txt', help='Output score file')
    parser.add_argument('--sae_k', type=int, default=128, help='SAE K value')
    parser.add_argument('--window_size', type=int, default=8, help='Window size for topk')
    args = parser.parse_args()
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    
    # Load checkpoint
    print(f'\nLoading checkpoint: {args.model_path}')
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Detect use_sparse_features from checkpoint
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
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
    
    # Create model
    print(f'\nCreating Model with sae_k={args.sae_k}, window_size={args.window_size}')
    model = Model(
        args=dummy_args,
        device=device,
        cp_path='xlsr2_300m.pt',
        use_sae=True,
        use_sparse_features=use_sparse_features,
        sae_dict_size=4096,
        sae_k=args.sae_k,
        sae_window_size=args.window_size,
        sae_weight=0.1
    )
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Handle DataParallel wrapper
    if 'module.' in list(state_dict.keys())[0]:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        state_dict = new_state_dict
    
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    
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
    
    eval_set = Dataset_ASVspoof2021_eval(
        list_IDs=flac_files,
        base_dir=args.database_path
    )
    
    eval_loader = DataLoader(
        eval_set,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=0  # Avoid multiprocessing issues with FLAC
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
    print(f'\nTo compute EER, run:')
    print(f'python compute_eer_2021LA.py {args.output_file}')

if __name__ == '__main__':
    main()
