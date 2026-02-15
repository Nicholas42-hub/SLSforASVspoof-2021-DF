#!/usr/bin/env python
"""
Evaluate CPC model on ASVspoof 2021 LA evaluation set
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

# Import CPC model
import importlib.util
spec = importlib.util.spec_from_file_location("model_cpc", "model_topk contrastive loss.py")
model_cpc_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_cpc_module)

def evaluate_2021_LA(model, eval_loader, device):
    """Evaluate model and return scores"""
    model.eval()
    
    file_ids = []
    scores = []
    
    total_batches = len(eval_loader)
    
    with torch.no_grad():
        for batch_idx, (batch_x, utt_list) in enumerate(eval_loader):
            batch_x = batch_x.to(device)
            
            # Get model output (no SAE loss during eval)
            batch_out = model(batch_x, return_sae_loss=False, return_cpc_loss=False)
            
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
    parser = argparse.ArgumentParser(description='Evaluate CPC model on ASVspoof 2021 LA')
    parser.add_argument('--model_path', type=str, required=True, help='Path to CPC checkpoint')
    parser.add_argument('--database_path', type=str, required=True, help='Path to ASVspoof2021_LA_eval')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--output_file', type=str, default='scores/scores_cpc_2021_LA.txt', help='Output score file')
    parser.add_argument('--sae_k', type=int, default=128, help='SAE K value')
    parser.add_argument('--window_size', type=int, default=8, help='Window size')
    parser.add_argument('--cpc_hidden_dim', type=int, default=256, help='CPC hidden dimension')
    parser.add_argument('--cpc_weight', type=float, default=0.5, help='CPC loss weight')
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
    
    # Create CPC model
    print(f'\nCreating CPC model:')
    print(f'  - sae_k={args.sae_k}')
    print(f'  - window_size={args.window_size}')
    print(f'  - use_sparse_features={use_sparse_features}')
    print(f'  - use_cpc=True')
    print(f'  - cpc_hidden_dim={args.cpc_hidden_dim}')
    print(f'  - cpc_weight={args.cpc_weight}')
    
    model = model_cpc_module.Model(
        args=dummy_args,
        device=device,
        cp_path='xlsr2_300m.pt',
        use_sae=True,
        use_sparse_features=use_sparse_features,
        sae_dict_size=4096,
        sae_k=args.sae_k,
        sae_window_size=args.window_size,
        sae_weight=0.1,
        use_cpc=True,
        cpc_hidden_dim=args.cpc_hidden_dim,
        cpc_weight=args.cpc_weight,
        cpc_temperature=0.07,
        cpc_prediction_steps=[1, 2, 4]
    )
    model = nn.DataParallel(model).to(device)
    
    # Load state dict
    try:
        model.load_state_dict(state_dict)
        print('✓ Loaded full state dict')
    except:
        # Try loading with strict=False for missing keys
        model.load_state_dict(state_dict, strict=False)
        print('✓ Loaded state dict with strict=False')
    
    model.eval()
    
    # Load evaluation dataset
    print(f'\nLoading evaluation dataset from: {args.database_path}')
    
    # Get list of files from flac directory
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
        num_workers=0  # Use main process to avoid multiprocessing issues with librosa
    )
    
    print(f'Evaluation set size: {len(eval_set)} files')
    print(f'Number of batches: {len(eval_loader)}')
    
    # Evaluate
    print('\n' + '='*80)
    print('Starting evaluation...')
    print('='*80)
    
    file_ids, scores = evaluate_2021_LA(model, eval_loader, device)
    
    # Save scores
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    print(f'\nSaving scores to: {args.output_file}')
    with open(args.output_file, 'w') as f:
        for file_id, score in zip(file_ids, scores):
            f.write(f'{file_id} {score:.6f}\n')
    
    print(f'✓ Saved {len(scores)} scores')
    
    # Print statistics
    print('\n' + '='*80)
    print('Score Statistics:')
    print('='*80)
    print(f'Mean:   {np.mean(scores):.4f}')
    print(f'Std:    {np.std(scores):.4f}')
    print(f'Min:    {np.min(scores):.4f}')
    print(f'Max:    {np.max(scores):.4f}')
    print(f'Median: {np.median(scores):.4f}')
    
    print('\n✓ Evaluation complete!')
    print(f'\nTo compute EER and min-tDCF, run:')
    print(f'python compute_eer_2021LA.py {args.output_file}')

if __name__ == '__main__':
    main()
