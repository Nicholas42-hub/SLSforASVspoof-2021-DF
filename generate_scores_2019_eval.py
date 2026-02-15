#!/usr/bin/env python
"""
Generate score file for ASVspoof 2019 LA evaluation set
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

from data_utils_SSL import genSpoof_list, Dataset_ASVspoof2019_train
from model import Model

def generate_scores(model, eval_loader, device):
    """Generate scores for each file"""
    model.eval()
    
    file_ids = []
    scores = []
    
    with torch.no_grad():
        for batch_x, utt_list in tqdm(eval_loader, desc='Generating scores'):
            batch_x = batch_x.to(device)
            
            # Get model output
            batch_out = model(batch_x, return_sae_loss=False)
            
            # Convert log softmax to probabilities
            batch_probs = torch.exp(batch_out)
            # Get bonafide probability (class 1) as score
            batch_scores = batch_probs[:, 1].cpu().numpy()
            
            scores.extend(batch_scores.tolist())
            file_ids.extend(utt_list)
    
    return file_ids, scores

def main():
    parser = argparse.ArgumentParser(description='Generate scores for ASVspoof 2019 LA eval')
    parser.add_argument('--model_path', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--database_path', type=str, required=True, help='Path to ASVspoof2019 LA')
    parser.add_argument('--protocols_path', type=str, required=True, help='Path to protocols')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--output_file', type=str, required=True, help='Output score file')
    args = parser.parse_args()
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    
    # Load checkpoint
    print(f'\nLoading checkpoint: {args.model_path}')
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Get args from checkpoint
    if 'args' in checkpoint:
        model_args = argparse.Namespace(**checkpoint['args'])
    else:
        print('Warning: No args in checkpoint, using defaults')
        model_args = argparse.Namespace(
            sae_dict_size=4096,
            sae_k=128,
            sae_weight=0.1,
            use_sparse_features=True
        )
    
    # Create model
    print('\nCreating model...')
    model = Model(
        args=model_args,
        device=device,
        cp_path='xlsr2_300m.pt',
        use_sae=True,
        use_sparse_features=getattr(model_args, 'use_sparse_features', False),
        sae_dict_size=getattr(model_args, 'sae_dict_size', 4096),
        sae_k=getattr(model_args, 'sae_k', 128),
        sae_weight=getattr(model_args, 'sae_weight', 0.1)
    )
    model = nn.DataParallel(model).to(device)
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=False)
    print(f'✓ Model loaded from epoch {checkpoint.get("epoch", "unknown")}')
    if 'best_val_eer' in checkpoint:
        print(f'  Best validation EER: {checkpoint["best_val_eer"]:.2f}%')
    
    # Load evaluation data
    print(f'\nLoading evaluation data...')
    
    # Read protocol file and parse file IDs
    protocol_file = os.path.join(args.protocols_path, 'ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt')
    file_eval = []
    with open(protocol_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                file_id = parts[1]  # Second column is the file ID
                file_eval.append(file_id)
    
    print(f'Found {len(file_eval)} evaluation files')
    
    # Create dummy labels (not used during evaluation)
    d_label_eval = {key: 0 for key in file_eval}
    
    eval_set = Dataset_ASVspoof2019_train(
        model_args,
        list_IDs=file_eval,
        labels=d_label_eval,
        base_dir=os.path.join(args.database_path, 'ASVspoof2019_LA_eval/'),
        algo=3
    )
    
    eval_loader = DataLoader(
        eval_set,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=8
    )
    
    # Generate scores
    print('\nGenerating scores...')
    file_ids, scores = generate_scores(model, eval_loader, device)
    
    # Save scores in the format expected by evaluation script
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    print(f'\nSaving scores to: {args.output_file}')
    
    with open(args.output_file, 'w') as f:
        for file_id, score in zip(file_ids, scores):
            f.write(f'{file_id} {score:.6f}\n')
    
    print(f'✓ Saved {len(scores)} scores')
    print(f'  Score range: [{min(scores):.4f}, {max(scores):.4f}]')
    print(f'  Mean score: {np.mean(scores):.4f}')
    print(f'\nScore file generated successfully!')
    print(f'\nTo evaluate, run:')
    print(f'  python eval_metrics_LA.py {args.output_file}')

if __name__ == '__main__':
    main()
