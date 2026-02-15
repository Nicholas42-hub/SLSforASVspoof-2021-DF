import argparse
import sys
import os
import csv
from datetime import datetime
from typing import Optional
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from data_utils_SSL import (
    genSpoof_list,
    Dataset_ASVspoof2019_train,
    Dataset_ASVspoof2021_eval,
    Dataset_in_the_wild_eval,
)
from model import Model as ModelTopK
from model_window_topk import Model as ModelWindowTopK
from tensorboardX import SummaryWriter
from core_scripts.startup_config import set_random_seed
from tqdm import tqdm
from sklearn.metrics import roc_curve

def compute_eer(scores, labels):
    """
    Compute Equal Error Rate (EER)
    Args:
        scores: prediction scores (higher scores for positive class)
        labels: true labels (0 for genuine, 1 for spoof)
    Returns:
        eer: Equal Error Rate as percentage
        threshold: EER threshold
    """
    try:
        scores = np.array(scores)
        labels = np.array(labels)
        
        # Handle edge cases
        if len(scores) == 0 or len(labels) == 0:
            return 50.0, 0.5
        
        # Remove NaN values
        valid_mask = ~(np.isnan(scores) | np.isnan(labels))
        if not np.any(valid_mask):
            return 50.0, 0.5
            
        scores = scores[valid_mask]
        labels = labels[valid_mask]
        
        # Check if we have both classes
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            return 50.0, 0.5
        
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
        fnr = 1 - tpr
        
        # Find EER point (where FPR = FNR)
        abs_diff = np.abs(fpr - fnr)
        eer_idx = np.argmin(abs_diff)
        eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
        eer_threshold = thresholds[eer_idx] if eer_idx < len(thresholds) else 0.5
        
        return eer * 100, eer_threshold  # Return as percentage
        
    except Exception as e:
        print(f"Error computing EER: {e}")
        return 50.0, 0.5

def init_csv_log(log_path):
    """Initialize CSV log file with headers"""
    with open(log_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'epoch', 'timestamp', 'train_loss', 'train_eer', 'train_recon_loss',
            'val_loss', 'val_acc', 'val_eer', 'val_recon_loss'
        ])

def log_training_metrics(log_data, model_save_path):
    """Log training metrics to CSV file"""
    log_path = os.path.join(model_save_path, 'training_log.csv')
    
    with open(log_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            log_data['epoch'],
            log_data['timestamp'],
            log_data['train_loss'],
            log_data['train_eer'],
            log_data.get('train_recon_loss', 0),
            log_data['val_loss'],
            log_data['val_acc'],
            log_data['val_eer'],
            log_data.get('val_recon_loss', 0)
        ])

def evaluate_accuracy(dev_loader, model, device, criterion, quick_test=False):
    val_loss = 0.0
    num_correct = 0.0
    num_total = 0.0
    total_recon_loss = 0.0
    
    # For EER calculation
    all_scores = []
    all_labels = []
    
    model.eval()
    
    for i, (batch_x, batch_y) in enumerate(tqdm(dev_loader)):
        if quick_test and i >= 5:
            print("Quick test: breaking evaluation loop after 5 batches")
            break
            
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        
        with torch.no_grad():
            batch_out, sae_loss = model(batch_x, return_sae_loss=True)
            classification_loss = criterion(batch_out, batch_y)
            batch_loss = model.module.compute_total_loss(classification_loss, sae_loss)
                
            _, batch_pred = batch_out.max(dim=1)
            num_correct += (batch_pred == batch_y).sum(dim=0).item()
            val_loss += (batch_loss.item() * batch_size)
            
            # Track reconstruction loss
            if sae_loss is not None:
                total_recon_loss += (sae_loss.item() * batch_size)
            
            # Collect scores and labels for EER calculation
            # Convert log probabilities to probabilities and get spoof scores
            batch_probs = torch.exp(batch_out)  # Convert from log probabilities
            batch_scores = batch_probs[:, 1].cpu().numpy()  # Spoof scores (class 1)
            batch_labels = batch_y.cpu().numpy()
            
            # Filter out invalid scores
            valid_mask = ~(np.isnan(batch_scores) | np.isinf(batch_scores))
            if np.any(valid_mask):
                all_scores.extend(batch_scores[valid_mask].tolist())
                all_labels.extend(batch_labels[valid_mask].tolist())

    val_loss /= num_total
    acc = 100 * (num_correct / num_total)
    avg_recon_loss = total_recon_loss / num_total if num_total > 0 else 0.0
    
    # Compute EER
    if len(all_scores) > 0 and len(all_labels) > 0:
        eer, eer_threshold = compute_eer(all_scores, all_labels)
    else:
        eer, eer_threshold = 50.0, 0.5
        print("Warning: No valid scores for EER calculation, using default EER=50%")
    
    return val_loss, acc, eer, avg_recon_loss

def produce_evaluation_file(dataset, model, device, save_path, quick_test=False):
    data_loader = DataLoader(
        dataset, 
        batch_size=20,
        shuffle=False, 
        drop_last=False, 
        pin_memory=True,
        num_workers=6
    )
    model.eval()
    
    fname_list = []
    score_list = []
    
    for i, (batch_x, utt_id) in enumerate(tqdm(data_loader)):
        if quick_test and i >= 5:
            print("Quick test: breaking evaluation loop after 5 batches")
            break
            
        batch_size = batch_x.size(0)
        batch_x = batch_x.to(device)
        
        with torch.no_grad():
            batch_out = model(batch_x, return_sae_loss=False)
            # Convert log probabilities to probabilities and get spoof scores
            batch_probs = torch.exp(batch_out)
            batch_score = batch_probs[:, 1].data.cpu().numpy().ravel()
        
        # add outputs
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())
        
        with open(save_path, 'a+') as fh:
            for f, cm in zip(fname_list, score_list):
                fh.write('{} {}\n'.format(f, cm))
        fh.close()   
        
        # Reset lists to avoid memory accumulation
        fname_list = []
        score_list = []
        
    print('Scores saved to {}'.format(save_path))

def train_epoch(train_loader, model, optim, device, criterion, quick_test=False):
    running_loss = 0
    num_total = 0.0
    total_recon_loss = 0.0
    
    # For training EER calculation
    all_scores = []
    all_labels = []
    
    model.train()
    
    for i, (batch_x, batch_y) in enumerate(tqdm(train_loader)):
        if quick_test and i >= 5:
            print("Quick test: breaking training loop after 5 batches")
            break
            
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        
        # Forward pass
        batch_out, sae_loss = model(batch_x, return_sae_loss=True)
        classification_loss = criterion(batch_out, batch_y)
        batch_loss = model.module.compute_total_loss(classification_loss, sae_loss)
        
        running_loss += (batch_loss.item() * batch_size)
        
        # Track reconstruction loss
        if sae_loss is not None:
            total_recon_loss += (sae_loss.item() * batch_size)
        
        # Collect scores for training EER
        with torch.no_grad():
            batch_probs = torch.exp(batch_out.detach())
            batch_scores = batch_probs[:, 1].cpu().numpy()
            batch_labels = batch_y.cpu().numpy()
            
            valid_mask = ~(np.isnan(batch_scores) | np.isinf(batch_scores))
            if np.any(valid_mask):
                all_scores.extend(batch_scores[valid_mask].tolist())
                all_labels.extend(batch_labels[valid_mask].tolist())
        
        # Backward pass
        optim.zero_grad()
        batch_loss.backward()
        optim.step()
    
    running_loss /= num_total
    avg_recon_loss = total_recon_loss / num_total if num_total > 0 else 0.0
    
    # Compute training EER
    if len(all_scores) > 0 and len(all_labels) > 0:
        train_eer, _ = compute_eer(all_scores, all_labels)
    else:
        train_eer = 50.0
    
    return running_loss, train_eer, avg_recon_loss

def find_latest_checkpoint(model_save_path):
    """
    Find the latest checkpoint in the model save directory
    Supports both formats: 'checkpoint_epoch_X_...' and 'epoch_X_...'
    
    Returns:
        tuple: (checkpoint_path, epoch_number) or (None, None) if not found
    """
    if not os.path.exists(model_save_path):
        return None, None
    
    all_files = [f for f in os.listdir(model_save_path) if f.endswith('.pth')]
    
    # Extract epoch numbers from both naming patterns
    epoch_numbers = []
    for f in all_files:
        try:
            # Try new format: checkpoint_epoch_X_...
            if f.startswith('checkpoint_epoch_'):
                epoch_str = f.split('checkpoint_epoch_')[1].split('_')[0].split('.')[0]
                epoch_numbers.append((int(epoch_str), f))
            # Try alternative format: epoch_X_...
            elif f.startswith('epoch_'):
                parts = f.replace('.pth', '').split('_')
                if len(parts) >= 2:
                    epoch_str = parts[1]
                    if epoch_str.isdigit():
                        epoch_numbers.append((int(epoch_str), f))
        except:
            continue
    
    if not epoch_numbers:
        return None, None
    
    latest_epoch, latest_file = max(epoch_numbers, key=lambda x: x[0])
    return os.path.join(model_save_path, latest_file), latest_epoch

def list_available_checkpoints(model_save_path):
    """
    List all available checkpoints with their details
    """
    if not os.path.exists(model_save_path):
        print(f'Directory does not exist: {model_save_path}')
        return
    
    checkpoint_files = [f for f in os.listdir(model_save_path) 
                       if f.endswith('.pth')]
    
    if not checkpoint_files:
        print(f'No checkpoints found in: {model_save_path}')
        return
    
    print(f'\n{"="*70}')
    print(f'Available Checkpoints in {model_save_path}')
    print(f'{"="*70}')
    
    for f in sorted(checkpoint_files):
        checkpoint_path = os.path.join(model_save_path, f)
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            if isinstance(checkpoint, dict) and 'epoch' in checkpoint:
                print(f'\n{f}')
                print(f'   Epoch: {checkpoint["epoch"]}')
                if 'val_eer' in checkpoint:
                    print(f'   Val EER: {checkpoint["val_eer"]:.2f}%')
                if 'val_acc' in checkpoint:
                    print(f'   Val Acc: {checkpoint["val_acc"]:.2f}%')
                if 'best_val_eer' in checkpoint:
                    print(f'   Best EER: {checkpoint["best_val_eer"]:.2f}%')
            else:
                print(f'\n{f} (legacy format - weights only)')
        except Exception as e:
            print(f'\n{f} - Error loading: {e}')
    
    print(f'{"="*70}\n')


def _atomic_torch_save(obj, path: str) -> None:
    """Save a torch object atomically to avoid partial/corrupt checkpoints."""
    tmp_path = path + '.tmp'
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def _resolve_resume_path(model_save_path: str, explicit_model_path: Optional[str], resume: bool) -> Optional[str]:
    """Resolve which checkpoint to load when resuming.

    Priority:
    1) explicit_model_path if provided
    2) last_checkpoint.pth in model_save_path (when resume=True)
    3) best checkpoint in model_save_path (when resume=True)
    """
    if explicit_model_path:
        return explicit_model_path
    if not resume:
        return None
    last_path = os.path.join(model_save_path, 'last_checkpoint.pth')
    if os.path.exists(last_path):
        return last_path
    # Fallback to best checkpoint if present
    for name in (
        'best_checkpoint_eer.pth',
        'best_checkpoint.pth',
    ):
        best_path = os.path.join(model_save_path, name)
        if os.path.exists(best_path):
            return best_path
    # If user used --comment, best checkpoint name includes it.
    if os.path.exists(model_save_path):
        candidates = [
            f for f in os.listdir(model_save_path)
            if f.startswith('best_checkpoint_eer') and f.endswith('.pth')
        ]
        if candidates:
            candidates.sort()
            return os.path.join(model_save_path, candidates[-1])
    return None


def _infer_epoch_from_checkpoint_path(path: str) -> Optional[int]:
    """Infer 0-based epoch index from a checkpoint filename.

    Supports:
      - checkpoint_epoch_{N}*.pth
      - epoch_{N}*.pth
    """
    base = os.path.basename(path)
    try:
        if base.startswith('checkpoint_epoch_'):
            tail = base.split('checkpoint_epoch_', 1)[1]
            epoch_str = tail.split('_', 1)[0].split('.', 1)[0]
            return int(epoch_str)
        if base.startswith('epoch_'):
            tail = base.split('epoch_', 1)[1]
            epoch_str = tail.split('_', 1)[0].split('.', 1)[0]
            return int(epoch_str)
    except Exception:
        return None
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description='ASVspoof2021 SSL+SAE (SAE-only)')

    # Dataset
    parser.add_argument('--database_path', type=str, default='/root/autodl-tmp/CLAD/Datasets/LA/')
    parser.add_argument('--protocols_path', type=str, default='/root/autodl-tmp/CLAD/Datasets/LA/')
    parser.add_argument('--track', type=str, default='DF', choices=['LA', 'In-the-Wild', 'DF'])

    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=14)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    # SSL + SAE
    parser.add_argument('--cp_path', type=str,
                        default='/root/autodl-tmp/SLSforASVspoof-2021-DF/xlsr2_300m.pt',
                        help='Path to fairseq SSL checkpoint (e.g., XLSR2)')
    parser.add_argument('--sae_weight', type=float, default=0.1, help='Weight for SAE loss')
    parser.add_argument('--sae_dict_size', type=int, default=4096, help='Dictionary size for TopK SAE')
    parser.add_argument('--sae_k', type=int, default=128, help='Top-K for SAE sparsity')
    parser.add_argument('--use_window_topk', action='store_true', default=False, 
                        help='Use window-based TopK instead of per-timestep TopK')
    parser.add_argument('--sae_window_size', type=int, default=8, help='Window size for temporal TopK selection (only used with --use_window_topk)')
    parser.add_argument('--use_sparse_features', action='store_true', default=True, 
                        help='Use sparse features (4096-dim) instead of reconstructed features (1024-dim). Default: True for better interpretability')

    # Runtime
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--comment', type=str, default=None)
    parser.add_argument('--quick_test', action='store_true', default=False, help='Run a quick test with few batches')

    # Checkpoint / Eval
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--resume', action='store_true', default=False,
                        help='Resume training from last_checkpoint.pth in the run folder (or --model_path if provided).')
    parser.add_argument('--fresh_start', action='store_true', default=False, help='Ignore optimizer/epoch when loading')
    parser.add_argument('--is_eval', action='store_true', default=False, help='Evaluation mode')
    parser.add_argument('--eval_output', type=str, default=None)
    
    # RawBoost data augmentation parameters
    parser.add_argument('--algo', type=int, default=3)
    parser.add_argument('--nBands', type=int, default=5)
    parser.add_argument('--minF', type=int, default=20)
    parser.add_argument('--maxF', type=int, default=8000)
    parser.add_argument('--minBW', type=int, default=100)
    parser.add_argument('--maxBW', type=int, default=1000)
    parser.add_argument('--minCoeff', type=int, default=10)
    parser.add_argument('--maxCoeff', type=int, default=100)
    parser.add_argument('--minG', type=int, default=0)
    parser.add_argument('--maxG', type=int, default=0)
    parser.add_argument('--minBiasLinNonLin', type=int, default=5)
    parser.add_argument('--maxBiasLinNonLin', type=int, default=20)
    parser.add_argument('--N_f', type=int, default=5)
    parser.add_argument('--P', type=int, default=10)
    parser.add_argument('--g_sd', type=int, default=2)
    parser.add_argument('--SNRmin', type=int, default=10)
    parser.add_argument('--SNRmax', type=int, default=40)

    args = parser.parse_args()

    if args.resume and args.fresh_start:
        parser.error('Cannot use both --resume and --fresh_start. Choose one.')

    set_random_seed(args.seed, args)

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    # Experiment name
    topk_type = f'window_w{args.sae_window_size}' if args.use_window_topk else 'timestep'
    model_tag = f'topk_sae_{topk_type}_{args.track}_e{args.num_epochs}_bs{args.batch_size}_lr{args.lr}_saeW{args.sae_weight}_dict{args.sae_dict_size}_k{args.sae_k}'
    if args.comment:
        model_tag = model_tag + f'_{args.comment}'

    os.makedirs('models', exist_ok=True)
    model_save_path = os.path.join('models', model_tag)
    os.makedirs(model_save_path, exist_ok=True)

    # If the user is resuming from an explicit checkpoint path, keep saving into
    # that same run directory (so training truly continues “where it left off”).
    if args.resume and args.model_path:
        candidate_dir = os.path.dirname(args.model_path)
        if candidate_dir and os.path.isdir(candidate_dir):
            model_save_path = candidate_dir
            model_tag = os.path.basename(os.path.normpath(model_save_path))
            os.makedirs(model_save_path, exist_ok=True)
            print(f'--resume with --model_path: saving to existing run dir {model_save_path}')

    # Model - choose between per-timestep and window-based TopK
    if args.use_window_topk:
        print(f'Using Window-based TopK (window_size={args.sae_window_size})')
        model = ModelWindowTopK(
            args=args,
            device=device,
            cp_path=args.cp_path,
            use_sae=True,
            use_sparse_features=args.use_sparse_features,
            sae_dict_size=args.sae_dict_size,
            sae_k=args.sae_k,
            sae_window_size=args.sae_window_size,
            sae_weight=args.sae_weight
        )
    else:
        print('Using Per-Timestep TopK')
        model = ModelTopK(
            args=args,
            device=device,
            cp_path=args.cp_path,
            use_sae=True,
            use_sparse_features=args.use_sparse_features,
            sae_dict_size=args.sae_dict_size,
            sae_k=args.sae_k,
            sae_weight=args.sae_weight
        )
    model = nn.DataParallel(model).to(device)

    nb_params = sum([param.numel() for param in model.parameters()])
    print(f'Total parameters: {nb_params:,}')

    # Loss + Optim
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.NLLLoss(weight=weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Checkpoint loading
    start_epoch = 0
    best_val_eer = float('inf')
    def _get_state_dict(ckpt):
        if isinstance(ckpt, dict):
            for key in ('model_state_dict', 'state_dict', 'model'):
                if key in ckpt and isinstance(ckpt[key], dict):
                    return ckpt[key]
        if isinstance(ckpt, dict):
            # Fallback: checkpoint might itself be a state_dict
            if all(isinstance(k, str) for k in ckpt.keys()):
                return ckpt
        raise ValueError('Could not locate model state_dict in checkpoint')

    def _fix_module_prefix(state_dict, model_is_wrapped):
        """
        Fix module prefix to match model's DataParallel wrapping state.
        - If model is wrapped and state_dict lacks 'module.', add it.
        - If model is not wrapped and state_dict has 'module.', remove it.
        """
        if not state_dict:
            return state_dict
        
        has_module_prefix = all(isinstance(k, str) and k.startswith('module.') for k in state_dict.keys())
        
        if model_is_wrapped and not has_module_prefix:
            # Model wrapped, checkpoint not - add prefix
            return {'module.' + k: v for k, v in state_dict.items()}
        elif not model_is_wrapped and has_module_prefix:
            # Model not wrapped, checkpoint is - remove prefix
            return {k[len('module.'):]: v for k, v in state_dict.items()}
        
        return state_dict

    resume_path = _resolve_resume_path(model_save_path, args.model_path, args.resume)
    
    print(f'\n{"="*70}')
    if resume_path:
        print(f'LOADING CHECKPOINT')
        print(f'{"="*70}')
        print(f'Checkpoint: {os.path.basename(resume_path)}')
    else:
        print(f'FRESH START')
        print(f'{"="*70}')
        if args.resume:
            print('Resume requested but no checkpoint found')
    print(f'{"="*70}\n')

    if resume_path:
        print(f'Loading: {resume_path}')
        # Prefer safer weights-only load when possible.
        try:
            checkpoint = torch.load(resume_path, map_location=device, weights_only=True)
        except TypeError:
            checkpoint = torch.load(resume_path, map_location=device)
        except Exception:
            checkpoint = torch.load(resume_path, map_location=device)

        state_dict = _fix_module_prefix(_get_state_dict(checkpoint), isinstance(model, nn.DataParallel))
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            print('⚠️  Strict load failed; retrying with strict=False.')
            print(f'    Reason: {e}')
            model.load_state_dict(state_dict, strict=False)

        # Distinguish full training checkpoints vs weights-only state_dict.
        is_full_checkpoint = (
            isinstance(checkpoint, dict)
            and any(k in checkpoint for k in ('model_state_dict', 'optimizer_state_dict', 'epoch', 'best_val_eer'))
        )

        if is_full_checkpoint:
            best_val_eer = checkpoint.get('best_val_eer', best_val_eer)
            if args.resume and not args.fresh_start and 'optimizer_state_dict' in checkpoint and 'epoch' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = int(checkpoint['epoch']) + 1
                print(f'Resuming from epoch {start_epoch} (best EER {best_val_eer:.2f}%)')
            else:
                if args.resume and not args.fresh_start:
                    print('⚠️  Resume requested, but checkpoint does not contain optimizer/epoch; starting from epoch 0.')
                else:
                    print('Loaded weights (fresh optimizer/epoch)')
        else:
            # Weights-only checkpoint.
            if args.resume and not args.fresh_start:
                inferred_epoch = _infer_epoch_from_checkpoint_path(resume_path)
                if inferred_epoch is not None:
                    start_epoch = inferred_epoch + 1
                    print(
                        f'⚠️  Resuming from weights-only checkpoint; optimizer state not available. '
                        f'Continuing from epoch {start_epoch}.'
                    )
                    last_ckpt_path = os.path.join(model_save_path, 'last_checkpoint.pth')
                    if os.path.exists(last_ckpt_path):
                        print(f'    Tip: for an exact resume (optimizer+epoch), prefer {last_ckpt_path}')
                else:
                    print('⚠️  Resume requested, but could not infer epoch from filename; starting from epoch 0.')
            else:
                print('Loaded weights (fresh optimizer/epoch)')

    # Eval mode
    if args.is_eval:
        if not args.model_path:
            print('Error: --model_path is required for evaluation mode')
            return 1

        file_eval = genSpoof_list(
            dir_meta=args.protocols_path,
            is_train=False,
            is_eval=True
        )

        if args.track == 'In-the-Wild':
            eval_set = Dataset_in_the_wild_eval(list_IDs=file_eval, base_dir=args.database_path)
        else:
            eval_set = Dataset_ASVspoof2021_eval(list_IDs=file_eval, base_dir=args.database_path)

        eval_output_path = args.eval_output or os.path.join('scores', f'scores_{args.track}.txt')
        os.makedirs(os.path.dirname(eval_output_path), exist_ok=True)
        if os.path.exists(eval_output_path):
            os.remove(eval_output_path)

        produce_evaluation_file(eval_set, model, device, eval_output_path, quick_test=args.quick_test)
        print(f'Scores saved to: {eval_output_path}')
        return 0

    # Training mode
    log_path = os.path.join(model_save_path, 'training_log.csv')
    if start_epoch == 0 or not os.path.exists(log_path):
        init_csv_log(log_path)

    print('Loading datasets...')
    d_label_trn, file_train = genSpoof_list(
        dir_meta=os.path.join(
            args.protocols_path,
            'ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'
        ),
        is_train=True,
        is_eval=False
    )
    d_label_dev, file_dev = genSpoof_list(
        dir_meta=os.path.join(
            args.protocols_path,
            'ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'
        ),
        is_train=False,
        is_eval=False
    )
    print(f'   Training samples: {len(file_train)}')
    print(f'   Validation samples: {len(file_dev)}')

    train_set = Dataset_ASVspoof2019_train(
        args,
        list_IDs=file_train,
        labels=d_label_trn,
        base_dir=os.path.join(args.database_path, 'ASVspoof2019_LA_train/'),
        algo=getattr(args, 'algo', 3)
    )
    dev_set = Dataset_ASVspoof2019_train(
        args,
        list_IDs=file_dev,
        labels=d_label_dev,
        base_dir=os.path.join(args.database_path, 'ASVspoof2019_LA_dev/'),
        algo=getattr(args, 'algo', 3)
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=8
    )
    dev_loader = DataLoader(
        dev_set,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        num_workers=8
    )

    writer = SummaryWriter('logs/{}'.format(model_tag))

    for epoch in range(start_epoch, args.num_epochs):
        print(f'\n{"="*70}')
        print(f'Epoch {epoch+1}/{args.num_epochs} | Progress: {((epoch+1)/args.num_epochs)*100:.1f}%')
        print(f'{"="*70}')

        train_loss, train_eer, train_recon_loss = train_epoch(
            train_loader, model, optimizer, device, criterion, quick_test=args.quick_test
        )
        val_loss, val_acc, val_eer, val_recon_loss = evaluate_accuracy(
            dev_loader, model, device, criterion, quick_test=args.quick_test
        )

        print(f'\nTraining   - Loss: {train_loss:.6f}, EER: {train_eer:.2f}%, Recon Loss: {train_recon_loss:.6f}')
        print(f'Validation - Loss: {val_loss:.6f}, Acc: {val_acc:.2f}%, EER: {val_eer:.2f}%, Recon Loss: {val_recon_loss:.6f}')
        if best_val_eer < float('inf'):
            print(f'Best EER   - {best_val_eer:.2f}% (Δ: {val_eer - best_val_eer:+.2f}%)')

        log_data = {
            'epoch': epoch,
            'timestamp': datetime.now().strftime('%d/%m/%Y %H:%M'),
            'train_loss': train_loss,
            'train_eer': train_eer,
            'train_recon_loss': train_recon_loss,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_eer': val_eer,
            'val_recon_loss': val_recon_loss
        }
        log_training_metrics(log_data, model_save_path)

        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/eer', train_eer, epoch)
        writer.add_scalar('train/recon_loss', train_recon_loss, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/accuracy', val_acc, epoch)
        writer.add_scalar('val/eer', val_eer, epoch)
        writer.add_scalar('val/recon_loss', val_recon_loss, epoch)

        # Save checkpoints
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'train_eer': train_eer,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_eer': val_eer,
            'best_val_eer': best_val_eer,
            'args': vars(args),
        }
        
        # Save best model
        if val_eer < best_val_eer:
            best_val_eer = val_eer
            checkpoint_data['best_val_eer'] = best_val_eer
            best_name = f'best_checkpoint_eer{("_" + args.comment) if args.comment else ""}.pth'
            _atomic_torch_save(checkpoint_data, os.path.join(model_save_path, best_name))
            print(f'✓ New best model saved: EER {best_val_eer:.2f}%')
        
        # Always save resumable checkpoint (enables --resume)
        _atomic_torch_save(checkpoint_data, os.path.join(model_save_path, 'last_checkpoint.pth'))
        print(f'✓ Checkpoint saved (epoch {epoch+1})')

    writer.close()

    print('\nTraining completed!')
    print(f'Models saved to: {model_save_path}')
    print(f'Best validation EER: {best_val_eer:.2f}%')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
