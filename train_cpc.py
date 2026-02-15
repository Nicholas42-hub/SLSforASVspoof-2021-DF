#!/usr/bin/env python3
"""
Training script for Window TopK SAE with Contrastive Predictive Coding (CPC)
"""

import argparse
import sys
import os
import csv
from datetime import datetime
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from data_utils_SSL import (
    genSpoof_list,
    Dataset_ASVspoof2019_train,
)
from tensorboardX import SummaryWriter
from core_scripts.startup_config import set_random_seed
from tqdm import tqdm
from sklearn.metrics import roc_curve

# Import the CPC-enabled model
from model_cpc import Model


def compute_eer(scores, labels):
    """Compute Equal Error Rate (EER)"""
    try:
        scores = np.array(scores)
        labels = np.array(labels)
        
        if len(scores) == 0 or len(labels) == 0:
            return 50.0, 0.5
        
        valid_mask = ~(np.isnan(scores) | np.isnan(labels))
        if not np.any(valid_mask):
            return 50.0, 0.5
            
        scores = scores[valid_mask]
        labels = labels[valid_mask]
        
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            return 50.0, 0.5
        
        fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
        fnr = 1 - tpr
        
        abs_diff = np.abs(fpr - fnr)
        eer_idx = np.argmin(abs_diff)
        eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
        eer_threshold = thresholds[eer_idx] if eer_idx < len(thresholds) else 0.5
        
        return eer * 100, eer_threshold
        
    except Exception as e:
        print(f"Error computing EER: {e}")
        return 50.0, 0.5


def init_csv_log(log_path):
    """Initialize CSV log file with headers"""
    with open(log_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'epoch', 'timestamp', 
            'train_loss', 'train_cls_loss', 'train_sae_loss', 'train_cpc_loss', 'train_eer',
            'val_loss', 'val_cls_loss', 'val_sae_loss', 'val_cpc_loss', 'val_acc', 'val_eer'
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
            log_data['train_cls_loss'],
            log_data['train_sae_loss'],
            log_data['train_cpc_loss'],
            log_data['train_eer'],
            log_data['val_loss'],
            log_data['val_cls_loss'],
            log_data['val_sae_loss'],
            log_data['val_cpc_loss'],
            log_data['val_acc'],
            log_data['val_eer']
        ])


def evaluate_accuracy(dev_loader, model, device, criterion):
    """Evaluate model on validation set"""
    val_loss = 0.0
    val_cls_loss = 0.0
    val_sae_loss = 0.0
    val_cpc_loss = 0.0
    num_correct = 0.0
    num_total = 0.0
    
    all_scores = []
    all_labels = []
    
    model.eval()
    
    for batch_x, batch_y in tqdm(dev_loader, desc="Validation"):
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        
        with torch.no_grad():
            # Forward pass with all losses
            batch_out, sae_loss, cpc_loss = model(
                batch_x, 
                return_sae_loss=True, 
                return_cpc_loss=True
            )
            
            classification_loss = criterion(batch_out, batch_y)
            batch_loss = model.module.compute_total_loss(
                classification_loss, sae_loss, cpc_loss
            )
                
            _, batch_pred = batch_out.max(dim=1)
            num_correct += (batch_pred == batch_y).sum(dim=0).item()
            val_loss += (batch_loss.item() * batch_size)
            val_cls_loss += (classification_loss.item() * batch_size)
            
            if sae_loss is not None:
                val_sae_loss += (sae_loss.item() * batch_size)
            
            if cpc_loss is not None:
                val_cpc_loss += (cpc_loss.item() * batch_size)
            
            # Collect scores for EER
            batch_probs = torch.exp(batch_out)
            batch_scores = batch_probs[:, 1].cpu().numpy()
            batch_labels = batch_y.cpu().numpy()
            
            valid_mask = ~(np.isnan(batch_scores) | np.isinf(batch_scores))
            if np.any(valid_mask):
                all_scores.extend(batch_scores[valid_mask].tolist())
                all_labels.extend(batch_labels[valid_mask].tolist())

    val_loss /= num_total
    val_cls_loss /= num_total
    val_sae_loss /= num_total
    val_cpc_loss /= num_total
    acc = 100 * (num_correct / num_total)
    
    if len(all_scores) > 0 and len(all_labels) > 0:
        eer, _ = compute_eer(all_scores, all_labels)
    else:
        eer = 50.0
    
    return val_loss, val_cls_loss, val_sae_loss, val_cpc_loss, acc, eer


def train_epoch(train_loader, model, optim, device, criterion):
    """Train model for one epoch"""
    running_loss = 0.0
    running_cls_loss = 0.0
    running_sae_loss = 0.0
    running_cpc_loss = 0.0
    num_total = 0.0
    
    all_scores = []
    all_labels = []
    
    model.train()
    
    for batch_x, batch_y in tqdm(train_loader, desc="Training"):
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        
        # Forward pass with all losses
        batch_out, sae_loss, cpc_loss = model(
            batch_x, 
            return_sae_loss=True, 
            return_cpc_loss=True
        )
        
        classification_loss = criterion(batch_out, batch_y)
        batch_loss = model.module.compute_total_loss(
            classification_loss, sae_loss, cpc_loss
        )
        
        running_loss += (batch_loss.item() * batch_size)
        running_cls_loss += (classification_loss.item() * batch_size)
        
        if sae_loss is not None:
            running_sae_loss += (sae_loss.item() * batch_size)
        
        if cpc_loss is not None:
            running_cpc_loss += (cpc_loss.item() * batch_size)
        
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
    running_cls_loss /= num_total
    running_sae_loss /= num_total
    running_cpc_loss /= num_total
    
    if len(all_scores) > 0 and len(all_labels) > 0:
        train_eer, _ = compute_eer(all_scores, all_labels)
    else:
        train_eer = 50.0
    
    return running_loss, running_cls_loss, running_sae_loss, running_cpc_loss, train_eer


def main(args):
    """Main training function"""
    
    # Set random seed
    set_random_seed(args.seed)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model save directory
    model_tag = (
        f"cpc_window_w{args.sae_window_size}_"
        f"LA_e{args.num_epochs}_bs{args.batch_size}_"
        f"lr{args.lr}_saeW{args.sae_weight}_cpcW{args.cpc_weight}_"
        f"dict{args.sae_dict_size}_k{args.sae_k}_"
        f"{args.comment}"
    )
    model_save_path = os.path.join("models", model_tag)
    os.makedirs(model_save_path, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Model: {model_tag}")
    print(f"Save path: {model_save_path}")
    print(f"{'='*80}\n")
    
    # Initialize CSV log
    init_csv_log(os.path.join(model_save_path, 'training_log.csv'))
    
    # Setup TensorBoard
    writer = SummaryWriter(os.path.join(model_save_path, 'tensorboard'))
    
    # Load dataset
    print("Loading training data...")
    d_label_trn, file_train = genSpoof_list(
        dir_meta=os.path.join(args.database_path, 'ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'),
        is_train=True,
        is_eval=False
    )
    
    print("Loading validation data...")
    d_label_dev, file_dev = genSpoof_list(
        dir_meta=os.path.join(args.database_path, 'ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'),
        is_train=False,
        is_eval=False
    )
    
    # Create datasets
    train_set = Dataset_ASVspoof2019_train(
        args=args,
        list_IDs=file_train,
        labels=d_label_trn,
        base_dir=os.path.join(args.database_path, 'ASVspoof2019_LA_train/'),
        algo=args.algo
    )
    
    dev_set = Dataset_ASVspoof2019_train(
        args=args,
        list_IDs=file_dev,
        labels=d_label_dev,
        base_dir=os.path.join(args.database_path, 'ASVspoof2019_LA_dev/'),
        algo=0  # No augmentation for validation
    )
    
    # Create data loaders
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
        drop_last=False,
        pin_memory=True,
        num_workers=8
    )
    
    print(f"Training samples: {len(train_set)}")
    print(f"Validation samples: {len(dev_set)}")
    
    # Initialize model with CPC
    print("\nInitializing model with CPC...")
    model = Model(
        args=args,
        device=device,
        cp_path=args.cp_path,
        use_sae=True,
        use_sparse_features=True,
        sae_dict_size=args.sae_dict_size,
        sae_k=args.sae_k,
        sae_window_size=args.sae_window_size,
        sae_weight=args.sae_weight,
        # CPC parameters
        use_cpc=True,
        cpc_hidden_dim=args.cpc_hidden_dim,
        cpc_weight=args.cpc_weight,
        cpc_temperature=args.cpc_temperature,
        cpc_prediction_steps=args.cpc_prediction_steps,
    )
    
    model = model.to(device)
    model = nn.DataParallel(model)
    
    print(f"\nModel architecture:")
    print(f"  SAE: dict_size={args.sae_dict_size}, k={args.sae_k}, window_size={args.sae_window_size}")
    print(f"  CPC: hidden_dim={args.cpc_hidden_dim}, weight={args.cpc_weight}, temp={args.cpc_temperature}")
    print(f"  CPC prediction steps: {args.cpc_prediction_steps}")
    
    # Optimizer
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.NLLLoss()
    
    # Training loop
    best_eer = 100.0
    start_epoch = 0
    
    # Resume from checkpoint if exists
    checkpoint_path = os.path.join(model_save_path, 'last_checkpoint.pth')
    if args.resume and os.path.exists(checkpoint_path):
        print(f"\nResuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_eer = checkpoint.get('best_eer', 100.0)
        print(f"Resuming from epoch {start_epoch}, best EER: {best_eer:.2f}%")
    
    print(f"\n{'='*80}")
    print(f"Starting training from epoch {start_epoch}")
    print(f"{'='*80}\n")
    
    for epoch in range(start_epoch, args.num_epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{args.num_epochs-1}")
        print(f"{'='*80}")
        
        # Train
        train_loss, train_cls_loss, train_sae_loss, train_cpc_loss, train_eer = train_epoch(
            train_loader, model, optim, device, criterion
        )
        
        # Validate
        val_loss, val_cls_loss, val_sae_loss, val_cpc_loss, val_acc, val_eer = evaluate_accuracy(
            dev_loader, model, device, criterion
        )
        
        # Print epoch summary
        print(f"\n📊 Epoch {epoch} Summary:")
        print(f"  Train - Loss: {train_loss:.4f} (cls: {train_cls_loss:.4f}, sae: {train_sae_loss:.4f}, cpc: {train_cpc_loss:.4f}), EER: {train_eer:.2f}%")
        print(f"  Val   - Loss: {val_loss:.4f} (cls: {val_cls_loss:.4f}, sae: {val_sae_loss:.4f}, cpc: {val_cpc_loss:.4f}), Acc: {val_acc:.2f}%, EER: {val_eer:.2f}%")
        
        # TensorBoard logging
        writer.add_scalar('train/total_loss', train_loss, epoch)
        writer.add_scalar('train/cls_loss', train_cls_loss, epoch)
        writer.add_scalar('train/sae_loss', train_sae_loss, epoch)
        writer.add_scalar('train/cpc_loss', train_cpc_loss, epoch)
        writer.add_scalar('train/eer', train_eer, epoch)
        writer.add_scalar('val/total_loss', val_loss, epoch)
        writer.add_scalar('val/cls_loss', val_cls_loss, epoch)
        writer.add_scalar('val/sae_loss', val_sae_loss, epoch)
        writer.add_scalar('val/cpc_loss', val_cpc_loss, epoch)
        writer.add_scalar('val/accuracy', val_acc, epoch)
        writer.add_scalar('val/eer', val_eer, epoch)
        
        # CSV logging
        log_data = {
            'epoch': epoch,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'train_loss': train_loss,
            'train_cls_loss': train_cls_loss,
            'train_sae_loss': train_sae_loss,
            'train_cpc_loss': train_cpc_loss,
            'train_eer': train_eer,
            'val_loss': val_loss,
            'val_cls_loss': val_cls_loss,
            'val_sae_loss': val_sae_loss,
            'val_cpc_loss': val_cpc_loss,
            'val_acc': val_acc,
            'val_eer': val_eer
        }
        log_training_metrics(log_data, model_save_path)
        
        # Save checkpoints
        # Save last checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'best_eer': best_eer,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_eer': val_eer,
        }, checkpoint_path)
        
        # Save best model
        if val_eer < best_eer:
            best_eer = val_eer
            best_path = os.path.join(model_save_path, f'best_checkpoint_eer_{args.comment}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'best_eer': best_eer,
                'val_eer': val_eer,
            }, best_path)
            print(f"  ✅ New best EER: {val_eer:.2f}% (saved to {best_path})")
    
    writer.close()
    print(f"\n{'='*80}")
    print(f"Training completed!")
    print(f"Best validation EER: {best_eer:.2f}%")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ASVspoof2019 CPC Training')
    
    # Dataset
    parser.add_argument('--database_path', type=str, required=True)
    parser.add_argument('--protocols_path', type=str, default=None)
    parser.add_argument('--track', type=str, default='LA', choices=['LA', 'PA', 'DF'])
    
    # Training
    parser.add_argument('--batch_size', type=int, default=14)
    parser.add_argument('--num_epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--algo', type=int, default=5, help='Data augmentation algorithm')
    
    # Model
    parser.add_argument('--cp_path', type=str, default='xlsr2_300m.pt')
    
    # SAE parameters
    parser.add_argument('--sae_dict_size', type=int, default=4096)
    parser.add_argument('--sae_k', type=int, default=128)
    parser.add_argument('--sae_window_size', type=int, default=8)
    parser.add_argument('--sae_weight', type=float, default=0.1)
    
    # CPC parameters
    parser.add_argument('--cpc_hidden_dim', type=int, default=256)
    parser.add_argument('--cpc_weight', type=float, default=0.5)
    parser.add_argument('--cpc_temperature', type=float, default=0.07)
    parser.add_argument('--cpc_prediction_steps', type=int, nargs='+', default=[1, 2, 4])
    
    # Augmentation parameters (from data_utils_SSL.py)
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
    
    parser.add_argument('--comment', type=str, default='cpc')
    
    args = parser.parse_args()
    
    main(args)
