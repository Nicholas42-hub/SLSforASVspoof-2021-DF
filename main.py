import argparse
import sys
import os
import csv
from datetime import datetime
import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
import yaml
from data_utils_SSL import genSpoof_list,Dataset_ASVspoof2019_train,Dataset_ASVspoof2021_eval,Dataset_in_the_wild_eval
from model_pyramid_grouping import Model
from model_swin_layer_attention import Swin_Model
from model_AdaptiveGroupingHierarchicalModel import AdaptiveGroupingHierarchicalModel
from model_hierachical_transformer import ModelTransformerHierarchical
from model_g3_heatmap import Model as ModelG3  # 导入 G3 模型
# 移除 hier_con 导入
# from model_hier_con import ModelHierarchicalContrastive, CombinedLossWithFocal, CombinedLossFlexible
from tensorboardX import SummaryWriter
from core_scripts.startup_config import set_random_seed
from tqdm import tqdm
from torchvision import transforms
from sklearn.metrics import roc_curve

# 添加条件导入可视化工具
try:
    from visualize_attention_heatmap import analyze_attention_patterns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    print("⚠️  Warning: visualize_attention_heatmap not found. Attention visualization will be disabled.")
    VISUALIZATION_AVAILABLE = False
    analyze_attention_patterns = None

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
            'epoch', 'timestamp', 'train_loss', 'train_eer',
            'val_loss', 'val_acc', 'val_eer'
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
            log_data['val_loss'],
            log_data['val_acc'],
            log_data['val_eer']
        ])

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
        print(f'📁 Directory does not exist: {model_save_path}')
        return
    
    checkpoint_files = [f for f in os.listdir(model_save_path) 
                       if f.endswith('.pth')]
    
    if not checkpoint_files:
        print(f'📁 No checkpoints found in: {model_save_path}')
        return
    
    print(f'\n{"="*70}')
    print(f'📋 Available Checkpoints in {model_save_path}')
    print(f'{"="*70}')
    
    for f in sorted(checkpoint_files):
        checkpoint_path = os.path.join(model_save_path, f)
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            if isinstance(checkpoint, dict) and 'epoch' in checkpoint:
                print(f'\n📦 {f}')
                print(f'   Epoch: {checkpoint["epoch"]}')
                if 'val_eer' in checkpoint:
                    print(f'   Val EER: {checkpoint["val_eer"]:.2f}%')
                if 'val_acc' in checkpoint:
                    print(f'   Val Acc: {checkpoint["val_acc"]:.2f}%')
                if 'best_val_eer' in checkpoint:
                    print(f'   Best EER: {checkpoint["best_val_eer"]:.2f}%')
            else:
                print(f'\n📦 {f} (legacy format - weights only)')
        except Exception as e:
            print(f'\n⚠️  {f} - Error loading: {e}')
    
    print(f'{"="*70}\n')

def evaluate_accuracy(dev_loader, model, device, criterion, use_contrastive=False):
    val_loss = 0.0
    num_correct = 0.0
    num_total = 0.0
    
    # For EER calculation
    all_scores = []
    all_labels = []
    
    model.eval()
    
    for batch_x, batch_y in tqdm(dev_loader):
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        
        with torch.no_grad():
            if use_contrastive:
                # 对比学习模型返回 (logits, features, contrastive_loss, supcon_loss)
                batch_out, features, _, _ = model(batch_x, batch_y)
            else:
                # G3 模型返回 (logits, features) 元组
                model_output = model(batch_x)
                
                # 检查输出类型并正确解包
                if isinstance(model_output, tuple):
                    batch_out = model_output[0]  # 只取 logits
                else:
                    batch_out = model_output
                
            _, batch_pred = batch_out.max(dim=1)
            num_correct += (batch_pred == batch_y).sum(dim=0).item()
            
            # 计算损失
            if use_contrastive:
                # 使用对比学习损失
                batch_loss = criterion(batch_out, features, batch_y)[0]  # total_loss
            else:
                batch_loss = criterion(batch_out, batch_y)
                
            val_loss += (batch_loss.item() * batch_size)
            
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
    
    # Compute EER
    if len(all_scores) > 0 and len(all_labels) > 0:
        eer, eer_threshold = compute_eer(all_scores, all_labels)
    else:
        eer, eer_threshold = 50.0, 0.5
        print("Warning: No valid scores for EER calculation, using default EER=50%")
    
    return val_loss, acc, eer

def produce_evaluation_file(dataset, model, device, save_path, use_contrastive=False):
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
    
    for batch_x, utt_id in tqdm(data_loader):
        batch_size = batch_x.size(0)
        batch_x = batch_x.to(device)
        
        with torch.no_grad():
            if use_contrastive:
                batch_out, _, _, _ = model(batch_x, None)
            else:
                # G3 模型返回 (logits, features) 元组
                model_output = model(batch_x)
                
                # 检查输出类型并正确解包
                if isinstance(model_output, tuple):
                    batch_out = model_output[0]  # 只取 logits
                else:
                    batch_out = model_output
                
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

def train_epoch(train_loader, model, lr, optim, device, criterion, use_contrastive=False):
    running_loss = 0
    num_total = 0.0
    
    # For training EER calculation
    all_scores = []
    all_labels = []
    
    model.train()
    
    for batch_x, batch_y in tqdm(train_loader):
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        
        # Forward pass
        if use_contrastive:
            # 对比学习模型
            batch_out, features, contrastive_loss, supcon_loss = model(batch_x, batch_y)
            # 计算总损失
            total_loss, primary_loss, combined_contrastive = criterion(batch_out, features, batch_y)
            batch_loss = total_loss
        else:
            # G3 模型返回 (logits, features) 元组
            model_output = model(batch_x)
            
            # 检查输出类型并正确解包
            if isinstance(model_output, tuple):
                batch_out = model_output[0]  # 只取 logits
            else:
                batch_out = model_output
                
            batch_loss = criterion(batch_out, batch_y)
        
        running_loss += (batch_loss.item() * batch_size)
        
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
    
    # Compute training EER
    if len(all_scores) > 0 and len(all_labels) > 0:
        train_eer, _ = compute_eer(all_scores, all_labels)
    else:
        train_eer = 50.0
    
    return running_loss, train_eer

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
        print(f'📁 Directory does not exist: {model_save_path}')
        return
    
    checkpoint_files = [f for f in os.listdir(model_save_path) 
                       if f.endswith('.pth')]
    
    if not checkpoint_files:
        print(f'📁 No checkpoints found in: {model_save_path}')
        return
    
    print(f'\n{"="*70}')
    print(f'📋 Available Checkpoints in {model_save_path}')
    print(f'{"="*70}')
    
    for f in sorted(checkpoint_files):
        checkpoint_path = os.path.join(model_save_path, f)
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            if isinstance(checkpoint, dict) and 'epoch' in checkpoint:
                print(f'\n📦 {f}')
                print(f'   Epoch: {checkpoint["epoch"]}')
                if 'val_eer' in checkpoint:
                    print(f'   Val EER: {checkpoint["val_eer"]:.2f}%')
                if 'val_acc' in checkpoint:
                    print(f'   Val Acc: {checkpoint["val_acc"]:.2f}%')
                if 'best_val_eer' in checkpoint:
                    print(f'   Best EER: {checkpoint["best_val_eer"]:.2f}%')
            else:
                print(f'\n📦 {f} (legacy format - weights only)')
        except Exception as e:
            print(f'\n⚠️  {f} - Error loading: {e}')
    
    print(f'{"="*70}\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ASVspoof2021 Hierarchical Attention System with Heatmap Visualization')
    
    # Dataset
    parser.add_argument('--database_path', type=str, default='/root/autodl-tmp/CLAD/Datasets/LA/')
    parser.add_argument('--protocols_path', type=str, default='/root/autodl-tmp/CLAD/Datasets/LA/')
    
    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=14)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.000001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--loss', type=str, default='CCE')
    
    # Group parameters
    parser.add_argument('--group_size', type=int, default=3)
    
    # Model selection - 移除 hier_con
    parser.add_argument('--model_type', type=str, default='g3_heatmap', 
                        choices=['pyramid', 'fusion', 'hierarchical', 'swin', 
                                'AdaptiveGroupingHierarchicalModel', 'ModelTransformerHierarchical',
                                'g3_heatmap'],
                        help='Model type')
    
    # ========== G3 Model Parameters ==========
    parser.add_argument('--use_contrastive', action='store_true', default=False,
                        help='Use contrastive learning (for G3 model)')
    parser.add_argument('--contrastive_weight', type=float, default=0.1,
                        help='Weight for contrastive loss')
    parser.add_argument('--supcon_weight', type=float, default=0.1,
                        help='Weight for supervised contrastive loss')
    
    # ========== Attention Visualization Parameters ==========
    parser.add_argument('--visualize_attention', action='store_true', default=False,
                        help='Generate attention heatmaps during/after training')
    parser.add_argument('--viz_samples', type=int, default=20,
                        help='Number of samples for attention visualization')
    parser.add_argument('--viz_frequency', type=int, default=10,
                        help='Visualize every N epochs (0=only at end)')
    
    # Model
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--comment', type=str, default=None)
    
    # Checkpoint management
    parser.add_argument('--resume', action='store_true', default=False,
                        help='Resume training from checkpoint (auto-finds latest if --model_path not provided)')
    parser.add_argument('--fresh_start', action='store_true', default=False,
                        help='Start fresh training, ignore existing checkpoints')
    parser.add_argument('--auto_resume', action='store_true', default=False,
                        help='Automatically resume from latest checkpoint if available')
    parser.add_argument('--list_checkpoints', action='store_true', default=False,
                        help='List all available checkpoints and exit')
    
    # Auxiliary arguments
    parser.add_argument('--track', type=str, default='DF', choices=['LA', 'In-the-Wild', 'DF'])
    parser.add_argument('--eval_output', type=str, default=None)
    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--is_eval', action='store_true', default=False)
    parser.add_argument('--eval_part', type=int, default=0)
    
    # Backend options
    parser.add_argument('--cudnn-deterministic-toggle', action='store_false', default=True)
    parser.add_argument('--cudnn-benchmark-toggle', action='store_true', default=False)

    # Rawboost parameters (keeping all existing ones)
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
    
    # Other model parameters (keeping existing ones)
    parser.add_argument('--processing_mode', type=str, default='full_swin_2d')
    parser.add_argument('--use_multiscale', action='store_true', default=True)
    parser.add_argument('--num_transformer_layers', type=int, default=4)
    parser.add_argument('--im_feedforward', type=int, default=4)
    
    if not os.path.exists('models'):
        os.mkdir('models')
    args = parser.parse_args()
 
    # ========== Validate checkpoint arguments ==========
    if args.resume and args.fresh_start:
        parser.error('❌ Cannot use both --resume and --fresh_start. Choose one.')
    
    if args.fresh_start and args.model_path:
        print('⚠️  Warning: --fresh_start specified but --model_path provided.')
        print('   Model weights will be loaded but training will start from epoch 0.')
        print('   Optimizer state will NOT be restored.')
    
    # Handle list checkpoints request
    if args.list_checkpoints:
        # Build model tag first
        track = args.track
        if args.model_type == 'g3_heatmap':
            model_tag = 'g3_heatmap_{}_{}_{}_{}_{}_group{}_contrastive{}'.format(
                track, args.loss, args.num_epochs, args.batch_size, args.lr,
                args.group_size, args.use_contrastive)
        elif args.model_type == 'fusion':
            model_tag = 'fusion_{}_{}_{}_{}_{}'.format(
                track, args.loss, args.num_epochs, args.batch_size, args.lr)
        elif args.model_type == 'hierarchical':
            model_tag = 'hierarchical_{}_{}_{}_{}_{}'.format(
                track, args.loss, args.num_epochs, args.batch_size, args.lr)
        elif args.model_type == 'swin':
            model_tag = 'swin_{}_{}_{}_{}_{}_processing_{}'.format(
                track, args.loss, args.num_epochs, args.batch_size, args.lr,
                args.processing_mode)
        elif args.model_type == 'AdaptiveGroupingHierarchicalModel':
            model_tag = 'AdaptiveGroupingHierarchicalModel_{}_{}_{}_{}_{}'.format(
                track, args.loss, args.num_epochs, args.batch_size, args.lr)
        elif args.model_type == 'ModelTransformerHierarchical':
            model_tag = 'ModelTransformerHierarchical_{}_{}_{}_{}_{}'.format(
                track, args.loss, args.num_epochs, args.batch_size, args.lr)
        else:  # pyramid
            model_tag = 'pyramid_{}_{}_{}_{}_{}'.format(
                track, args.loss, args.num_epochs, args.batch_size, args.lr)
        
        if args.comment:
            model_tag = model_tag + '_{}'.format(args.comment)
        model_save_path = os.path.join('models', model_tag)
        
        list_available_checkpoints(model_save_path)
        sys.exit(0)
    
    # Handle auto-resume or resume without model_path
    if (args.auto_resume or args.resume) and not args.model_path and not args.fresh_start:
        # Build model tag and path (same logic as above)
        track = args.track
        if args.model_type == 'g3_heatmap':
            model_tag = 'g3_heatmap_{}_{}_{}_{}_{}_group{}_contrastive{}'.format(
                track, args.loss, args.num_epochs, args.batch_size, args.lr,
                args.group_size, args.use_contrastive)
        elif args.model_type == 'fusion':
            model_tag = 'fusion_{}_{}_{}_{}_{}'.format(
                track, args.loss, args.num_epochs, args.batch_size, args.lr)
        elif args.model_type == 'hierarchical':
            model_tag = 'hierarchical_{}_{}_{}_{}_{}'.format(
                track, args.loss, args.num_epochs, args.batch_size, args.lr)
        elif args.model_type == 'swin':
            model_tag = 'swin_{}_{}_{}_{}_{}_processing_{}'.format(
                track, args.loss, args.num_epochs, args.batch_size, args.lr,
                args.processing_mode)
        elif args.model_type == 'AdaptiveGroupingHierarchicalModel':
            model_tag = 'AdaptiveGroupingHierarchicalModel_{}_{}_{}_{}_{}'.format(
                track, args.loss, args.num_epochs, args.batch_size, args.lr)
        elif args.model_type == 'ModelTransformerHierarchical':
            model_tag = 'ModelTransformerHierarchical_{}_{}_{}_{}_{}'.format(
                track, args.loss, args.num_epochs, args.batch_size, args.lr)
        else:  # pyramid
            model_tag = 'pyramid_{}_{}_{}_{}_{}'.format(
                track, args.loss, args.num_epochs, args.batch_size, args.lr)
        
        if args.comment:
            model_tag = model_tag + '_{}'.format(args.comment)
        model_save_path_temp = os.path.join('models', model_tag)
        
        latest_checkpoint, latest_epoch = find_latest_checkpoint(model_save_path_temp)
        if latest_checkpoint:
            if args.resume:
                print(f'🔍 --resume flag: Auto-finding latest checkpoint at epoch {latest_epoch}')
            else:
                print(f'🔍 Auto-resume enabled: Found checkpoint at epoch {latest_epoch}')
            args.model_path = latest_checkpoint
            args.resume = True  # Ensure resume flag is set
        else:
            if args.resume:
                print(f'⚠️  --resume flag specified but no checkpoint found in {model_save_path_temp}')
                print(f'   Starting fresh training instead...')
                args.resume = False  # Can't resume without checkpoint
            else:
                print(f'🔍 Auto-resume enabled: No checkpoint found, starting fresh')
    
    # Make experiment reproducible
    set_random_seed(args.seed, args)
    
    track = args.track

    # Define model saving path
    if args.model_type == 'g3_heatmap':
        model_tag = 'g3_heatmap_{}_{}_{}_{}_{}_group{}_contrastive{}'.format(
            track, args.loss, args.num_epochs, args.batch_size, args.lr,
            args.group_size, args.use_contrastive)
    elif args.model_type == 'fusion':
        model_tag = 'fusion_{}_{}_{}_{}_{}'.format(
            track, args.loss, args.num_epochs, args.batch_size, args.lr)
    elif args.model_type == 'hierarchical':
        model_tag = 'hierarchical_{}_{}_{}_{}_{}'.format(
            track, args.loss, args.num_epochs, args.batch_size, args.lr)
    elif args.model_type == 'swin':
        model_tag = 'swin_{}_{}_{}_{}_{}_processing_{}'.format(
            track, args.loss, args.num_epochs, args.batch_size, args.lr,
            args.processing_mode)
    elif args.model_type == 'AdaptiveGroupingHierarchicalModel':
        model_tag = 'AdaptiveGroupingHierarchicalModel_{}_{}_{}_{}_{}'.format(
            track, args.loss, args.num_epochs, args.batch_size, args.lr)
    elif args.model_type == 'ModelTransformerHierarchical':
        model_tag = 'ModelTransformerHierarchical_{}_{}_{}_{}_{}'.format(
            track, args.loss, args.num_epochs, args.batch_size, args.lr)
    else:  # pyramid
        model_tag = 'pyramid_{}_{}_{}_{}_{}'.format(
            track, args.loss, args.num_epochs, args.batch_size, args.lr)

    if args.comment:
        model_tag = model_tag + '_{}'.format(args.comment)
    model_save_path = os.path.join('models', model_tag)

    # Set model save directory
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
    
    # Initialize CSV log
    init_csv_log(os.path.join(model_save_path, 'training_log.csv'))
    
    # Print configuration
    print('=' * 70)
    print('🚀 CONFIGURATION')
    print('=' * 70)
    print(f'Model Type: {args.model_type}')
    print(f'Group Size: {args.group_size}')
    if args.model_type == 'g3_heatmap':
        print(f'Use Contrastive: {args.use_contrastive}')
        if args.use_contrastive:
            print(f'  Contrastive Weight: {args.contrastive_weight}')
            print(f'  SupCon Weight: {args.supcon_weight}')
    print(f'Attention Visualization: {args.visualize_attention and VISUALIZATION_AVAILABLE}')
    if args.visualize_attention:
        if VISUALIZATION_AVAILABLE:
            print(f'  Viz Samples: {args.viz_samples}')
            print(f'  Viz Frequency: every {args.viz_frequency} epochs')
        else:
            print('  ⚠️  Visualization module not available')
    print('=' * 70)
    
    # GPU device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'                  
    print(f'Device: {device}')

    # Initialize model based on type
    if args.model_type == 'g3_heatmap':
        model = ModelG3(args, device)
        print('✅ Using G3 Heatmap Model with Contrastive Learning')
        use_contrastive_training = args.use_contrastive
    elif args.model_type == 'fusion':
        model = FusionBestModel(args, device)
        print('✅ Using Fusion Best Model')
        use_contrastive_training = False
    elif args.model_type == 'hierarchical':
        model = HierarchicalLayerModel(args, device)
        print('✅ Using Hierarchical Layer Attention Model')
        use_contrastive_training = False
    elif args.model_type == 'swin':
        model = Swin_Model(args, device)
        print('✅ Using Swin Attention Model')
        use_contrastive_training = False
    elif args.model_type == 'AdaptiveGroupingHierarchicalModel':
        model = AdaptiveGroupingHierarchicalModel(args, device)
        print('✅ Using AdaptiveGroupingHierarchicalModel')
        use_contrastive_training = False
    elif args.model_type == 'ModelTransformerHierarchical':
        model = ModelTransformerHierarchical(args, device)
        print('✅ Using ModelTransformerHierarchical')
        use_contrastive_training = False
    else:  # pyramid
        model = Model(args, device)  # PyramidModel
        print('✅ Using Pyramid Model')
        use_contrastive_training = False
    
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print(f'📊 Total parameters: {nb_params:,}')

    model = nn.DataParallel(model).to(device)

    # Set loss function - 简化版本
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.NLLLoss(weight=weight)
    print('📉 Using Negative Log-Likelihood Loss')

    # Set Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # ========== Enhanced Checkpoint Loading ==========
    start_epoch = 0
    best_val_eer = float('inf')
    is_resuming = False  # Track if we're resuming from checkpoint
    
    if args.model_path:
        print(f'\n{"="*70}')
        print('📦 Loading checkpoint...')
        print(f'{"="*70}')
        
        # Check if checkpoint file exists
        if not os.path.exists(args.model_path):
            print(f'❌ Checkpoint file not found: {args.model_path}')
            print(f'   Starting fresh training...')
            print(f'{"="*70}\n')
        else:
            try:
                checkpoint = torch.load(args.model_path, map_location=device)
                
                # Determine if we should resume training state or just load weights
                should_resume_training_state = args.resume and not args.fresh_start
                
                # Check if it's a full checkpoint or just model weights
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    # Load model weights
                    model.load_state_dict(checkpoint['model_state_dict'])
                    
                    if should_resume_training_state:
                        # Full resume: restore optimizer, epoch, and metrics
                        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        start_epoch = checkpoint['epoch'] + 1  # Resume from next epoch
                        best_val_eer = checkpoint.get('best_val_eer', float('inf'))
                        is_resuming = True
                        
                        print(f'✅ RESUMING TRAINING from checkpoint')
                        print(f'   📦 Loaded model weights + optimizer state')
                        print(f'   📍 Checkpoint from epoch: {checkpoint["epoch"]}')
                        print(f'   🔄 Will resume from epoch: {start_epoch}')
                        print(f'   📊 Best validation EER so far: {best_val_eer:.2f}%')
                        
                        # Display training history
                        if 'train_loss' in checkpoint:
                            print(f'   📉 Last train loss: {checkpoint["train_loss"]:.6f}')
                        if 'train_eer' in checkpoint:
                            print(f'   📈 Last train EER: {checkpoint["train_eer"]:.2f}%')
                        if 'val_eer' in checkpoint:
                            print(f'   📈 Last val EER: {checkpoint["val_eer"]:.2f}%')
                        if 'val_acc' in checkpoint:
                            print(f'   🎯 Last val accuracy: {checkpoint["val_acc"]:.2f}%')
                        
                        # Display saved hyperparameters
                        if 'args' in checkpoint:
                            saved_args = checkpoint['args']
                            print(f'\n   📋 Checkpoint hyperparameters:')
                            print(f'      Learning rate: {saved_args.get("lr", "N/A")}')
                            print(f'      Batch size: {saved_args.get("batch_size", "N/A")}')
                            print(f'      Group size: {saved_args.get("group_size", "N/A")}')
                            
                            # Warn if hyperparameters changed
                            if saved_args.get('lr') != args.lr:
                                print(f'   ⚠️  WARNING: Learning rate changed from {saved_args.get("lr")} to {args.lr}')
                            if saved_args.get('batch_size') != args.batch_size:
                                print(f'   ⚠️  WARNING: Batch size changed from {saved_args.get("batch_size")} to {args.batch_size}')
                    else:
                        # Fresh start with pre-trained weights
                        print(f'✅ FRESH START with pre-trained weights')
                        print(f'   📦 Loaded model weights only')
                        print(f'   🆕 Starting from epoch 0')
                        print(f'   ⚠️  Optimizer state NOT restored (fresh optimizer)')
                        is_resuming = False
                        
                else:
                    # Legacy checkpoint (only model weights)
                    model.load_state_dict(checkpoint)
                    print(f'✅ Loaded model weights (legacy format)')
                    print(f'   🆕 Starting from epoch 0')
                    print(f'   ℹ️  This is a fresh start with pre-trained weights')
                    is_resuming = False
                    
            except Exception as e:
                print(f'❌ Error loading checkpoint: {e}')
                print(f'   Starting training from scratch...')
                import traceback
                traceback.print_exc()
                start_epoch = 0
                best_val_eer = float('inf')
                is_resuming = False
            
            print(f'{"="*70}\n')
    elif not args.fresh_start:
        # No checkpoint specified and not explicitly fresh start - check for existing checkpoints
        print(f'\n{"="*70}')
        print('🔍 Checking for existing checkpoints...')
        print(f'{"="*70}')
        
        # Look for latest checkpoint in model save path
        if os.path.exists(model_save_path):
            # Support both checkpoint naming formats
            all_checkpoint_files = [f for f in os.listdir(model_save_path) if f.endswith('.pth')]
            checkpoint_files = [f for f in all_checkpoint_files 
                              if f.startswith('checkpoint_epoch_') or f.startswith('epoch_')]
            
            if checkpoint_files:
                # Find the latest checkpoint using our helper function
                latest_checkpoint_path, latest_epoch = find_latest_checkpoint(model_save_path)
                
                if latest_checkpoint_path:
                    latest_file = os.path.basename(latest_checkpoint_path)
                    print(f'📦 Found existing checkpoint: {latest_file}')
                    print(f'   Epoch: {latest_epoch}')
                    print(f'\n   💡 To resume from this checkpoint, use:')
                    print(f'      --resume')
                    print(f'\n   💡 Or specify a specific checkpoint with:')
                    print(f'      --resume --model_path {latest_checkpoint_path}')
                    print(f'\n   💡 To start fresh and ignore existing checkpoints, use:')
                    print(f'      --fresh_start')
                    print(f'\n   🆕 Starting fresh training (no --resume flag)...')
                else:
                    print(f'✨ No valid checkpoints found - starting fresh training')
            else:
                print(f'✨ No checkpoints found - starting fresh training')
        else:
            print(f'✨ Model directory does not exist yet - starting fresh training')
        
        print(f'{"="*70}\n')
    else:
        # Explicit fresh start
        print(f'\n{"="*70}')
        print(f'✨ FRESH START requested (--fresh_start flag)')
        print(f'{"="*70}\n')

    # Print training mode summary
    print(f'\n{"="*70}')
    if is_resuming:
        print(f'🔄 RESUMING TRAINING FROM CHECKPOINT')
        print(f'{"="*70}')
        print(f'📍 Starting epoch: {start_epoch + 1}')
        print(f'🎯 Target epoch: {args.num_epochs}')
        print(f'📊 Epochs remaining: {args.num_epochs - start_epoch}')
        print(f'🏆 Best EER to beat: {best_val_eer:.2f}%')
    else:
        print(f'✨ FRESH TRAINING START')
        print(f'{"="*70}')
        print(f'📍 Starting from epoch: 1')
        print(f'🎯 Total epochs: {args.num_epochs}')
    print(f'{"="*70}\n')
    
    # ========== Enhanced Checkpoint Loading ==========
    start_epoch = 0
    best_val_eer = float('inf')
    is_resuming = False  # Track if we're resuming from checkpoint
    
    if args.model_path:
        print(f'\n{"="*70}')
        print('📦 Loading checkpoint...')
        print(f'{"="*70}')
        
        # Check if checkpoint file exists
        if not os.path.exists(args.model_path):
            print(f'❌ Checkpoint file not found: {args.model_path}')
            print(f'   Starting fresh training...')
            print(f'{"="*70}\n')
        else:
            try:
                checkpoint = torch.load(args.model_path, map_location=device)
                
                # Check if it's a full checkpoint or just model weights
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    # Full checkpoint with training state
                    model.load_state_dict(checkpoint['model_state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    start_epoch = checkpoint['epoch'] + 1  # Resume from next epoch
                    best_val_eer = checkpoint.get('best_val_eer', float('inf'))
                    is_resuming = True
                    
                    print(f'✅ Loaded full checkpoint from epoch {checkpoint["epoch"]}')
                    print(f'   📊 Best validation EER so far: {best_val_eer:.2f}%')
                    print(f'   🔄 Resuming training from epoch {start_epoch}')
                    print(f'   📍 Will train from epoch {start_epoch} to {args.num_epochs}')
                    
                    # Load and display training history if available
                    if 'train_loss' in checkpoint:
                        print(f'   📉 Last train loss: {checkpoint["train_loss"]:.6f}')
                    if 'train_eer' in checkpoint:
                        print(f'   📈 Last train EER: {checkpoint["train_eer"]:.2f}%')
                    if 'val_eer' in checkpoint:
                        print(f'   📈 Last val EER: {checkpoint["val_eer"]:.2f}%')
                    if 'val_acc' in checkpoint:
                        print(f'   🎯 Last val accuracy: {checkpoint["val_acc"]:.2f}%')
                    
                    # Display saved hyperparameters if available
                    if 'args' in checkpoint:
                        saved_args = checkpoint['args']
                        print(f'\n   📋 Checkpoint hyperparameters:')
                        print(f'      Learning rate: {saved_args.get("lr", "N/A")}')
                        print(f'      Batch size: {saved_args.get("batch_size", "N/A")}')
                        print(f'      Group size: {saved_args.get("group_size", "N/A")}')
                        
                        # Warn if hyperparameters changed
                        if saved_args.get('lr') != args.lr:
                            print(f'   ⚠️  WARNING: Learning rate changed from {saved_args.get("lr")} to {args.lr}')
                        if saved_args.get('batch_size') != args.batch_size:
                            print(f'   ⚠️  WARNING: Batch size changed from {saved_args.get("batch_size")} to {args.batch_size}')
                        
                else:
                    # Legacy checkpoint (only model weights)
                    model.load_state_dict(checkpoint)
                    print(f'✅ Loaded model weights (legacy format)')
                    print(f'   ⚠️  Starting from epoch 0 (optimizer state not saved)')
                    print(f'   ℹ️  This is a fresh start with pre-trained weights')
                    is_resuming = False
                    
            except Exception as e:
                print(f'❌ Error loading checkpoint: {e}')
                print(f'   Starting training from scratch...')
                import traceback
                traceback.print_exc()
                start_epoch = 0
                best_val_eer = float('inf')
                is_resuming = False
            
            print(f'{"="*70}\n')
    else:
        # No checkpoint specified - check if auto-resume is possible
        print(f'\n{"="*70}')
        print('🔍 Checking for existing checkpoints...')
        print(f'{"="*70}')
        
        # Look for latest checkpoint in model save path using helper function
        if os.path.exists(model_save_path):
            latest_checkpoint_path, latest_epoch = find_latest_checkpoint(model_save_path)
            
            if latest_checkpoint_path:
                latest_file = os.path.basename(latest_checkpoint_path)
                print(f'📦 Found existing checkpoint: {latest_file}')
                print(f'   Epoch: {latest_epoch}')
                
                # Ask user if they want to resume (or auto-resume)
                print(f'\n   💡 You can resume from this checkpoint by using:')
                print(f'      --resume')
                print(f'\n   💡 Or with a specific checkpoint:')
                print(f'      --resume --model_path {latest_checkpoint_path}')
                print(f'\n   🆕 Starting fresh training instead...')
            else:
                print(f'✨ No valid checkpoints found - starting fresh training')
        else:
            print(f'✨ Model directory does not exist yet - starting fresh training')
        
        print(f'{"="*70}\n')

    # Print training mode
    print(f'\n{"="*70}')
    if is_resuming:
        print(f'🔄 RESUMING TRAINING FROM CHECKPOINT')
        print(f'{"="*70}')
        print(f'📍 Starting epoch: {start_epoch + 1}')
        print(f'🎯 Target epoch: {args.num_epochs}')
        print(f'📊 Epochs remaining: {args.num_epochs - start_epoch}')
        print(f'🏆 Best EER to beat: {best_val_eer:.2f}%')
    else:
        print(f'✨ FRESH TRAINING START')
        print(f'{"="*70}')
        print(f'📍 Starting from epoch: 1')
        print(f'🎯 Total epochs: {args.num_epochs}')
    print(f'{"="*70}\n')

    # ========== Evaluation Mode (Check Early) ==========
    if args.is_eval:
        print(f'\n{"="*70}')
        print('🔍 EVALUATION MODE')
        print(f'{"="*70}')
        
        # Check if model is loaded
        if not args.model_path:
            print('❌ Error: --model_path is required for evaluation mode')
            sys.exit(1)
        
        # Load evaluation file list
        # protocols_path should be the direct path to the protocol file
        file_eval = genSpoof_list(
            dir_meta=args.protocols_path,
            is_train=False,
            is_eval=True
        )
        
        print(f'📊 Track: {track}')
        print(f'📊 Evaluation samples: {len(file_eval)}')
        
        # Determine evaluation dataset based on track
        if track == 'In-the-Wild':
            print('📊 Using In-the-Wild dataset loader...')
            eval_set = Dataset_in_the_wild_eval(
                list_IDs=file_eval,
                base_dir=args.database_path
            )
        else:
            # For LA and DF tracks, use ASVspoof2021 eval dataset
            print(f'📊 Using ASVspoof2021 {track} eval dataset loader...')
            eval_set = Dataset_ASVspoof2021_eval(
                list_IDs=file_eval,
                base_dir=args.database_path
            )
        
        # Set output path
        if args.eval_output:
            eval_output_path = args.eval_output
        else:
            # Default output path
            eval_output_path = os.path.join('scores', f'scores_{track}.txt')
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(eval_output_path), exist_ok=True)
        
        # Clear output file if it exists
        if os.path.exists(eval_output_path):
            os.remove(eval_output_path)
            print(f'   Cleared existing output file: {eval_output_path}')
        
        print(f'   Output file: {eval_output_path}')
        
        # Run evaluation
        produce_evaluation_file(
            eval_set, model, device, eval_output_path, use_contrastive_training
        )
        
        print(f'\n{"="*70}')
        print('✅ Evaluation completed!')
        print(f'📁 Scores saved to: {eval_output_path}')
        print(f'{"="*70}\n')
        
        sys.exit(0)
    
    # ========== Training Mode - Data Loading ==========
    print('🔄 Loading datasets...')
    
    # Generate file lists - protocols are ALWAYS in ASVspoof2019_LA_cm_protocols/
    # regardless of track (DF or LA)
    d_label_trn, file_train = genSpoof_list(
        dir_meta=os.path.join(
            args.protocols_path,
            'ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'
        ),
        is_train=True,
        is_eval=False
    )
    
    print(f'   Training samples: {len(file_train)}')
    
    d_label_dev, file_dev = genSpoof_list(
        dir_meta=os.path.join(
            args.protocols_path,
            'ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'
        ),
        is_train=False,
        is_eval=False
    )
    
    print(f'   Validation samples: {len(file_dev)}')
    
    # Create datasets - but use track-specific data directories
    # Note: Dataset_ASVspoof2019_train signature is (args, list_IDs, labels, base_dir, algo)
    train_set = Dataset_ASVspoof2019_train(
        args,
        list_IDs=file_train,
        labels=d_label_trn,
        base_dir=os.path.join(args.database_path, 'ASVspoof2019_{}_train/'.format(track)),
        algo=args.algo
    )
    
    dev_set = Dataset_ASVspoof2019_train(
        args,
        list_IDs=file_dev,
        labels=d_label_dev,
        base_dir=os.path.join(args.database_path, 'ASVspoof2019_{}_dev/'.format(track)),
        algo=args.algo
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
        drop_last=True,
        pin_memory=True,
        num_workers=8
    )
    
    print('✅ Datasets loaded successfully\n')
    
    # ========== Training Mode ==========
    # Training and validation 
    num_epochs = args.num_epochs
    writer = SummaryWriter('logs/{}'.format(model_tag))

    patience_counter = 0

    for epoch in range(start_epoch, num_epochs):  # Start from checkpoint epoch
        print(f'\n{"="*70}')
        if is_resuming and epoch == start_epoch:
            print(f'📅 Epoch: {epoch+1}/{num_epochs} 🔄 RESUMED')
        else:
            print(f'📅 Epoch: {epoch+1}/{num_epochs}')
        print(f'{"="*70}')
        
        # Training
        train_loss, train_eer = train_epoch(
            train_loader, model, args.lr, optimizer, device, criterion, use_contrastive_training
        )
        
        # Validation  
        val_loss, val_acc, val_eer = evaluate_accuracy(
            dev_loader, model, device, criterion, use_contrastive_training
        )
        
        print(f'\n📊 Training   - Loss: {train_loss:.6f}, EER: {train_eer:.2f}%')
        print(f'📊 Validation - Loss: {val_loss:.6f}, Acc: {val_acc:.2f}%, EER: {val_eer:.2f}%')
        
        # Log to CSV
        log_data = {
            'epoch': epoch,
            'timestamp': datetime.now().strftime('%d/%m/%Y %H:%M'),
            'train_loss': train_loss,
            'train_eer': train_eer,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_eer': val_eer
        }
        log_training_metrics(log_data, model_save_path)
        
        # TensorBoard logging
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/eer', train_eer, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/accuracy', val_acc, epoch)
        writer.add_scalar('val/eer', val_eer, epoch)
        
        # ========== Enhanced Checkpoint Saving ==========
        # Strategy: Save full checkpoint only for best model, weights-only for regular epochs
        
        # Save best model based on EER
        if val_eer < best_val_eer:
            best_val_eer = val_eer
            patience_counter = 0
            
            # Save FULL checkpoint for best model (with optimizer state for resuming)
            checkpoint_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'train_eer': train_eer,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_eer': val_eer,
                'best_val_eer': best_val_eer,
                'args': vars(args),  # Save hyperparameters
            }
            
            if args.comment:
                best_checkpoint_filename = f'best_checkpoint_eer_{args.comment}.pth'
            else:
                best_checkpoint_filename = 'best_checkpoint_eer.pth'
            
            torch.save(checkpoint_dict, os.path.join(model_save_path, best_checkpoint_filename))
            print(f'🎉 New best model saved with EER: {best_val_eer:.2f}% (full checkpoint)')
        else:
            patience_counter += 1
        
        if args.comment:
            checkpoint_filename = f'checkpoint_epoch_{epoch}_{args.comment}.pth'
        else:
            checkpoint_filename = f'checkpoint_epoch_{epoch}.pth'
        
        # Only save model weights (no optimizer state)
        torch.save(model.state_dict(), os.path.join(model_save_path, checkpoint_filename))
        print(f'💾 Epoch checkpoint saved (weights only, ~1.2GB)')

        # Generate attention heatmaps at specified intervals
        if args.visualize_attention and VISUALIZATION_AVAILABLE and args.model_type == 'g3_heatmap':
            should_visualize = (
                (args.viz_frequency > 0 and (epoch + 1) % args.viz_frequency == 0) or
                (epoch + 1 == num_epochs)  # Always visualize at the end
            )
            
            if should_visualize:
                print(f'\n🎨 Generating attention heatmaps for epoch {epoch+1}...')
                viz_save_dir = os.path.join(model_save_path, f'attention_epoch_{epoch+1}')
                try:
                    analyze_attention_patterns(
                        model, dev_loader, device, 
                        num_samples=args.viz_samples, 
                        save_dir=viz_save_dir
                    )
                except Exception as e:
                    print(f'⚠️  Warning: Failed to generate attention heatmaps: {e}')

    writer.close()
    
    # Final attention visualization
    if args.visualize_attention and VISUALIZATION_AVAILABLE and args.model_type == 'g3_heatmap':
        print(f'\n{"="*70}')
        print('🎨 Generating final attention heatmap analysis...')
        print(f'{"="*70}')
        final_viz_dir = os.path.join(model_save_path, 'final_attention_analysis')
        try:
            analyze_attention_patterns(
                model, dev_loader, device, 
                num_samples=min(50, args.viz_samples * 2),  # More samples for final analysis
                save_dir=final_viz_dir
            )
        except Exception as e:
            print(f'⚠️  Warning: Failed to generate final attention heatmaps: {e}')

    print(f'\n{"="*70}')
    print('✅ Training completed!')
    print(f'📁 Models saved to: {model_save_path}')
    print(f'🏆 Best validation EER: {best_val_eer:.2f}%')
    print(f'{"="*70}\n')