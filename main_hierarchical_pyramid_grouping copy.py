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
# 修改1: 支持多个模型导入
from model_pyramid_grouping import Model as PyramidModel
from model_pyramid_grouping import FusionBestModel
from tensorboardX import SummaryWriter
from core_scripts.startup_config import set_random_seed
from tqdm import tqdm
from torchvision import transforms
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

def evaluate_accuracy(dev_loader, model, device, criterion):
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
            batch_out = model(batch_x)
            _, batch_pred = batch_out.max(dim=1)
            num_correct += (batch_pred == batch_y).sum(dim=0).item()
            
            # 计算损失
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

def produce_evaluation_file(dataset, model, device, save_path):
    data_loader = DataLoader(
        dataset, 
        batch_size=20,  # Increase from default (probably 1)
        shuffle=False, 
        drop_last=False, 
        pin_memory=True,  # Faster GPU transfer
        num_workers=6     # Parallel data loading
    )
    model.eval()
    
    fname_list = []
    score_list = []
    
    for batch_x, utt_id in tqdm(data_loader):
        batch_size = batch_x.size(0)
        batch_x = batch_x.to(device)
        
        with torch.no_grad():
            batch_out = model(batch_x)
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

def train_epoch(train_loader, model, lr, optim, device, criterion):
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
        batch_out = model(batch_x)
        
        # 计算损失
        batch_loss = criterion(batch_out, batch_y)
        
        running_loss += (batch_loss.item() * batch_size)
        
        # Collect scores for training EER (computed with no_grad for efficiency)
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ASVspoof2021 Pyramid Hierarchical Attention System')
    
    # Dataset
    parser.add_argument('--database_path', type=str, default='/root/autodl-tmp/CLAD/Datasets/LA/', help='Change this to user\'s full directory address of LA database (ASVspoof2019- for training & development (used as validation), ASVspoof2021 DF for evaluation scores). We assume that all three ASVspoof 2019 LA train, LA dev and ASVspoof2021 DF eval data folders are in the same database_path directory.')
    parser.add_argument('--protocols_path', type=str, default='/root/autodl-tmp/CLAD/Datasets/LA/', help='Change with path to user\'s DF database protocols directory address')
    
    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=14)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.000001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--loss', type=str, default='CCE')
    
    # 修改2: 添加模型选择参数
    parser.add_argument('--model_type', type=str, default='pyramid', 
                        choices=['pyramid', 'fusion'],
                        help='Model type: pyramid (original) or fusion (best)')
    
    # ========== Pyramid Hierarchical Attention Parameters ==========
    # 旧的参数（保留兼容性）
    parser.add_argument('--group_size', type=int, default=3,
                        help='Group size for fixed grouping (used when use_pyramid=False)')
    
    # 新的Pyramid参数
    parser.add_argument('--use_pyramid', action='store_true', default=False,
                        help='Use pyramid hierarchical attention (default: False, use fixed grouping)')
    parser.add_argument('--base_dim', type=int, default=256,
                        help='Base dimension for pyramid (default: 256)')
    parser.add_argument('--merge_ratio', type=int, default=3,
                        help='Merge ratio for pyramid: how many layers merge to 1 group (default: 3)')
    parser.add_argument('--num_stages', type=int, default=3,
                        help='Number of pyramid stages (default: 3)')
    
    # 修改3: 添加 Fusion Model 参数
    parser.add_argument('--low_dim', type=int, default=256,
                        help='Low dimension for fusion model (default: 256)')
    parser.add_argument('--num_scales', type=int, default=3,
                        help='Number of temporal scales for fusion model (default: 3)')
    
    # model
    parser.add_argument('--seed', type=int, default=1234,
                        help='random seed (default: 1234)')
    
    parser.add_argument('--model_path', type=str,
                        default=None, help='Model checkpoint')
    parser.add_argument('--comment', type=str, default=None,
                        help='Comment to describe the saved model and experiment')
    
    # Auxiliary arguments
    parser.add_argument('--track', type=str, default='DF',choices=['LA', 'In-the-Wild','DF'], help='LA/PA/DF')
    parser.add_argument('--eval_output', type=str, default=None,
                        help='Path to save the evaluation result')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='eval mode')
    parser.add_argument('--is_eval', action='store_true', default=False,help='eval database')
    parser.add_argument('--eval_part', type=int, default=0)
    
    # backend options
    parser.add_argument('--cudnn-deterministic-toggle', action='store_false', \
                        default=True, 
                        help='use cudnn-deterministic? (default true)')    
    
    parser.add_argument('--cudnn-benchmark-toggle', action='store_true', \
                        default=False, 
                        help='use cudnn-benchmark? (default false)') 

    # Rawboost data augmentation parameters
    parser.add_argument('--algo', type=int, default=3, 
                    help='Rawboost algos discriptions. 0: No augmentation 1: LnL_convolutive_noise, 2: ISD_additive_noise, 3: SSI_additive_noise, 4: series algo (1+2+3), \
                          5: series algo (1+2), 6: series algo (1+3), 7: series algo(2+3), 8: parallel algo(1,2) .default=0]')

    # LnL_convolutive_noise parameters 
    parser.add_argument('--nBands', type=int, default=5, 
                    help='number of notch filters.The higher the number of bands, the more aggresive the distortions is.[default=5]')
    parser.add_argument('--minF', type=int, default=20, 
                    help='minimum centre frequency [Hz] of notch filter.[default=20] ')
    parser.add_argument('--maxF', type=int, default=8000, 
                    help='maximum centre frequency [Hz] (<sr/2)  of notch filter.[default=8000]')
    parser.add_argument('--minBW', type=int, default=100, 
                    help='minimum width [Hz] of filter.[default=100] ')
    parser.add_argument('--maxBW', type=int, default=1000, 
                    help='maximum width [Hz] of filter.[default=1000] ')
    parser.add_argument('--minCoeff', type=int, default=10, 
                    help='minimum filter coefficients. More the filter coefficients more ideal the filter slope.[default=10]')
    parser.add_argument('--maxCoeff', type=int, default=100, 
                    help='maximum filter coefficients. More the filter coefficients more ideal the filter slope.[default=100]')
    parser.add_argument('--minG', type=int, default=0, 
                    help='minimum gain factor of linear component.[default=0]')
    parser.add_argument('--maxG', type=int, default=0, 
                    help='maximum gain factor of linear component.[default=0]')
    parser.add_argument('--minBiasLinNonLin', type=int, default=5, 
                    help=' minimum gain difference between linear and non-linear components.[default=5]')
    parser.add_argument('--maxBiasLinNonLin', type=int, default=20, 
                    help=' maximum gain difference between linear and non-linear components.[default=20]')
    parser.add_argument('--N_f', type=int, default=5, 
                    help='order of the (non-)linearity where N_f=1 refers only to linear components.[default=5]')

    # ISD_additive_noise parameters
    parser.add_argument('--P', type=int, default=10, 
                    help='Maximum number of uniformly distributed samples in [%].[defaul=10]')
    parser.add_argument('--g_sd', type=int, default=2, 
                    help='gain parameters > 0. [default=2]')

    # SSI_additive_noise parameters
    parser.add_argument('--SNRmin', type=int, default=10, 
                    help='Minimum SNR value for coloured additive noise.[defaul=10]')
    parser.add_argument('--SNRmax', type=int, default=40, 
                    help='Maximum SNR value for coloured additive noise.[defaul=40]')

    if not os.path.exists('models'):
        os.mkdir('models')
    args = parser.parse_args()
 
    #make experiment reproducible
    set_random_seed(args.seed, args)
    
    track = args.track

    # 修改4: 根据模型类型生成不同的模型标签
    if args.model_type == 'fusion':
        model_tag = 'model_fusion_{}_{}_{}_{}_{}_scales{}'.format(
            track, args.loss, args.num_epochs, args.batch_size, args.lr, args.num_scales)
    else:
        model_tag = 'model_{}hierarchical_{}_{}_{}_{}_{}'.format(
            'pyramid_' if args.use_pyramid else '',
            track, args.loss, args.num_epochs, args.batch_size, args.lr)
    
    if args.comment:
        model_tag = model_tag + '_{}'.format(args.comment)
    model_save_path = os.path.join('models', model_tag)

    #set model save directory
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
    
    # Initialize CSV log
    init_csv_log(os.path.join(model_save_path, 'training_log.csv'))
    
    # 修改5: 打印不同模型的配置
    print('=' * 50)
    print('Configuration:')
    print(f'  Model Type: {args.model_type}')
    if args.model_type == 'fusion':
        print(f'  Low Dim: {args.low_dim}')
        print(f'  Num Scales: {args.num_scales}')
    else:
        print(f'  Use Pyramid: {args.use_pyramid}')
        if args.use_pyramid:
            print(f'  Base Dim: {args.base_dim}')
            print(f'  Merge Ratio: {args.merge_ratio}')
            print(f'  Num Stages: {args.num_stages}')
        else:
            print(f'  Group Size: {args.group_size}')
    print('=' * 50)
    
    #GPU device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'                  
    print('Device: {}'.format(device))
    
    # 修改6: 根据参数选择模型
    if args.model_type == 'fusion':
        model = FusionBestModel(args, device)
        print('Using Fusion Best Model')
    else:
        model = PyramidModel(args, device)
        print('Using Pyramid Model')
    
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])

    model = nn.DataParallel(model).to(device)
    print('nb_params:', nb_params)

    # 设置损失函数
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.NLLLoss(weight=weight)

    #set Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print('Model loaded : {}'.format(args.model_path))

    # evaluation mode on the In-the-Wild dataset.
    if args.track == 'In-the-Wild':
        file_eval = genSpoof_list(dir_meta=os.path.join(args.protocols_path), is_train=False, is_eval=True)
        print('no. of eval trials', len(file_eval))
        eval_set = Dataset_in_the_wild_eval(list_IDs=file_eval, base_dir=os.path.join(args.database_path))
        produce_evaluation_file(eval_set, model, device, args.eval_output)
        sys.exit(0)

    # evaluation mode on the DF or LA dataset.
    if args.eval:
        file_eval = genSpoof_list(dir_meta=os.path.join(args.protocols_path), is_train=False, is_eval=True)
        print('no. of eval trials', len(file_eval))
        eval_set = Dataset_ASVspoof2021_eval(list_IDs=file_eval, base_dir=os.path.join(args.database_path))
        produce_evaluation_file(eval_set, model, device, args.eval_output)
        sys.exit(0)
   
    # define train dataloader
    d_label_trn, file_train = genSpoof_list(dir_meta=os.path.join(args.protocols_path+'ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'), is_train=True, is_eval=False)
    
    print('no. of training trials', len(file_train))
    
    train_set = Dataset_ASVspoof2019_train(args, list_IDs=file_train, labels=d_label_trn, base_dir=os.path.join(args.database_path+'ASVspoof2019_LA_train/'), algo=args.algo)
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=8, shuffle=True, drop_last=True)
    
    del train_set, d_label_trn
    
    # define dev (validation) dataloader
    d_label_dev, file_dev = genSpoof_list(dir_meta=os.path.join(args.protocols_path+'ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'), is_train=False, is_eval=False)

    print('no. of validation trials', len(file_dev))

    dev_set = Dataset_ASVspoof2019_train(args, list_IDs=file_dev, labels=d_label_dev, base_dir=os.path.join(args.database_path+'ASVspoof2019_LA_dev/'), algo=args.algo)

    dev_loader = DataLoader(dev_set, batch_size=args.batch_size, num_workers=8, shuffle=False)

    del dev_set, d_label_dev

    # Training and validation 
    num_epochs = args.num_epochs
    writer = SummaryWriter('logs/{}'.format(model_tag))

    best_val_eer = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        print('\nEpoch: {}'.format(epoch))
        
        # Training
        train_loss, train_eer = train_epoch(
            train_loader, model, args.lr, optimizer, device, criterion
        )
        
        # Validation  
        val_loss, val_acc, val_eer = evaluate_accuracy(
            dev_loader, model, device, criterion
        )
        
        print(f'Train Loss: {train_loss:.6f}, EER: {train_eer:.2f}%')
        print(f'Val Loss: {val_loss:.6f}, Acc: {val_acc:.2f}%, EER: {val_eer:.2f}%')
        
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
        
        # Save model checkpoint
        if args.comment:
            model_filename = f'epoch_{epoch}_{args.comment}.pth'
        else:
            model_filename = f'epoch_{epoch}.pth'
            
        torch.save(model.state_dict(), os.path.join(model_save_path, model_filename))
        
        # Save best model based on EER (lower is better)
        if val_eer < best_val_eer:
            best_val_eer = val_eer
            patience_counter = 0
            if args.comment:
                best_model_filename = f'best_model_eer_{args.comment}.pth'
            else:
                best_model_filename = 'best_model_eer.pth'
            torch.save(model.state_dict(), os.path.join(model_save_path, best_model_filename))
            print(f'🎉 New best model saved with EER: {best_val_eer:.2f}%')
        else:
            patience_counter += 1

        # Optional: Early stopping
        # if patience_counter >= 10:
        #     print(f"Early stopping triggered at epoch {epoch}, best EER: {best_val_eer:.2f}%")
        #     break

    writer.close()
    print('Training completed!')