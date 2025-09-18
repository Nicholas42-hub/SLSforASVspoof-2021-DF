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
from model_v1_tempscaled import Model, SupConLoss
from tensorboardX import SummaryWriter
from core_scripts.startup_config import set_random_seed
from tqdm import tqdm
from torchvision import transforms
from sklearn.metrics import roc_curve
from model_v1_tempscaled import Model, SupConLoss, EmbeddingVisualizer

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
            'epoch', 'timestamp', 'train_loss', 'train_cce', 'train_contrastive', 'train_eer',
            'val_loss', 'val_cce', 'val_contrastive', 'val_acc', 'val_eer'
        ])

def log_training_metrics(log_data, model_save_path):
    """Log training metrics to CSV file"""
    log_path = os.path.join(model_save_path, 'training_log.csv')
    
    with open(log_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            log_data['epoch'],
            log_data['timestamp'],
            log_data['train_total_loss'],
            log_data['train_cce'],
            log_data['train_contrastive'],
            log_data['train_eer'],
            log_data['val_total_loss'],
            log_data['val_cce'],
            log_data['val_contrastive'],
            log_data['val_acc'],
            log_data['val_eer']
        ])

def evaluate_accuracy(dev_loader, model, device, criterion_cce, contrastive_weight, test_mode=False, test_batches=5):
    val_loss = 0.0
    val_cce_loss = 0.0
    val_contrastive_loss = 0.0
    num_correct = 0.0
    num_total = 0.0
    
    # For EER calculation
    all_scores = []
    all_labels = []
    
    model.eval()
    
    batch_count = 0
    for batch_x, batch_y in tqdm(dev_loader):
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        
        with torch.no_grad():
            # 获取分类输出和对比损失 (fixed unpacking)
            batch_out, contrastive_loss, per_class_stats = model(batch_x, batch_y, return_contrastive=True)
            _, batch_pred = batch_out.max(dim=1)
            num_correct += (batch_pred == batch_y).sum(dim=0).item()
            
            # 计算分类损失
            cce_loss = criterion_cce(batch_out, batch_y)
            
            # 总损失
            batch_loss = cce_loss + contrastive_weight * contrastive_loss
            
            val_loss += (batch_loss.item() * batch_size)
            val_cce_loss += (cce_loss.item() * batch_size)
            val_contrastive_loss += (contrastive_loss.item() * batch_size)
            
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
        
        batch_count += 1
        
        # Test mode: only run a few batches
        if test_mode and batch_count >= test_batches:
            print(f"Test mode: stopping validation after {batch_count} batches")
            break

    val_loss /= num_total
    val_cce_loss /= num_total
    val_contrastive_loss /= num_total
    acc = 100 * (num_correct / num_total)
    
    # Compute EER
    if len(all_scores) > 0 and len(all_labels) > 0:
        eer, eer_threshold = compute_eer(all_scores, all_labels)
    else:
        eer, eer_threshold = 50.0, 0.5
        print("Warning: No valid scores for EER calculation, using default EER=50%")
    
    return val_loss, val_cce_loss, val_contrastive_loss, acc, eer

def produce_evaluation_file(dataset, model, device, save_path):
    data_loader = DataLoader(
        dataset, 
        batch_size=20,  # Increase from default (probably 1)
        shuffle=False, 
        drop_last=False, 
        pin_memory=True,  # Faster GPU transfer
        num_workers=6     # Parallel data loading
    )
    num_correct = 0.0
    num_total = 0.0
    model.eval()
    
    fname_list = []
    key_list = []
    score_list = []
    
    for batch_x,utt_id in tqdm(data_loader):
        fname_list = []
        score_list = []  
        batch_size = batch_x.size(0)
        batch_x = batch_x.to(device)
        torch.set_printoptions(threshold=10_000)
        batch_out = model(batch_x)
        batch_score = (batch_out[:, 1]
                       ).data.cpu().numpy().ravel() 
        # add outputs
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())
        
        with open(save_path, 'a+') as fh:
            for f, cm in zip(fname_list,score_list):
                fh.write('{} {}\n'.format(f, cm))
        fh.close()   
    print('Scores saved to {}'.format(save_path))
def evaluate_with_visualization(dev_loader, model, device, criterion_cce, contrastive_weight, 
                               epoch, save_dir, strategy="hierarchical", skip_viz=False, test_mode=False):
    """Enhanced evaluation with embedding visualization"""
    val_loss = 0.0
    val_cce_loss = 0.0
    val_contrastive_loss = 0.0
    num_correct = 0.0
    num_total = 0.0
    
    all_scores = []
    all_labels = []
    per_class_stats_accumulator = {'class_0': [], 'class_1': []}
    
    model.eval()
    # Fix: Access the underlying module when using DataParallel
    if hasattr(model, 'module'):
        model.module.clear_stored_embeddings()  # Clear previous embeddings
    else:
        model.clear_stored_embeddings()  # Clear previous embeddings
    
    batch_count = 0
    max_test_batches = 3 if test_mode else float('inf')
    
    for batch_x, batch_y in tqdm(dev_loader, desc="Validation"):
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        
        with torch.no_grad():
            # Store embeddings for visualization only if not skipping
            store_embeddings = not skip_viz
            if strategy == "data_augment":
                # For data augment strategy, we need augmented data
                batch_out, contrastive_loss, per_class_stats = model(
                    batch_x, batch_y, x_augmented=batch_x, return_contrastive=True, store_embeddings=store_embeddings)
            else:
                batch_out, contrastive_loss, per_class_stats = model(
                    batch_x, batch_y, return_contrastive=True, store_embeddings=store_embeddings)
            
            _, batch_pred = batch_out.max(dim=1)
            num_correct += (batch_pred == batch_y).sum(dim=0).item()
            
            cce_loss = criterion_cce(batch_out, batch_y)
            batch_loss = cce_loss + contrastive_weight * contrastive_loss
            
            val_loss += (batch_loss.item() * batch_size)
            val_cce_loss += (cce_loss.item() * batch_size)
            val_contrastive_loss += (contrastive_loss.item() * batch_size)
            
            # Accumulate per-class statistics
            for key, value in per_class_stats.items():
                if key in per_class_stats_accumulator:
                    per_class_stats_accumulator[key].append(value)
            
            # Collect scores for EER
            batch_probs = torch.exp(batch_out)
            batch_scores = batch_probs[:, 1].cpu().numpy()
            batch_labels = batch_y.cpu().numpy()
            
            valid_mask = ~(np.isnan(batch_scores) | np.isinf(batch_scores))
            if np.any(valid_mask):
                all_scores.extend(batch_scores[valid_mask].tolist())
                all_labels.extend(batch_labels[valid_mask].tolist())
        
        batch_count += 1
        if test_mode and batch_count >= max_test_batches:
            print(f"Test mode: stopping visualization validation after {batch_count} batches")
            break

    # Compute averages
    val_loss /= num_total
    val_cce_loss /= num_total
    val_contrastive_loss /= num_total
    acc = 100 * (num_correct / num_total)
    
    # Compute EER
    if len(all_scores) > 0 and len(all_labels) > 0:
        eer, eer_threshold = compute_eer(all_scores, all_labels)
    else:
        eer, eer_threshold = 50.0, 0.5
    
    # Compute average per-class statistics
    avg_per_class_stats = {}
    for key, values in per_class_stats_accumulator.items():
        if values:
            avg_per_class_stats[key] = np.mean(values)
    
    # Skip visualization if requested
    if skip_viz:
        print("Skipping visualization as requested.")
        separation_metrics = {}
    else:
        # Visualization and separation analysis
        # Fix: Access the underlying module when using DataParallel
        if hasattr(model, 'module'):
            embeddings, labels = model.module.get_embeddings_for_visualization()
        else:
            embeddings, labels = model.get_embeddings_for_visualization()
            
        if embeddings is not None and labels is not None:
            try:
                visualizer = EmbeddingVisualizer()
                
                # Create visualization directory
                vis_dir = os.path.join(save_dir, 'visualizations')
                os.makedirs(vis_dir, exist_ok=True)
                
                # Plot t-SNE only (skip UMAP to avoid import issues)
                tsne_path = os.path.join(vis_dir, f'tsne_epoch_{epoch}_{strategy}.png')
                visualizer.plot_embeddings(embeddings, labels, method='tsne', 
                                          save_path=tsne_path, 
                                          title=f"t-SNE Epoch {epoch} ({strategy})")
                
                # Compute separation metrics
                separation_metrics = visualizer.compute_separation_metrics(embeddings, labels)
                
                print(f"Separation Metrics (Epoch {epoch}):")
                for key, value in separation_metrics.items():
                    print(f"  {key}: {value:.4f}")
                
            except Exception as e:
                print(f"Visualization failed: {e}")
                separation_metrics = {}
        else:
            # Return empty separation metrics if no embeddings available
            separation_metrics = {}
    
    print(f"Per-class Contrastive Loss:")
    for key, value in avg_per_class_stats.items():
        print(f"  {key}: {value:.4f}")
    
    return val_loss, val_cce_loss, val_contrastive_loss, acc, eer, separation_metrics, avg_per_class_stats

def train_epoch(train_loader, model, lr, optim, device, criterion_cce, contrastive_weight, test_mode=False, test_batches=5):
    running_loss = 0
    running_cce_loss = 0
    running_contrastive_loss = 0
    num_total = 0.0
    
    # For training EER calculation (optional)
    all_scores = []
    all_labels = []
    
    model.train()
    
    batch_count = 0
    for batch_x, batch_y in tqdm(train_loader):
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        
        # Forward pass - 获取分类输出和对比损失 (fixed unpacking)
        batch_out, contrastive_loss, per_class_stats = model(batch_x, batch_y, return_contrastive=True)
        
        # 计算分类损失
        cce_loss = criterion_cce(batch_out, batch_y)
        
        # 总损失
        batch_loss = cce_loss + contrastive_weight * contrastive_loss
        
        running_loss += (batch_loss.item() * batch_size)
        running_cce_loss += (cce_loss.item() * batch_size)
        running_contrastive_loss += (contrastive_loss.item() * batch_size)
        
        # Collect scores for training EER (optional, computed with no_grad for efficiency)
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
        
        batch_count += 1
        
        # Test mode: only run a few batches
        if test_mode and batch_count >= test_batches:
            print(f"Test mode: stopping after {batch_count} batches")
            break
    
    running_loss /= num_total
    running_cce_loss /= num_total
    running_contrastive_loss /= num_total
    
    # Compute training EER
    if len(all_scores) > 0 and len(all_labels) > 0:
        train_eer, _ = compute_eer(all_scores, all_labels)
    else:
        train_eer = 50.0
    
    return running_loss, running_cce_loss, running_contrastive_loss, train_eer

# 在参数解析器中添加新参数


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ASVspoof2021 Hierarchical Attention Contrastive Learning System')
    parser.add_argument('--contrastive_strategy', type=str, default='hierarchical',
                    choices=['hierarchical', 'local_patch', 'data_augment'],
                    help='Contrastive learning strategy')
    parser.add_argument('--visualization_interval', type=int, default=5,
                        help='Interval for generating visualizations')
    # Dataset
    parser.add_argument('--database_path', type=str, default='/root/autodl-tmp/CLAD/Datasets/LA/', help='Change this to user\'s full directory address of LA database (ASVspoof2019- for training & development (used as validation), ASVspoof2021 DF for evaluation scores). We assume that all three ASVspoof 2019 LA train, LA dev and ASVspoof2021 DF eval data folders are in the same database_path directory.')
    parser.add_argument('--protocols_path', type=str, default='/root/autodl-tmp/CLAD/Datasets/LA/', help='Change with path to user\'s DF database protocols directory address')
    
    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=14)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.000001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--loss', type=str, default='combined_CCE_contrastive')
    
    # Hierarchical attention parameters
    parser.add_argument('--group_size', type=int, default=3,
                        help='Group size for hierarchical attention')
    
    # Contrastive learning parameters
    parser.add_argument('--contrastive_weight', type=float, default=0,
                        help='Weight for contrastive loss')
    parser.add_argument('--contrastive_temperature', type=float, default=0.1,  # Changed from --temperature
                        help='Temperature for contrastive loss')
    parser.add_argument('--contrastive_frequency', type=int, default=1,
                    help='Apply contrastive loss every N batches (1=every batch)')
    parser.add_argument('--fast_mode', action='store_true', default=False,
                    help='Use faster approximation for contrastive learning')
    
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
    
    # Test mode - quick test with minimal batches
    parser.add_argument('--test', action='store_true', default=False,
                        help='Run in test mode (only few batches per epoch for quick testing)')
    parser.add_argument('--test_batches', type=int, default=5,
                        help='Number of batches to run in test mode (default: 5)')
    
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
                    help='maximum width [Hz] of filter.[default=1000]')
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

    #define model saving path
    model_tag = 'model_v1_tempscaled_{}_{}_{}_{}_{}'.format(
        track, args.loss, args.num_epochs, args.batch_size, args.lr)
    if args.comment:
        model_tag = model_tag + '_{}'.format(args.comment)
    model_save_path = os.path.join('models', model_tag)

    #set model save directory
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
    
    # Initialize CSV log
    init_csv_log(os.path.join(model_save_path, 'training_log.csv'))
    
    #GPU device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'                  
    print('Device: {}'.format(device))
    
    # 使用新的模型类
    model = Model(args, device)
    args.contrastive_temperature = args.contrastive_temperature  # No change needed since it's already the correct name
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])

    model = nn.DataParallel(model).to(device)
    print('nb_params:', nb_params)

    # 设置损失函数 - 分别设置分类损失和对比损失权重
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion_cce = nn.NLLLoss(weight=weight)
    contrastive_weight = args.contrastive_weight

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
        train_loss, train_cce, train_contrastive, train_eer = train_epoch(
            train_loader, model, args.lr, optimizer, device, criterion_cce, contrastive_weight, 
            test_mode=args.test, test_batches=args.test_batches
        )
        
        # Validation  
        val_loss, val_cce, val_contrastive, val_acc, val_eer = evaluate_accuracy(
            dev_loader, model, device, criterion_cce, contrastive_weight,
            test_mode=args.test, test_batches=args.test_batches
        )
        
        print(f'Train Loss: {train_loss:.6f} (CCE: {train_cce:.6f}, Contrastive: {train_contrastive:.6f}), EER: {train_eer:.2f}%')
        print(f'Val Loss: {val_loss:.6f} (CCE: {val_cce:.6f}, Contrastive: {val_contrastive:.6f})')
        print(f'Val Acc: {val_acc:.2f}%, EER: {val_eer:.2f}%')
        
        # Log to CSV
        log_data = {
            'epoch': epoch,
            'timestamp': datetime.now().strftime('%d/%m/%Y %H:%M'),
            'train_total_loss': train_loss,
            'train_cce': train_cce,
            'train_contrastive': train_contrastive,
            'train_eer': train_eer,
            'val_total_loss': val_loss,
            'val_cce': val_cce,
            'val_contrastive': val_contrastive,
            'val_acc': val_acc,
            'val_eer': val_eer
        }
        log_training_metrics(log_data, model_save_path)
        
        # TensorBoard logging
        writer.add_scalar('train/total_loss', train_loss, epoch)
        writer.add_scalar('train/cce_loss', train_cce, epoch)
        writer.add_scalar('train/contrastive_loss', train_contrastive, epoch)
        writer.add_scalar('train/eer', train_eer, epoch)
        writer.add_scalar('val/total_loss', val_loss, epoch)
        writer.add_scalar('val/cce_loss', val_cce, epoch)
        writer.add_scalar('val/contrastive_loss', val_contrastive, epoch)
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

        # Visualization and separation analysis every few epochs (or test mode)
        if (not args.test and (epoch % args.visualization_interval == 0 or epoch == num_epochs - 1)) or args.test:
            print("Generating visualizations...")
            evaluate_with_visualization(dev_loader, model, device, criterion_cce, contrastive_weight, 
                                       epoch, model_save_path, strategy=args.contrastive_strategy,
                                       skip_viz=False, test_mode=args.test)
        
        # In test mode, print quick summary and exit after first epoch
        if args.test:
            print(f"✅ Test mode completed successfully!")
            print(f"Forward pass: ✓")
            print(f"Backward pass: ✓") 
            print(f"Loss computation: ✓")
            print(f"All functions working correctly. You can now run full training.")
            break

    writer.close()
    print('Training completed!')



    

# CUDA_VISIBLE_DEVICES=0 python model_local_patch.py --track=In-the-Wild --eval --model_path=/root/autodl-tmp/SLSforASVspoof-2021-DF/models/Ablation/epoch_8_local_patch_contrastive_fixed.pth --protocols_path=/root/autodl-tmp/CLAD/Datasets/release_in_the_wild/ --database_path=/root/autodl-tmp/CLAD/Datasets/release_in_the_wild/ --eval_output=/root/autodl-tmp/SLSforASVspoof-2021-DF/scores_model_best_model_eer_local_patch_wild.txt