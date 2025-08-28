import argparse
import sys
import os
import numpy as np
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F  # 添加这个导入
from torch.utils.data import DataLoader
import yaml
import warnings
import csv
from datetime import datetime

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")
warnings.filterwarnings("ignore", category=UserWarning, module=".*soundfile.*")
warnings.filterwarnings("ignore", message="PySoundFile failed")

from data_utils_SSL import genSpoof_list,Dataset_ASVspoof2019_train,Dataset_ASVspoof2021_eval,Dataset_in_the_wild_eval
from model import Model
from tensorboardX import SummaryWriter
from core_scripts.startup_config import set_random_seed
from tqdm import tqdm
from torchvision import transforms
from sklearn.metrics import roc_curve

def set_deterministic_training(seed=42):
    """设置确定性训练以减少随机性影响"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 关键：设置确定性算法
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 设置环境变量
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    # PyTorch 1.8+
    if hasattr(torch, 'use_deterministic_algorithms'):
        try:
            torch.use_deterministic_algorithms(True)
        except RuntimeError:
            # 如果某些操作不支持确定性，则跳过
            print("Warning: Some operations do not support deterministic algorithms")
            pass

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def evaluate_accuracy(dev_loader, model, device):
    val_loss = 0.0
    num_correct = 0.0
    num_total = 0.0
    all_labels = []
    all_scores = []
    
    model.eval()
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    
    with torch.no_grad():  # 重要：验证时禁用梯度计算
        for batch_x, batch_y in tqdm(dev_loader):
            batch_size = batch_x.size(0)
            num_total += batch_size
            batch_x = batch_x.to(device)
            batch_y = batch_y.view(-1).type(torch.int64).to(device)
            
            # Updated to handle new model output
            batch_out, _, _, _ = model(batch_x, batch_y)
            
            # Check for NaN values in model output
            if torch.isnan(batch_out).any() or torch.isinf(batch_out).any():
                print(f"Warning: NaN or Inf detected in model output")
                # Skip this batch or handle gracefully
                continue
            
            # 使用温度缩放改善校准
            temperature = 1.5
            batch_out_calibrated = batch_out / temperature
            
            # Get prediction scores for EER calculation
            batch_scores = torch.softmax(batch_out_calibrated, dim=1)[:, 1]
            
            # Additional safety check for scores
            if torch.isnan(batch_scores).any() or torch.isinf(batch_scores).any():
                print(f"Warning: NaN or Inf detected in batch scores")
                continue
                
            all_scores.extend(batch_scores.cpu().detach().numpy())
            all_labels.extend(batch_y.cpu().numpy())
            
            _, batch_pred = batch_out_calibrated.max(dim=1)
            num_correct += (batch_pred == batch_y).sum(dim=0).item()

            batch_loss = criterion(batch_out, batch_y)
            
            # Check for NaN in loss
            if torch.isnan(batch_loss) or torch.isinf(batch_loss):
                print(f"Warning: NaN or Inf detected in loss")
                continue
                
            val_loss += (batch_loss.item() * batch_size)

    # Safety check before calculating EER
    if len(all_scores) == 0 or len(all_labels) == 0:
        print("Warning: No valid scores collected, returning default values")
        return float('inf'), 0.0, 100.0
    
    # Convert to numpy and check for NaN
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    # Remove any NaN values
    valid_mask = ~(np.isnan(all_scores) | np.isinf(all_scores))
    if not valid_mask.all():
        print(f"Warning: Found {(~valid_mask).sum()} invalid scores, removing them")
        all_scores = all_scores[valid_mask]
        all_labels = all_labels[valid_mask]
    
    # Final check
    if len(all_scores) == 0:
        print("Warning: No valid scores remaining after filtering")
        return float('inf'), 0.0, 100.0

    val_loss /= num_total
    acc = 100 * (num_correct / num_total)
    
    # Calculate EER with error handling
    try:
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(all_labels, all_scores, pos_label=1)
        fnr = 1 - tpr
        eer_idx = np.argmin(np.abs(fnr - fpr))
        eer = fpr[eer_idx] * 100
    except Exception as e:
        print(f"Error calculating EER: {e}")
        eer = 100.0  # Return worst case EER
    
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
        # Updated to handle new model output
        batch_out, _, _, _ = model(batch_x)
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

def train_epoch(train_loader, model, lr, optim, device):
    running_loss = 0
    num_total = 0.0
    
    model.train()

    # 改用Focal Loss处理类别不平衡
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    
    # 添加学习率预热
    if epoch < 3:  # 前3个epoch预热
        warmup_lr = lr * (epoch + 1) / 3
        for param_group in optim.param_groups:
            param_group['lr'] = warmup_lr
    
    for batch_x, batch_y in tqdm(train_loader):
        batch_size = batch_x.size(0)
        num_total += batch_size
        
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        
        # 添加Mixup数据增强减少过拟合
        if np.random.rand() < 0.3:  # 30%概率使用mixup
            lam = np.random.beta(0.2, 0.2)
            index = torch.randperm(batch_size).to(device)
            mixed_x = lam * batch_x + (1 - lam) * batch_x[index]
            batch_out = model(mixed_x)
            
            # Mixup loss
            batch_loss = lam * criterion(batch_out, batch_y) + \
                         (1 - lam) * criterion(batch_out, batch_y[index])
        else:
            batch_out = model(batch_x)
            batch_loss = criterion(batch_out, batch_y)
        
        running_loss += (batch_loss.item() * batch_size)
       
        optim.zero_grad()
        batch_loss.backward()
        
        # 更保守的梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        
        optim.step()
       
    running_loss /= num_total
    return running_loss

def evaluate_accuracy(dev_loader, model, device):
    val_loss = 0.0
    num_correct = 0.0
    num_total = 0.0
    all_labels = []
    all_scores = []
    
    model.eval()
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    
    with torch.no_grad():  # 重要：验证时禁用梯度计算
        for batch_x, batch_y in tqdm(dev_loader):
            batch_size = batch_x.size(0)
            num_total += batch_size
            batch_x = batch_x.to(device)
            batch_y = batch_y.view(-1).type(torch.int64).to(device)
            
            batch_out = model(batch_x)
            
            # 使用温度缩放改善校准
            temperature = 1.5
            batch_out_calibrated = batch_out / temperature
            
            # Get prediction scores for EER calculation
            batch_scores = torch.softmax(batch_out_calibrated, dim=1)[:, 1]
            all_scores.extend(batch_scores.cpu().detach().numpy())
            all_labels.extend(batch_y.cpu().numpy())
            
            _, batch_pred = batch_out_calibrated.max(dim=1)
            num_correct += (batch_pred == batch_y).sum(dim=0).item()

            batch_loss = criterion(batch_out, batch_y)
            val_loss += (batch_loss.item() * batch_size)

    val_loss /= num_total
    acc = 100 * (num_correct / num_total)
    
    # Calculate EER
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(all_labels, all_scores, pos_label=1)
    fnr = 1 - tpr
    eer_idx = np.argmin(np.abs(fnr - fpr))
    eer = fpr[eer_idx] * 100
    
    return val_loss, acc, eer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ASVspoof2021 improved system')
    # Dataset
    parser.add_argument('--database_path', type=str, default='/root/autodl-tmp/CLAD/Datasets/LA/', help='Change this to user\'s full directory address of LA database (ASVspoof2019- for training & development (used as validation), ASVspoof2021 DF for evaluation scores). We assume that all three ASVspoof 2019 LA train, LA dev and ASVspoof2021 DF eval data folders are in the same database_path directory.')
    '''
    % database_path/
    %   |- DF
    %      |- ASVspoof2021_DF_eval/flac
    %      |- ASVspoof2019_LA_train/flac
    %      |- ASVspoof2019_LA_dev/flac
    '''

    parser.add_argument('--protocols_path', type=str, default='/root/autodl-tmp/CLAD/Datasets/LA/', help='Change with path to user\'s DF database protocols directory address')
    '''
    % protocols_path/
    %   |- ASVspoof_LA_cm_protocols
    %      |- ASVspoof2021.LA.cm.eval.trl.txt
    %      |- ASVspoof2019.LA.cm.dev.trl.txt 
    %      |- ASVspoof2019.LA.cm.train.trn.txt
 
    %   |- ASVspoof_DF_cm_protocols
    %      |- ASVspoof2021.DF.cm.eval.trl.txt
  
    '''

    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=14)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.000001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--loss', type=str, default='weighted_CCE')
    # model
    parser.add_argument('--seed', type=int, default=4229,
                        help='random seed (default: 1234)')
    
    parser.add_argument('--model_path', type=str,
                        default=None, help='Model checkpoint')
    parser.add_argument('--comment', type=str, default=None,
                        help='Comment to describe the saved model')
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


    ##===================================================Rawboost data augmentation ======================================================================#

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
    
    ##===================================================Rawboost data augmentation ======================================================================#
    

    if not os.path.exists('models'):
        os.mkdir('models')
    args = parser.parse_args()
 
    #make experiment reproducible
    set_random_seed(args.seed, args)
    
    track = args.track

    #define model saving path
    model_tag = 'model_{}_{}_{}_{}_{}_bs{}'.format(
        track, args.loss, args.num_epochs, args.batch_size, args.lr, args.batch_size)
    if args.comment:
        model_tag = model_tag + '_{}'.format(args.comment)
    model_save_path = os.path.join('models', model_tag)

    #set model save directory
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
    
    # Create CSV file for tracking metrics
    csv_filename = os.path.join(model_save_path, 'training_metrics.csv')
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Epoch', 'Train_Loss', 'Val_Loss', 'Val_Acc', 'EER', 'Learning_Rate', 'Timestamp'])
    
    #GPU device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'                  
    print('Device: {}'.format(device))
    
    model = Model(args,device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])

    model =nn.DataParallel(model).to(device)
    print('nb_params:',nb_params)

    #set Adam optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=0.01,  # 增加权重衰减
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # 改进的学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-7
    )
    
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path,map_location=device))
        print('Model loaded : {}'.format(args.model_path))

    # evaluation mode on the In-the-Wild dataset.
    if args.track == 'In-the-Wild':
        file_eval = genSpoof_list( dir_meta =  os.path.join(args.protocols_path),is_train=False,is_eval=True)
        print('no. of eval trials',len(file_eval))
        eval_set=Dataset_in_the_wild_eval(list_IDs = file_eval,base_dir = os.path.join(args.database_path))
        produce_evaluation_file(eval_set, model, device, args.eval_output)
        sys.exit(0)

    # evaluation mode on the DF or LA dataset.
    if args.eval:
        file_eval = genSpoof_list(dir_meta =  os.path.join(args.protocols_path),is_train=False,is_eval=True)
        print('no. of eval trials',len(file_eval))
        eval_set=Dataset_ASVspoof2021_eval(list_IDs = file_eval,base_dir = os.path.join(args.database_path))
        produce_evaluation_file(eval_set, model, device, args.eval_output)
        sys.exit(0)
   
    

     
    # define train dataloader
    d_label_trn,file_train = genSpoof_list( dir_meta =  os.path.join(args.protocols_path+'ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'),is_train=True,is_eval=False)
    
    print('no. of training trials',len(file_train))
    
    train_set=Dataset_ASVspoof2019_train(args,list_IDs = file_train,labels = d_label_trn,base_dir = os.path.join(args.database_path+'ASVspoof2019_LA_train/'),algo=args.algo)
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size,num_workers=8, shuffle=True,drop_last = True)
    
    del train_set,d_label_trn
    

    # #define dev (validation) dataloader
    #
    d_label_dev,file_dev = genSpoof_list( dir_meta =  os.path.join(args.protocols_path+'ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'),is_train=False,is_eval=False)
    
    print('no. of validation trials',len(file_dev))
    
    dev_set = Dataset_ASVspoof2019_train(args,list_IDs = file_dev,labels = d_label_dev,base_dir = os.path.join(args.database_path+'ASVspoof2019_LA_dev/'),algo=args.algo)
    
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size,num_workers=8, shuffle=False)
    
    del dev_set,d_label_dev

    
    

    # Training and validation 
    num_epochs = args.num_epochs
    writer = SummaryWriter('logs/{}'.format(model_tag))

    best_val_loss = float('inf')
    best_eer = float('inf')  # Add this
    patience_counter = 0
    
    # Track training history
    training_history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'val_acc': [],
        'eer': [],
        'lr': []
    }

    for epoch in range(num_epochs):
        
        running_loss = train_epoch(train_loader,model, args.lr,optimizer, device)
        val_loss, val_acc, eer = evaluate_accuracy(dev_loader, model, device)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        scheduler.step(val_loss)
        
        # Save metrics to CSV
        with open(csv_filename, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([
                epoch + 1,  # Epoch starts from 1
                f'{running_loss:.6f}',
                f'{val_loss:.6f}',
                f'{val_acc:.2f}',
                f'{eer:.2f}',
                f'{current_lr:.8f}',
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ])
        
        # Store in history dictionary
        training_history['epoch'].append(epoch + 1)
        training_history['train_loss'].append(running_loss)
        training_history['val_loss'].append(val_loss)
        training_history['val_acc'].append(val_acc)
        training_history['eer'].append(eer)
        training_history['lr'].append(current_lr)
        
        # Save best model based on EER
        if eer < best_eer:
            best_eer = eer
            torch.save(model.state_dict(), os.path.join(
                    model_save_path, 'best_model_eer.pth'))
            print(f"New best EER: {eer:.2f}%")
            
            # Save best metrics to a separate file
            with open(os.path.join(model_save_path, 'best_metrics.txt'), 'w') as f:
                f.write(f"Best EER: {eer:.2f}% at Epoch {epoch + 1}\n")
                f.write(f"Val Loss: {val_loss:.6f}\n")
                f.write(f"Val Acc: {val_acc:.2f}%\n")
                f.write(f"Train Loss: {running_loss:.6f}\n")
                f.write(f"Learning Rate: {current_lr:.8f}\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(
                    model_save_path, 'best_model_val_loss.pth'))
        else:
            patience_counter += 1

        # Log metrics to tensorboard
        writer.add_scalar('train_loss', running_loss, epoch)
        writer.add_scalar('val_loss', val_loss, epoch)
        writer.add_scalar('val_acc', val_acc, epoch)
        writer.add_scalar('val_eer', eer, epoch)
        writer.add_scalar('learning_rate', current_lr, epoch)
        
        print('\nEpoch: {} - Train Loss: {:.4f} - Val Loss: {:.4f} - Val Acc: {:.2f}% - EER: {:.2f}% - LR: {:.8f}'.format(
            epoch + 1, running_loss, val_loss, val_acc, eer*100, current_lr))
        
        torch.save(model.state_dict(), os.path.join(
            model_save_path, 'epoch_{}.pth'.format(epoch)))

        if patience_counter >= 10:  # Increased patience for better convergence
            print(f"Early stopping triggered at epoch {epoch + 1}, best model is epoch: {epoch + 1 - patience_counter}")
            
            # Save final summary
            with open(os.path.join(model_save_path, 'training_summary.txt'), 'w') as f:
                f.write(f"Training Summary\n")
                f.write(f"================\n")
                f.write(f"Total Epochs: {epoch + 1}\n")
                f.write(f"Best EER: {best_eer:.2f}%\n")
                f.write(f"Best Val Loss: {best_val_loss:.6f}\n")
                f.write(f"Early Stopped at Epoch: {epoch + 1}\n")
                f.write(f"Batch Size: {args.batch_size}\n")
                f.write(f"Initial LR: {args.lr}\n")
                f.write(f"Final LR: {current_lr:.8f}\n")
                f.write(f"Model Path: {model_save_path}\n")
            break
    
    # Save complete training history as numpy arrays for later analysis
    np.savez(os.path.join(model_save_path, 'training_history.npz'), **training_history)
    
    writer.close()
    print(f"\nTraining complete! Metrics saved to: {csv_filename}")
    print(f"Best EER: {best_eer:.2f}%")
    print(f"Model saved in: {model_save_path}")
    
# CUDA_VISIBLE_DEVICES=0 python main.py --track=DF --lr=0.000001 --batch_size=20 --loss=weighted_CCE --num_epochs=100
# CUDA_VISIBLE_DEVICES=0 python main.py \
#   --track=LA \
#   --is_eval \
#   --eval \
#   --model_path=/root/autodl-tmp/SLSforASVspoof-2021-DF/models/Ablation/best_model_eer_weighted_CCE_16_1e-06_group3.pth \
#   --protocols_path=/root/autodl-tmp/CLAD/Datasets/ASVspoof2021_LA_eval/ASVspoof2021.LA.cm.eval.trl.txt \
#   --database_path=/root/autodl-tmp/CLAD/Datasets/ASVspoof2021_LA_eval/ \
#   --eval_output=/root/autodl-tmp/SLSforASVspoof-2021-DF/scores_best_model_eer_weighted_CCE_16_1e-06_group3.pth.txt