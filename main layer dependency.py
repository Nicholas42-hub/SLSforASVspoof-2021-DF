import argparse
import sys
import os
import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
import yaml
from data_utils_SSL import genSpoof_list,Dataset_ASVspoof2019_train,Dataset_ASVspoof2021_eval,Dataset_in_the_wild_eval
from model import Model
from tensorboardX import SummaryWriter
from core_scripts.startup_config import set_random_seed
from tqdm import tqdm
from torchvision import transforms
from sklearn.metrics import roc_curve
import logging
import csv
import pandas as pd
from datetime import datetime

def setup_logger(log_file):
    """设置日志记录器"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def setup_csv_logger(csv_file):
    """设置CSV日志记录器"""
    # 创建CSV文件并写入表头
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'epoch', 'timestamp',
            'train_loss', 'train_reg_loss', 'train_total_loss', 'train_acc', 'train_eer',
            'val_loss', 'val_reg_loss', 'val_total_loss', 'val_acc', 'val_eer',
            'is_best_model', 'patience_counter'
        ])
    
    def log_metrics(epoch, train_metrics, val_metrics, is_best, patience):
        """记录一个epoch的指标到CSV"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,  # epoch从1开始计数
                timestamp,
                train_metrics['loss'],
                train_metrics['reg_loss'], 
                train_metrics['total_loss'],
                train_metrics['acc'],
                train_metrics['eer'],
                val_metrics['loss'],
                val_metrics['reg_loss'],
                val_metrics['total_loss'], 
                val_metrics['acc'],
                val_metrics['eer'],
                is_best,
                patience
            ])
    
    return log_metrics

def compute_eer(scores, labels):
    """计算等错误率 (Equal Error Rate)"""
    # scores: 真实度分数 (higher = more likely to be genuine)
    # labels: 真实标签 (0 = spoof, 1 = genuine)
    
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    
    # 找到FPR和FNR最接近的点
    eer_idx = np.nanargmin(np.absolute((fnr - fpr)))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    eer_threshold = thresholds[eer_idx]
    
    return eer * 100, eer_threshold  # 返回百分比形式的EER

def evaluate_accuracy(dev_loader, model, device, logger=None):
    """评估模型准确率、损失和EER"""
    val_loss = 0.0
    val_reg_loss = 0.0
    num_correct = 0.0
    num_total = 0.0
    model.eval()
    
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in tqdm(dev_loader, desc="Validating"):
            batch_size = batch_x.size(0)
            num_total += batch_size
            batch_x = batch_x.to(device)
            batch_y = batch_y.view(-1).type(torch.int64).to(device)
            
            # 处理新模型输出格式
            model_output = model(batch_x)
            if isinstance(model_output, dict):
                batch_out = model_output['output']
                reg_loss = model_output['reg_loss']
            else:
                batch_out = model_output
                reg_loss = 0
            
            _, batch_pred = batch_out.max(dim=1)
            num_correct += (batch_pred == batch_y).sum(dim=0).item()

            # 计算损失
            classification_loss = criterion(batch_out, batch_y)
            val_loss += (classification_loss.item() * batch_size)
            val_reg_loss += (reg_loss.item() * batch_size if isinstance(reg_loss, torch.Tensor) else reg_loss * batch_size)
            
            # 收集分数和标签用于EER计算
            batch_scores = torch.softmax(batch_out, dim=1)[:, 1].cpu().numpy()  # 真实度分数
            all_scores.extend(batch_scores)
            all_labels.extend(batch_y.cpu().numpy())

    val_loss /= num_total
    val_reg_loss /= num_total
    acc = 100 * (num_correct / num_total)
    
    # 计算EER
    eer, eer_threshold = compute_eer(np.array(all_scores), np.array(all_labels))
    
    if logger:
        logger.info(f'Validation - Loss: {val_loss:.6f}, Reg Loss: {val_reg_loss:.6f}, Acc: {acc:.2f}%, EER: {eer:.2f}%')
    
    return val_loss, val_reg_loss, acc, eer, eer_threshold

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
    
    for batch_x, utt_id in tqdm(data_loader):
        fname_list = []
        score_list = []  
        batch_size = batch_x.size(0)
        batch_x = batch_x.to(device)
        torch.set_printoptions(threshold=10_000)
        
        # 处理新模型输出格式
        model_output = model(batch_x)
        if isinstance(model_output, dict):
            batch_out = model_output['output']
        else:
            batch_out = model_output
            
        batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel() 
        
        # add outputs
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())
        
        with open(save_path, 'a+') as fh:
            for f, cm in zip(fname_list, score_list):
                fh.write('{} {}\n'.format(f, cm))
        fh.close()   
    print('Scores saved to {}'.format(save_path))

def train_epoch(train_loader, model, lr, optimizer, device, logger=None):
    """训练一个epoch并返回详细指标"""
    running_loss = 0
    running_reg_loss = 0
    num_total = 0.0
    num_correct = 0.0
    
    model.train()

    # set objective (Loss) functions
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    
    all_scores = []
    all_labels = []
    
    for batch_x, batch_y in tqdm(train_loader, desc="Training"):
        batch_size = batch_x.size(0)
        num_total += batch_size
        
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        
        # 处理新模型输出格式
        model_output = model(batch_x)
        if isinstance(model_output, dict):
            batch_out = model_output['output']
            reg_loss = model_output['reg_loss']
        else:
            batch_out = model_output
            reg_loss = 0
        
        # 计算准确率
        _, batch_pred = batch_out.max(dim=1)
        num_correct += (batch_pred == batch_y).sum(dim=0).item()
        
        # 计算总损失（分类损失 + 正则化损失）
        classification_loss = criterion(batch_out, batch_y)
        total_loss = classification_loss + reg_loss
        
        running_loss += (classification_loss.item() * batch_size)
        running_reg_loss += (reg_loss.item() * batch_size if isinstance(reg_loss, torch.Tensor) else reg_loss * batch_size)
       
        # 收集分数和标签用于EER计算
        batch_scores = torch.softmax(batch_out, dim=1)[:, 1].detach().cpu().numpy()
        all_scores.extend(batch_scores)
        all_labels.extend(batch_y.cpu().numpy())
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
       
    running_loss /= num_total
    running_reg_loss /= num_total
    train_acc = 100 * (num_correct / num_total)
    
    # 计算训练EER
    train_eer, _ = compute_eer(np.array(all_scores), np.array(all_labels))
    
    if logger:
        logger.info(f'Training - Loss: {running_loss:.6f}, Reg Loss: {running_reg_loss:.6f}, Acc: {train_acc:.2f}%, EER: {train_eer:.2f}%')

    return running_loss, running_reg_loss, train_acc, train_eer

def save_final_results_csv(csv_file, model_save_path, model_tag, nb_params, args, best_val_eer, best_epoch, total_epochs):
    """保存最终训练结果的CSV摘要"""
    results_file = os.path.join(model_save_path, 'training_results.csv')
    
    with open(results_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # 写入表头
        writer.writerow(['metric', 'value'])
        
        # 写入训练结果
        writer.writerow(['model_name', model_tag])
        writer.writerow(['total_epochs', total_epochs])
        writer.writerow(['best_epoch', best_epoch + 1])
        writer.writerow(['best_val_eer', f'{best_val_eer:.4f}'])
        writer.writerow(['num_parameters', nb_params])
        writer.writerow(['num_groups', args.num_groups])
        writer.writerow(['batch_size', args.batch_size])
        writer.writerow(['learning_rate', args.lr])
        writer.writerow(['reg_weight', args.reg_weight])
        writer.writerow(['track', args.track])
        writer.writerow(['loss_function', args.loss])
        writer.writerow(['training_completed', datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
    
    return results_file

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ASVspoof2021 baseline system')
    # Dataset
    parser.add_argument('--database_path', type=str, default='/root/autodl-tmp/CLAD/Datasets/LA/', help='Change this to user\'s full directory address of LA database (ASVspoof2019- for training & development (used as validation), ASVspoof2021 DF for evaluation scores). We assume that all three ASVspoof 2019 LA train, LA dev and ASVspoof2021 DF eval data folders are in the same database_path directory.')
    parser.add_argument('--protocols_path', type=str, default='/root/autodl-tmp/CLAD/Datasets/LA/', help='Change with path to user\'s DF database protocols directory address')

    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=14)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.000001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--loss', type=str, default='weighted_CCE')
    
    # 新增：自适应分组相关参数
    parser.add_argument('--num_groups', type=int, default=4, 
                        help='Number of adaptive groups for layer dependency modeling')
    parser.add_argument('--reg_weight', type=float, default=1.0,
                        help='Weight for regularization loss')
    
    # model
    parser.add_argument('--seed', type=int, default=1234,
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
    
    ##===================================================Rawboost data augmentation ======================================================================#
    

    if not os.path.exists('models'):
        os.mkdir('models')
    args = parser.parse_args()
 
    #make experiment reproducible
    set_random_seed(args.seed, args)
    
    track = args.track

    #define model saving path
    model_tag = 'model_adaptive_{}_{}_{}_{}_{}_{}'.format(
        track, args.loss, args.num_epochs, args.batch_size, args.lr, args.num_groups)
    if args.comment:
        model_tag = model_tag + '_{}'.format(args.comment)
    model_save_path = os.path.join('models', model_tag)

    #set model save directory
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
    
    # 设置日志文件
    log_file = os.path.join(model_save_path, 'training.log')
    csv_file = os.path.join(model_save_path, 'training_metrics.csv')
    
    logger = setup_logger(log_file)
    csv_logger = setup_csv_logger(csv_file)  # 新增CSV日志记录器
    
    logger.info("="*50)
    logger.info("TRAINING CONFIGURATION")
    logger.info("="*50)
    logger.info(f"Model: {model_tag}")
    logger.info(f"Track: {track}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"Number of epochs: {args.num_epochs}")
    logger.info(f"Number of groups: {args.num_groups}")
    logger.info(f"Regularization weight: {args.reg_weight}")
    logger.info(f"Database path: {args.database_path}")
    logger.info(f"Protocols path: {args.protocols_path}")
    logger.info(f"Training log: {log_file}")
    logger.info(f"Training CSV: {csv_file}")
    logger.info("="*50)
    
    #GPU device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'                  
    logger.info('Device: {}'.format(device))
    
    model = Model(args, device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])

    model = nn.DataParallel(model).to(device)
    logger.info('Number of parameters: {:,}'.format(nb_params))
    logger.info('Number of groups: {}'.format(args.num_groups))

    #set Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        logger.info('Model loaded: {}'.format(args.model_path))

    # evaluation mode on the In-the-Wild dataset.
    if args.track == 'In-the-Wild':
        file_eval = genSpoof_list(dir_meta=os.path.join(args.protocols_path), is_train=False, is_eval=True)
        logger.info('Number of eval trials: {}'.format(len(file_eval)))
        eval_set = Dataset_in_the_wild_eval(list_IDs=file_eval, base_dir=os.path.join(args.database_path))
        produce_evaluation_file(eval_set, model, device, args.eval_output)
        sys.exit(0)

    # evaluation mode on the DF or LA dataset.
    if args.eval:
        file_eval = genSpoof_list(dir_meta=os.path.join(args.protocols_path), is_train=False, is_eval=True)
        logger.info('Number of eval trials: {}'.format(len(file_eval)))
        eval_set = Dataset_ASVspoof2021_eval(list_IDs=file_eval, base_dir=os.path.join(args.database_path))
        produce_evaluation_file(eval_set, model, device, args.eval_output)
        sys.exit(0)
   
    # define train dataloader
    d_label_trn, file_train = genSpoof_list(dir_meta=os.path.join(args.protocols_path+'ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'), is_train=True, is_eval=False)
    
    logger.info('Number of training trials: {}'.format(len(file_train)))
    
    train_set = Dataset_ASVspoof2019_train(args, list_IDs=file_train, labels=d_label_trn, base_dir=os.path.join(args.database_path+'ASVspoof2019_LA_train/'), algo=args.algo)
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=8, shuffle=True, drop_last=True)
    
    del train_set, d_label_trn
    
    # define dev (validation) dataloader
    d_label_dev, file_dev = genSpoof_list(dir_meta=os.path.join(args.protocols_path+'ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'), is_train=False, is_eval=False)

    logger.info('Number of validation trials: {}'.format(len(file_dev)))

    dev_set = Dataset_ASVspoof2019_train(args, list_IDs=file_dev, labels=d_label_dev, base_dir=os.path.join(args.database_path+'ASVspoof2019_LA_dev/'), algo=args.algo)

    dev_loader = DataLoader(dev_set, batch_size=args.batch_size, num_workers=8, shuffle=False)

    del dev_set, d_label_dev
    
    # Training and validation 
    num_epochs = args.num_epochs
    writer = SummaryWriter('logs/{}'.format(model_tag))

    best_val_eer = float('inf')
    best_epoch = 0
    patience_counter = 0
    patience_limit = 10

    logger.info("="*50)
    logger.info("STARTING TRAINING")
    logger.info("="*50)

    for epoch in range(num_epochs):
        logger.info(f"\n--- EPOCH {epoch+1}/{num_epochs} ---")
        
        # 训练
        train_loss, train_reg_loss, train_acc, train_eer = train_epoch(
            train_loader, model, args.lr, optimizer, device, logger)
        
        # 验证
        val_loss, val_reg_loss, val_acc, val_eer, val_threshold = evaluate_accuracy(
            dev_loader, model, device, logger)

        total_train_loss = train_loss + train_reg_loss * args.reg_weight
        total_val_loss = val_loss + val_reg_loss * args.reg_weight
        
        # 记录到TensorBoard
        writer.add_scalar('Loss/Train_Classification', train_loss, epoch)
        writer.add_scalar('Loss/Train_Regularization', train_reg_loss, epoch)
        writer.add_scalar('Loss/Train_Total', total_train_loss, epoch)
        writer.add_scalar('Loss/Val_Classification', val_loss, epoch)
        writer.add_scalar('Loss/Val_Regularization', val_reg_loss, epoch)
        writer.add_scalar('Loss/Val_Total', total_val_loss, epoch)
        
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Val', val_acc, epoch)
        
        writer.add_scalar('EER/Train', train_eer, epoch)
        writer.add_scalar('EER/Val', val_eer, epoch)
        
        # 保存最佳模型
        is_best = False
        if val_eer < best_val_eer:
            best_val_eer = val_eer
            best_epoch = epoch
            patience_counter = 0
            is_best = True
            
            # 保存最佳模型
            torch.save(model.state_dict(), os.path.join(model_save_path, 'best_model.pth'))
            logger.info(f"*** NEW BEST MODEL SAVED - Val EER: {val_eer:.2f}% ***")
        else:
            patience_counter += 1
        
        # 记录到CSV
        train_metrics = {
            'loss': train_loss,
            'reg_loss': train_reg_loss,
            'total_loss': total_train_loss,
            'acc': train_acc,
            'eer': train_eer *100
        }
        
        val_metrics = {
            'loss': val_loss,
            'reg_loss': val_reg_loss,
            'total_loss': total_val_loss,
            'acc': val_acc,
            'eer': val_eer *100
        }
        
        csv_logger(epoch, train_metrics, val_metrics, is_best, patience_counter)
        
        # 记录总结信息
        logger.info(f"EPOCH {epoch+1} SUMMARY:")
        logger.info(f"  Train - Loss: {train_loss:.6f}, Reg: {train_reg_loss:.6f}, Total: {total_train_loss:.6f}, Acc: {train_acc:.2f}%, EER: {train_eer:.2f}%")
        logger.info(f"  Val   - Loss: {val_loss:.6f}, Reg: {val_reg_loss:.6f}, Total: {total_val_loss:.6f}, Acc: {val_acc:.2f}%, EER: {val_eer:.2f}%")
        logger.info(f"  Best Val EER: {best_val_eer:.2f}% (Epoch {best_epoch+1})")
        logger.info(f"  Patience: {patience_counter}/{patience_limit}")
        
        # 保存每个epoch的模型
        torch.save(model.state_dict(), os.path.join(model_save_path, f'epoch_{epoch}.pth'))

        # Early stopping
        if patience_counter >= patience_limit:
            logger.info(f"Early stopping triggered after {patience_limit} epochs without improvement")
            logger.info(f"Best model was at epoch {best_epoch+1} with Val EER: {best_val_eer:.2f}%")
            break
    
    logger.info("="*50)
    logger.info("TRAINING COMPLETED")
    logger.info("="*50)
    logger.info(f"Best Validation EER: {best_val_eer:.2f}% at Epoch {best_epoch+1}")
    
    # 保存最终结果CSV
    results_csv = save_final_results_csv(csv_file, model_save_path, model_tag, nb_params, args, best_val_eer, best_epoch, epoch+1)
    logger.info(f"Training results saved to {results_csv}")
    
    # 保存分组可视化信息
    logger.info("Saving grouping visualization data...")
    
    # 获取一个batch用于可视化
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        model.eval()
        with torch.no_grad():
            if hasattr(model.module, 'get_grouping_heatmap'):
                heatmap = model.module.get_grouping_heatmap(batch_x[:1])  # 只取一个样本
                # 保存热力图数据
                torch.save(heatmap, os.path.join(model_save_path, 'grouping_heatmap.pt'))
                logger.info(f"Grouping heatmap saved to {model_save_path}/grouping_heatmap.pt")
                logger.info(f"Heatmap shape: {heatmap.shape} (Batch, Layers, Groups)")
        break
    
    logger.info("="*50)
    logger.info("SAVED FILES:")
    logger.info(f"  - Training log: {log_file}")
    logger.info(f"  - Training metrics CSV: {csv_file}")
    logger.info(f"  - Training results CSV: {results_csv}")
    logger.info(f"  - Best model: {os.path.join(model_save_path, 'best_model.pth')}")
    logger.info(f"  - Grouping heatmap: {os.path.join(model_save_path, 'grouping_heatmap.pt')}")
    logger.info("="*50)
    
    writer.close()