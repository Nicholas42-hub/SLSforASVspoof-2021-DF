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
from model_hierarchical_contrastive import ModelHierarchicalContrastive, CombinedLoss
from tensorboardX import SummaryWriter
from core_scripts.startup_config import set_random_seed
from tqdm import tqdm
from torchvision import transforms

def init_csv_log(log_path):
    """Initialize CSV log file with headers"""
    with open(log_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'epoch', 'timestamp', 'train_loss', 'train_cce', 'train_contrastive',
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
            log_data['train_loss'],
            log_data['train_cce'],
            log_data['train_contrastive'],
            log_data['val_loss'],
            log_data['val_cce'],
            log_data['val_contrastive'],
            log_data['val_acc'],
            log_data['val_eer']
        ])

def evaluate_accuracy(dev_loader, model, device, criterion):
    val_loss = 0.0
    val_cce_loss = 0.0
    val_contrastive_loss = 0.0
    num_correct = 0.0
    num_total = 0.0
    model.eval()
    
    for batch_x, batch_y in tqdm(dev_loader):
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        
        with torch.no_grad():
            batch_out, features = model(batch_x, return_features=True)
            _, batch_pred = batch_out.max(dim=1)
            num_correct += (batch_pred == batch_y).sum(dim=0).item()
            
            batch_loss, cce_loss, contrastive_loss = criterion(batch_out, features, batch_y)
            val_loss += (batch_loss.item() * batch_size)
            val_cce_loss += (cce_loss.item() * batch_size)
            val_contrastive_loss += (contrastive_loss.item() * batch_size)

    val_loss /= num_total
    val_cce_loss /= num_total
    val_contrastive_loss /= num_total
    acc = 100 * (num_correct / num_total)
    
    return val_loss, val_cce_loss, val_contrastive_loss, acc

def produce_evaluation_file(dataset, model, device, save_path):
    data_loader = DataLoader(dataset, batch_size=8, shuffle=False, drop_last=False)
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

def train_epoch(train_loader, model, lr, optim, device, criterion):
    running_loss = 0
    running_cce_loss = 0
    running_contrastive_loss = 0
    num_total = 0.0
    
    model.train()
    
    for batch_x, batch_y in tqdm(train_loader):
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        
        # Forward pass - get both logits and features
        batch_out, features = model(batch_x, return_features=True)
        
        # Compute combined loss
        batch_loss, cce_loss, contrastive_loss = criterion(batch_out, features, batch_y)
        
        running_loss += (batch_loss.item() * batch_size)
        running_cce_loss += (cce_loss.item() * batch_size)
        running_contrastive_loss += (contrastive_loss.item() * batch_size)
        
        # Backward pass
        optim.zero_grad()
        batch_loss.backward()
        optim.step()
    
    running_loss /= num_total
    running_cce_loss /= num_total
    running_contrastive_loss /= num_total
    
    return running_loss, running_cce_loss, running_contrastive_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ASVspoof2021 Hierarchical Attention Contrastive Learning System')
    
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
    parser.add_argument('--contrastive_weight', type=float, default=0.1,
                        help='Weight for contrastive loss')
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='Temperature for contrastive loss')
    
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

    #define model saving path
    model_tag = 'model_hierarchical_contrastive_{}_{}_{}_{}_{}'.format(
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
    
    model = ModelHierarchicalContrastive(args, device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])

    model = nn.DataParallel(model).to(device)
    print('nb_params:', nb_params)

    # Set combined loss function
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = CombinedLoss(
        weight=weight, 
        temperature=args.temperature,
        contrastive_weight=args.contrastive_weight
    )

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
    d_label_trn, file_train = genSpoof_list(dir_meta=os.path.join(args.protocols_path+'ASVspoof_DF_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'), is_train=True, is_eval=False)
    
    print('no. of training trials', len(file_train))
    
    train_set = Dataset_ASVspoof2019_train(args, list_IDs=file_train, labels=d_label_trn, base_dir=os.path.join(args.database_path+'ASVspoof2019_LA_train/'), algo=args.algo)
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=8, shuffle=True, drop_last=True)
    
    del train_set, d_label_trn
    
    # define dev (validation) dataloader
    d_label_dev, file_dev = genSpoof_list(dir_meta=os.path.join(args.protocols_path+'ASVspoof_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'), is_train=False, is_eval=False)

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
        train_loss, train_cce, train_contrastive = train_epoch(
            train_loader, model, args.lr, optimizer, device, criterion
        )
        
        # Validation  
        val_loss, val_cce, val_contrastive, val_acc = evaluate_accuracy(
            dev_loader, model, device, criterion
        )
        
        print(f'Train Loss: {train_loss:.6f} (CCE: {train_cce:.6f}, Contrastive: {train_contrastive:.6f})')
        print(f'Val Loss: {val_loss:.6f} (CCE: {val_cce:.6f}, Contrastive: {val_contrastive:.6f})')
        print(f'Val Acc: {val_acc:.2f}%')
        
        # For now, use val_loss as EER placeholder (you can implement actual EER calculation)
        val_eer = val_loss  # Replace with actual EER calculation
        
        # Log to CSV
        log_data = {
            'epoch': epoch,
            'timestamp': datetime.now().strftime('%d/%m/%Y %H:%M'),
            'train_loss': train_loss,
            'train_cce': train_cce,
            'train_contrastive': train_contrastive,
            'val_loss': val_loss,
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
        writer.add_scalar('val/total_loss', val_loss, epoch)
        writer.add_scalar('val/cce_loss', val_cce, epoch)
        writer.add_scalar('val/contrastive_loss', val_contrastive, epoch)
        writer.add_scalar('val/accuracy', val_acc, epoch)
        
        # Save model checkpoint
        if args.comment:
            model_filename = f'epoch_{epoch}_{args.comment}.pth'
        else:
            model_filename = f'epoch_{epoch}.pth'
            
        torch.save(model.state_dict(), os.path.join(model_save_path, model_filename))
        
        # Save best model based on EER
        if val_eer < best_val_eer:
            best_val_eer = val_eer
            patience_counter = 0
            if args.comment:
                best_model_filename = f'best_model_eer_{args.comment}.pth'
            else:
                best_model_filename = 'best_model_eer.pth'
            torch.save(model.state_dict(), os.path.join(model_save_path, best_model_filename))
            print(f'New best model saved with EER: {best_val_eer:.6f}')
        else:
            patience_counter += 1

        # if patience_counter >= 10:  # Early stopping patience
        #     print(f"Early stopping triggered at epoch {epoch}, best EER: {best_val_eer:.6f}")
        #     break

    writer.close()
    print('Training completed!')