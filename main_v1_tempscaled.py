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
from model_hierarchical_contrastive import Model  # Updated import
from tensorboardX import SummaryWriter
from core_scripts.startup_config import set_random_seed
from tqdm import tqdm
from torchvision import transforms
from sklearn.metrics import roc_curve

def init_csv_log(log_path):
    """Initialize CSV log file with headers"""
    with open(log_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'epoch', 'timestamp', 'train_loss', 'contrastive_loss', 'total_loss', 
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
            log_data['contrastive_loss'],
            log_data['total_loss'],
            log_data['val_loss'],
            log_data['val_acc'],
            log_data['val_eer']
        ])

def compute_eer(y_true, y_score):
    """
    Compute Equal Error Rate (EER)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    return eer * 100, eer_threshold

def evaluate_accuracy(dev_loader, model, device):
    val_loss = 0.0
    num_correct = 0.0
    num_total = 0.0
    model.eval()
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    
    all_labels = []
    all_scores = []
    
    for batch_x, batch_y in tqdm(dev_loader):
        
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x)
        _, batch_pred = batch_out.max(dim=1)
        num_correct += (batch_pred == batch_y).sum(dim=0).item()

        batch_loss = criterion(batch_out, batch_y)
        val_loss += (batch_loss.item() * batch_size)
        
        # Collect scores and labels for EER calculation
        batch_scores = torch.softmax(batch_out, dim=1)[:, 1].detach().cpu().numpy()
        all_scores.extend(batch_scores)
        all_labels.extend(batch_y.cpu().numpy())

    val_loss /= num_total
    acc = 100 * (num_correct / num_total)
    
    # Calculate EER
    eer, eer_threshold = compute_eer(all_labels, all_scores)
    
    return val_loss, acc, eer

def produce_evaluation_file(dataset, model, device, save_path):
    data_loader = DataLoader(
        dataset, 
        batch_size=16,  # Increase from default (probably 1)
        shuffle=False, 
        drop_last=False, 
        pin_memory=True,  # Faster GPU transfer
        num_workers=8     # Parallel data loading
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

def train_epoch(train_loader, model, lr, optim, device, contrastive_weight=0.1):
    running_loss = 0
    running_contrastive_loss = 0
    running_total_loss = 0
    
    num_total = 0.0
    
    model.train()

    #set objective (Loss) functions
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    
    for batch_x, batch_y in tqdm(train_loader):
       
        batch_size = batch_x.size(0)
        num_total += batch_size
        
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        
        # Forward pass with contrastive learning
        batch_out, contrastive_loss = model(batch_x, labels=batch_y, return_contrastive=True)
        
        # Classification loss
        classification_loss = criterion(batch_out, batch_y)
        
        # Total loss with contrastive component
        total_loss = classification_loss + contrastive_weight * contrastive_loss
        
        running_loss += (classification_loss.item() * batch_size)
        running_contrastive_loss += (contrastive_loss.item() * batch_size)
        running_total_loss += (total_loss.item() * batch_size)
       
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
       
    running_loss /= num_total
    running_contrastive_loss /= num_total
    running_total_loss /= num_total

    return running_loss, running_contrastive_loss, running_total_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ASVspoof2021 baseline system with hierarchical contrastive learning')
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
    
    # Contrastive learning parameters
    parser.add_argument('--contrastive_weight', type=float, default=0.1, 
                        help='Weight for contrastive loss component')
    parser.add_argument('--contrastive_temperature', type=float, default=0.3,
                        help='Temperature for contrastive loss')
    parser.add_argument('--group_size', type=int, default=3,
                        help='Group size for hierarchical attention')
    
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
    model_tag = 'model_hierarchical_contrastive_{}_{}_{}_{}_{}_cw{}_temp{}'.format(
        track, args.loss, args.num_epochs, args.batch_size, args.lr, 
        args.contrastive_weight, args.contrastive_temperature)
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
    
    model = Model(args, device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])

    model = nn.DataParallel(model).to(device)
    print('nb_params:', nb_params)

    #set Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
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
    d_label_dev,file_dev = genSpoof_list( dir_meta =  os.path.join(args.protocols_path+'ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'),is_train=False,is_eval=False)
    
    print('no. of validation trials',len(file_dev))
    
    dev_set = Dataset_ASVspoof2019_train(args,list_IDs = file_dev,labels = d_label_dev,base_dir = os.path.join(args.database_path+'ASVspoof2019_LA_dev/'),algo=args.algo)
    
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size,num_workers=8, shuffle=False)
    
    del dev_set,d_label_dev

    # Training and validation 
    num_epochs = args.num_epochs
    writer = SummaryWriter('logs/{}'.format(model_tag))

    best_val_loss = float('inf')
    best_eer = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        print('\nEpoch: {}'.format(epoch))
        
        running_loss, running_contrastive_loss, running_total_loss = train_epoch(
            train_loader, model, args.lr, optimizer, device, args.contrastive_weight)
        
        if epoch >= 0:
            val_loss, val_acc, val_eer = evaluate_accuracy(dev_loader, model, device)

            # Log to CSV
            log_data = {
                'epoch': epoch,
                'timestamp': datetime.now().strftime('%d/%m/%Y %H:%M'),
                'train_loss': running_loss,
                'contrastive_loss': running_contrastive_loss,
                'total_loss': running_total_loss,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_eer': val_eer
            }
            log_training_metrics(log_data, model_save_path)

            if val_eer < best_eer:
                best_eer = val_eer
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model based on EER
                if args.comment:
                    best_model_filename = f'best_model_eer_{args.comment}.pth'
                else:
                    best_model_filename = 'best_model_eer.pth'
                torch.save(model.state_dict(), os.path.join(model_save_path, best_model_filename))
                print(f'New best model saved with EER: {best_eer:.6f}')
            else:
                patience_counter += 1

            writer.add_scalar('val_loss', val_loss, epoch)
            writer.add_scalar('val_acc', val_acc, epoch)
            writer.add_scalar('val_eer', val_eer, epoch)
            writer.add_scalar('train_loss', running_loss, epoch)
            writer.add_scalar('contrastive_loss', running_contrastive_loss, epoch)
            writer.add_scalar('total_loss', running_total_loss, epoch)
            
            print('Train Loss: {:.6f} - Contrastive Loss: {:.6f} - Total Loss: {:.6f} - Val Loss: {:.6f} - Val Acc: {:.2f}% - Val EER: {:.6f}'.format(
                running_loss, running_contrastive_loss, running_total_loss, val_loss, val_acc, val_eer))
            
            # Save epoch model with comment
            if args.comment:
                epoch_model_filename = f'epoch_{epoch}_{args.comment}.pth'
            else:
                epoch_model_filename = f'epoch_{epoch}.pth'
            torch.save(model.state_dict(), os.path.join(model_save_path, epoch_model_filename))

            # if patience_counter >= 3:  # Increased patience for better training
            #     print("Early stopping triggered, best EER: {:.2f}% at epoch: {}".format(best_eer*100, epoch - patience_counter))
            #     break
    
    print("Training completed. Best EER: {:.6f}".format(best_eer))
    writer.close()