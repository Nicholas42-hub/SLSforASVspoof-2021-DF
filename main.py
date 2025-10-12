import argparse
import sys
import os
import csv
from datetime import datetime
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from data_utils_SSL import genSpoof_list, Dataset_ASVspoof2019_train, Dataset_ASVspoof2021_eval
from model_hiercon import ModelHierarchicalContrastive, CompatibleCombinedLoss
from tensorboardX import SummaryWriter
from core_scripts.startup_config import set_random_seed
from tqdm import tqdm
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

def compute_eer(scores, labels):
    """计算EER"""
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


def visualize_interpretability(model, audio, label, save_path, device):
    """可视化模型的可解释性分析"""
    model.eval()
    audio = audio.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output, interp = model(audio, return_interpretability=True)
    
    prediction = torch.argmax(output, dim=1).item()
    confidence = torch.exp(output[0, prediction]).item()
    
    layer_imp = interp['layer_importance'][0].cpu().numpy()
    temporal_imp = interp['temporal_importance'][0].cpu().numpy()
    temporal_attn = interp['attention_weights']['temporal'][0].cpu().numpy()
    
    # 创建图表
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. 层级重要性柱状图
    ax1 = fig.add_subplot(gs[0, :2])
    bars = ax1.bar(range(len(layer_imp)), layer_imp, color='steelblue', alpha=0.7)
    ax1.axhline(y=layer_imp.mean(), color='r', linestyle='--', label='Average', linewidth=2)
    ax1.set_xlabel('Transformer Layer', fontsize=12)
    ax1.set_ylabel('Importance Score', fontsize=12)
    ax1.set_title('Layer-wise Contribution to Decision', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 高亮top-3层
    top3_idx = np.argsort(layer_imp)[-3:]
    for idx in top3_idx:
        bars[idx].set_color('coral')
        bars[idx].set_alpha(0.9)
    
    # 2. 时序重要性曲线
    ax2 = fig.add_subplot(gs[1, :2])
    ax2.plot(temporal_imp, linewidth=2, color='darkgreen')
    ax2.fill_between(range(len(temporal_imp)), temporal_imp, alpha=0.3, color='lightgreen')
    ax2.set_xlabel('Time Frame', fontsize=12)
    ax2.set_ylabel('Importance Score', fontsize=12)
    ax2.set_title('Temporal Importance Pattern', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 标记峰值区域
    peak_idx = np.argmax(temporal_imp)
    ax2.plot(peak_idx, temporal_imp[peak_idx], 'r*', markersize=15, label='Peak')
    ax2.legend()
    
    # 3. Temporal attention heatmap
    ax3 = fig.add_subplot(gs[2, :])
    im = ax3.imshow(temporal_attn, aspect='auto', cmap='viridis', interpolation='bilinear')
    ax3.set_xlabel('Time Frame', fontsize=12)
    ax3.set_ylabel('Layer', fontsize=12)
    ax3.set_title('Temporal Attention across All Layers', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax3, label='Attention Weight')
    
    # 4. 预测信息和文本解释
    ax4 = fig.add_subplot(gs[0:2, 2])
    ax4.axis('off')
    
    # 创建信息文本
    pred_label = 'FAKE' if prediction == 1 else 'REAL'
    true_label = 'FAKE' if label == 1 else 'REAL'
    correct = '✓' if prediction == label else '✗'
    
    info_text = f"{'='*35}\n"
    info_text += f"PREDICTION SUMMARY\n"
    info_text += f"{'='*35}\n\n"
    info_text += f"Prediction:  {pred_label} {correct}\n"
    info_text += f"Confidence:  {confidence:.1%}\n"
    info_text += f"Ground Truth: {true_label}\n\n"
    info_text += f"{'='*35}\n"
    info_text += f"INTERPRETATION\n"
    info_text += f"{'='*35}\n\n"
    info_text += interp['text_explanations'][0]
    
    # 设置文本样式
    bbox_props = dict(boxstyle='round,pad=1', facecolor='wheat', alpha=0.8, edgecolor='black', linewidth=2)
    ax4.text(0.05, 0.95, info_text, transform=ax4.transAxes,
             fontsize=10, verticalalignment='top', family='monospace',
             bbox=bbox_props)
    
    plt.suptitle(f'Interpretability Analysis - Sample Classification', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'✅ Visualization saved to {save_path}')


def analyze_batch_interpretability(model, data_loader, device, num_samples=10, save_dir='interpretability'):
    """批量分析并保存可解释性可视化"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    model.eval()
    
    # 收集真假样本的层重要性分布
    real_layer_importance = []
    fake_layer_importance = []
    
    sample_count = 0
    for batch_x, batch_y in data_loader:
        if sample_count >= num_samples:
            break
        
        batch_x = batch_x.to(device)
        batch_y = batch_y.cpu().numpy()
        
        with torch.no_grad():
            _, interp = model(batch_x, return_interpretability=True)
        
        layer_imp = interp['layer_importance'].cpu().numpy()
        
        for i in range(len(batch_y)):
            if sample_count >= num_samples:
                break
            
            # 保存单个样本的可视化
            save_path = os.path.join(save_dir, f'sample_{sample_count:03d}_label_{batch_y[i]}.png')
            visualize_interpretability(model, batch_x[i].cpu(), batch_y[i], save_path, device)
            
            # 收集统计数据
            if batch_y[i] == 0:  # Real
                real_layer_importance.append(layer_imp[i])
            else:  # Fake
                fake_layer_importance.append(layer_imp[i])
            
            sample_count += 1
    
    # 绘制对比分析图
    if len(real_layer_importance) > 0 and len(fake_layer_importance) > 0:
        plot_comparative_analysis(real_layer_importance, fake_layer_importance, save_dir)
    
    print(f'\n✅ Analyzed {sample_count} samples, saved to {save_dir}/')


def plot_comparative_analysis(real_importance, fake_importance, save_dir):
    """绘制真假样本的层重要性对比分析"""
    real_importance = np.array(real_importance)
    fake_importance = np.array(fake_importance)
    
    real_mean = real_importance.mean(axis=0)
    fake_mean = fake_importance.mean(axis=0)
    real_std = real_importance.std(axis=0)
    fake_std = fake_importance.std(axis=0)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 平均层重要性对比
    ax1 = axes[0, 0]
    x = np.arange(len(real_mean))
    width = 0.35
    ax1.bar(x - width/2, real_mean, width, label='Real', alpha=0.7, color='green')
    ax1.bar(x + width/2, fake_mean, width, label='Fake', alpha=0.7, color='red')
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Average Importance')
    ax1.set_title('Average Layer Importance: Real vs Fake')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 差异分析
    ax2 = axes[0, 1]
    diff = fake_mean - real_mean
    colors = ['red' if d > 0 else 'green' for d in diff]
    ax2.bar(x, diff, color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Importance Difference (Fake - Real)')
    ax2.set_title('Layer Importance Difference')
    ax2.grid(True, alpha=0.3)
    
    # 3. 分布热图
    ax3 = axes[1, 0]
    combined = np.vstack([real_mean, fake_mean])
    im = ax3.imshow(combined, aspect='auto', cmap='RdYlGn_r')
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(['Real', 'Fake'])
    ax3.set_xlabel('Layer')
    ax3.set_title('Layer Importance Heatmap')
    plt.colorbar(im, ax=ax3)
    
    # 4. Top discriminative layers
    ax4 = axes[1, 1]
    abs_diff = np.abs(diff)
    top_k = 5
    top_indices = np.argsort(abs_diff)[-top_k:][::-1]
    
    ax4.barh(range(top_k), abs_diff[top_indices], color='steelblue', alpha=0.7)
    ax4.set_yticks(range(top_k))
    ax4.set_yticklabels([f'Layer {i}' for i in top_indices])
    ax4.set_xlabel('Absolute Importance Difference')
    ax4.set_title(f'Top {top_k} Discriminative Layers')
    ax4.grid(True, alpha=0.3, axis='x')
    
    plt.suptitle('Comparative Layer Importance Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'comparative_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'✅ Comparative analysis saved to {save_path}')


def train_epoch(train_loader, model, optimizer, device, criterion):
    """训练一个epoch"""
    running_loss = 0
    num_total = 0.0
    all_scores = []
    all_labels = []
    
    model.train()
    
    for batch_x, batch_y in tqdm(train_loader, desc='Training'):
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        
        # Forward pass with contrastive learning
        batch_out, projected_features = model(batch_x, return_features=True)
        
        # 计算损失
        logits = torch.exp(batch_out)  # Convert from log_softmax
        batch_loss, cce_loss, contrastive_loss = criterion(logits, projected_features, batch_y)
        
        running_loss += (batch_loss.item() * batch_size)
        
        # 收集scores用于EER计算
        with torch.no_grad():
            batch_scores = logits[:, 1].cpu().numpy()
            batch_labels = batch_y.cpu().numpy()
            valid_mask = ~(np.isnan(batch_scores) | np.isinf(batch_scores))
            if np.any(valid_mask):
                all_scores.extend(batch_scores[valid_mask].tolist())
                all_labels.extend(batch_labels[valid_mask].tolist())
        
        # Backward pass
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
    
    running_loss /= num_total
    
    # 计算训练EER
    if len(all_scores) > 0 and len(all_labels) > 0:
        train_eer, _ = compute_eer(all_scores, all_labels)
    else:
        train_eer = 50.0
    
    return running_loss, train_eer


def evaluate_accuracy(dev_loader, model, device, criterion):
    """验证模型性能"""
    val_loss = 0.0
    num_correct = 0.0
    num_total = 0.0
    all_scores = []
    all_labels = []
    
    model.eval()
    
    for batch_x, batch_y in tqdm(dev_loader, desc='Validating'):
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        
        with torch.no_grad():
            batch_out, projected_features = model(batch_x, return_features=True)
            _, batch_pred = batch_out.max(dim=1)
            num_correct += (batch_pred == batch_y).sum(dim=0).item()
            
            # 计算损失
            logits = torch.exp(batch_out)
            batch_loss, _, _ = criterion(logits, projected_features, batch_y)
            val_loss += (batch_loss.item() * batch_size)
            
            # 收集scores用于EER
            batch_scores = logits[:, 1].cpu().numpy()
            batch_labels = batch_y.cpu().numpy()
            valid_mask = ~(np.isnan(batch_scores) | np.isinf(batch_scores))
            if np.any(valid_mask):
                all_scores.extend(batch_scores[valid_mask].tolist())
                all_labels.extend(batch_labels[valid_mask].tolist())
    
    val_loss /= num_total
    acc = 100 * (num_correct / num_total)
    
    # 计算EER
    if len(all_scores) > 0 and len(all_labels) > 0:
        eer, _ = compute_eer(all_scores, all_labels)
    else:
        eer = 50.0
    
    return val_loss, acc, eer


def produce_evaluation_file(dataset, model, device, save_path):
    """生成评估文件"""
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
    
    for batch_x, utt_id in tqdm(data_loader, desc='Evaluating'):
        batch_x = batch_x.to(device)
        
        with torch.no_grad():
            batch_out = model(batch_x)
            batch_probs = torch.exp(batch_out)
            batch_score = batch_probs[:, 1].data.cpu().numpy().ravel()
        
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())
        
        with open(save_path, 'a+') as fh:
            for f, cm in zip(fname_list, score_list):
                fh.write('{} {}\n'.format(f, cm))
        
        fname_list = []
        score_list = []
    
    print(f'✅ Evaluation scores saved to {save_path}')


def init_csv_log(log_path):
    """初始化CSV日志"""
    with open(log_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'epoch', 'timestamp', 'train_loss', 'train_eer',
            'val_loss', 'val_acc', 'val_eer'
        ])


def log_training_metrics(log_data, model_save_path):
    """记录训练指标到CSV"""
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ASVspoof Audio Deepfake Detection with Hierarchical Attention')
    
    # Dataset paths
    parser.add_argument('--database_path', type=str, 
                        default='/root/autodl-tmp/CLAD/Datasets/LA/',
                        help='Path to database')
    parser.add_argument('--protocols_path', type=str,
                        default='/root/autodl-tmp/CLAD/Datasets/LA/',
                        help='Path to protocols')
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=14)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.000001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    
    # Model parameters
    parser.add_argument('--group_size', type=int, default=3,
                        help='Group size for hierarchical attention')
    parser.add_argument('--seed', type=int, default=1234)
    
    # Contrastive learning parameters
    parser.add_argument('--contrastive_weight', type=float, default=0.1,
                        help='Weight for contrastive loss')
    parser.add_argument('--temperature', type=float, default=0.07,
                        help='Temperature for contrastive learning')
    
    # Paths and modes
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to load pretrained model')
    parser.add_argument('--comment', type=str, default=None,
                        help='Experiment comment')
    parser.add_argument('--track', type=str, default='DF',
                        choices=['LA', 'DF', 'In-the-Wild'])
    parser.add_argument('--eval_output', type=str, default=None,
                        help='Path to save evaluation results')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='Evaluation mode')
    
    # Interpretability options
    parser.add_argument('--analyze_interpretability', action='store_true', default=False,
                        help='Run interpretability analysis')
    parser.add_argument('--interpretability_samples', type=int, default=10,
                        help='Number of samples for interpretability analysis')
    
    # Data augmentation (RawBoost) - Algorithm selection
    parser.add_argument('--algo', type=int, default=3,
                        help='RawBoost algorithm: 0=None, 1=LnL, 2=ISD, 3=SSI, 4-8=combinations')
    
    # RawBoost - LnL parameters
    parser.add_argument('--nBands', type=int, default=5,
                        help='Number of frequency bands for SSI')
    parser.add_argument('--minF', type=int, default=20,
                        help='Minimum frequency for SSI')
    parser.add_argument('--maxF', type=int, default=8000,
                        help='Maximum frequency for SSI')
    parser.add_argument('--minBW', type=int, default=100,
                        help='Minimum bandwidth for SSI')
    parser.add_argument('--maxBW', type=int, default=1000,
                        help='Maximum bandwidth for SSI')
    parser.add_argument('--minCoeff', type=int, default=10,
                        help='Minimum filter coefficient for SSI')
    parser.add_argument('--maxCoeff', type=int, default=100,
                        help='Maximum filter coefficient for SSI')
    parser.add_argument('--minG', type=int, default=0,
                        help='Minimum gain for SSI')
    parser.add_argument('--maxG', type=int, default=0,
                        help='Maximum gain for SSI')
    parser.add_argument('--minBiasLinNonLin', type=int, default=5,
                        help='Minimum bias for linear non-linearity')
    parser.add_argument('--maxBiasLinNonLin', type=int, default=20,
                        help='Maximum bias for linear non-linearity')
    parser.add_argument('--N_f', type=int, default=5,
                        help='Number of formants for LnL')
    parser.add_argument('--P', type=int, default=10,
                        help='Order of LP analysis for LnL')
    parser.add_argument('--g_sd', type=int, default=2,
                        help='Standard deviation of gain')
    
    # RawBoost - ISD parameters
    parser.add_argument('--SNRmin', type=int, default=10,
                        help='Minimum SNR for ISD')
    parser.add_argument('--SNRmax', type=int, default=40,
                        help='Maximum SNR for ISD')
    
    # Backend options
    parser.add_argument('--cudnn-deterministic-toggle', action='store_false',
                        default=True)
    parser.add_argument('--cudnn-benchmark-toggle', action='store_true',
                        default=False)
    
    args = parser.parse_args()
    
    # Set random seed
    set_random_seed(args.seed, args)
    
    track = args.track
    
    # Define model save path
    model_tag = f'hiercon_{track}_epochs{args.num_epochs}_bs{args.batch_size}_lr{args.lr}_group{args.group_size}'
    if args.comment:
        model_tag = f'{model_tag}_{args.comment}'
    
    model_save_path = os.path.join('models', model_tag)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    
    # Initialize CSV log
    init_csv_log(os.path.join(model_save_path, 'training_log.csv'))
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'🖥️  Device: {device}')
    
    # Create model
    print('\n' + '='*60)
    print('🚀 Initializing Hierarchical Contrastive Model')
    print('='*60)
    model = ModelHierarchicalContrastive(args, device)
    
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print(f'📊 Total parameters: {nb_params:,}')
    print(f'🔧 Group size: {args.group_size}')
    print(f'🔧 Contrastive weight: {args.contrastive_weight}')
    print(f'🔧 Data augmentation: RawBoost algo={args.algo}')
    print('='*60 + '\n')
    
    model = nn.DataParallel(model).to(device)
    
    # Loss function
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = CompatibleCombinedLoss(
        weight=weight,
        temperature=args.temperature,
        contrastive_weight=args.contrastive_weight
    )
    
    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Load model if specified
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f'✅ Model loaded from: {args.model_path}')
    
    # Evaluation mode
    if args.eval:
        print('\n' + '='*60)
        print('🔍 EVALUATION MODE')
        print('='*60)
        
        file_eval = genSpoof_list(
            dir_meta=os.path.join(args.protocols_path),
            is_train=False,
            is_eval=True
        )
        print(f'📊 Number of evaluation trials: {len(file_eval)}')
        
        if args.track == 'In-the-Wild':
            from data_utils_SSL import Dataset_in_the_wild_eval
            eval_set = Dataset_in_the_wild_eval(
                list_IDs=file_eval,
                base_dir=os.path.join(args.database_path)
            )
        else:
            eval_set = Dataset_ASVspoof2021_eval(
                list_IDs=file_eval,
                base_dir=os.path.join(args.database_path)
            )
        
        produce_evaluation_file(eval_set, model, device, args.eval_output)
        
        # Run interpretability analysis if requested
        if args.analyze_interpretability:
            print('\n' + '='*60)
            print('🔬 INTERPRETABILITY ANALYSIS')
            print('='*60)
            
            interp_dir = os.path.join(model_save_path, 'interpretability_eval')
            eval_loader = DataLoader(
                eval_set,
                batch_size=1,
                shuffle=True,
                num_workers=4
            )
            
            analyze_batch_interpretability(
                model.module,  # Unwrap DataParallel
                eval_loader,
                device,
                num_samples=args.interpretability_samples,
                save_dir=interp_dir
            )
        
        sys.exit(0)
    
    # Training mode
    print('\n' + '='*60)
    print('📚 LOADING TRAINING DATA')
    print('='*60)
    
    # Load training data
    d_label_trn, file_train = genSpoof_list(
        dir_meta=os.path.join(
            args.protocols_path,
            'ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'
        ),
        is_train=True,
        is_eval=False
    )
    print(f'📊 Training samples: {len(file_train)}')
    
    train_set = Dataset_ASVspoof2019_train(
        args,
        list_IDs=file_train,
        labels=d_label_trn,
        base_dir=os.path.join(args.database_path, 'ASVspoof2019_LA_train/'),
        algo=args.algo
    )
    
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=8,
        shuffle=True,
        drop_last=True,
        pin_memory=True
    )
    
    del train_set, d_label_trn
    
    # Load validation data
    d_label_dev, file_dev = genSpoof_list(
        dir_meta=os.path.join(
            args.protocols_path,
            'ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'
        ),
        is_train=False,
        is_eval=False
    )
    print(f'📊 Validation samples: {len(file_dev)}')
    
    dev_set = Dataset_ASVspoof2019_train(
        args,
        list_IDs=file_dev,
        labels=d_label_dev,
        base_dir=os.path.join(args.database_path, 'ASVspoof2019_LA_dev/'),
        algo=args.algo
    )
    
    dev_loader = DataLoader(
        dev_set,
        batch_size=args.batch_size,
        num_workers=8,
        shuffle=False,
        pin_memory=True
    )
    
    del dev_set, d_label_dev
    
    print('='*60 + '\n')
    
    # TensorBoard
    writer = SummaryWriter(f'logs/{model_tag}')
    
    # Training loop
    print('='*60)
    print('🎯 STARTING TRAINING')
    print('='*60 + '\n')
    
    best_val_eer = float('inf')
    patience_counter = 0
    
    for epoch in range(args.num_epochs):
        print(f'\n{"="*60}')
        print(f'📅 Epoch {epoch+1}/{args.num_epochs}')
        print(f'{"="*60}')
        
        # Training
        train_loss, train_eer = train_epoch(
            train_loader, model, optimizer, device, criterion
        )
        
        # Validation
        val_loss, val_acc, val_eer = evaluate_accuracy(
            dev_loader, model, device, criterion
        )
        
        # Print metrics
        print(f'\n📊 Training   - Loss: {train_loss:.6f}, EER: {train_eer:.2f}%')
        print(f'📊 Validation - Loss: {val_loss:.6f}, Acc: {val_acc:.2f}%, EER: {val_eer:.2f}%')
        
        # Log to CSV
        log_data = {
            'epoch': epoch + 1,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
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
        
        # Save checkpoint
        checkpoint_path = os.path.join(model_save_path, f'epoch_{epoch+1}.pth')
        torch.save(model.state_dict(), checkpoint_path)
        
        # Save best model
        if val_eer < best_val_eer:
            best_val_eer = val_eer
            patience_counter = 0
            best_model_path = os.path.join(model_save_path, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f'\n🎉 New best model! EER: {best_val_eer:.2f}% saved to {best_model_path}')
        else:
            patience_counter += 1
        
        # Run interpretability analysis every 10 epochs
        if args.analyze_interpretability and (epoch + 1) % 10 == 0:
            print(f'\n🔬 Running interpretability analysis...')
            interp_dir = os.path.join(model_save_path, f'interpretability_epoch_{epoch+1}')
            analyze_batch_interpretability(
                model.module,  # Unwrap DataParallel
                dev_loader,
                device,
                num_samples=args.interpretability_samples,
                save_dir=interp_dir
            )
    
    writer.close()
    
    print('\n' + '='*60)
    print('✅ TRAINING COMPLETED')
    print('='*60)
    print(f'🏆 Best Validation EER: {best_val_eer:.2f}%')
    print(f'💾 Models saved to: {model_save_path}')
    print(f'📊 Logs saved to: logs/{model_tag}')
    print('='*60 + '\n')
