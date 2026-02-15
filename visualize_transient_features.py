"""
可视化和分析瞬态 vs 持久特征

识别标准：
- 瞬态特征: median_lifetime < window_size (短暂激活)
- 持久特征: median_lifetime >= window_size (长时间激活)

这个脚本会：
1. 分析特征的激活持续时间
2. 识别瞬态和持久特征
3. 可视化特征激活模式
4. 保存详细的特征分类结果
"""

import argparse
import json
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from data_utils_SSL import genSpoof_list, Dataset_ASVspoof2019_train
from model_window_topk import Model


def analyze_feature_lifetimes(model, dataloader, num_samples=20, device='cuda'):
    """
    分析特征的激活持续时间（lifetime）
    
    Args:
        model: 训练好的模型
        dataloader: 数据加载器
        num_samples: 分析样本数
        device: 设备
    
    Returns:
        dict: 特征分类和统计信息
    """
    model.eval()
    window_size = model.sae.window_size if model.use_sae else 8
    dict_size = model.sae.dict_size if model.use_sae else 4096
    
    print(f"Window size: {window_size}")
    print(f"Dictionary size: {dict_size}")
    
    # 收集所有特征的lifetime
    feature_lifetimes = {i: [] for i in range(dict_size)}  # {feature_idx: [lifetimes]}
    feature_activation_patterns = {}  # 保存示例激活模式
    
    count = 0
    print(f"\n分析特征激活模式...")
    
    for batch_idx, batch in enumerate(dataloader):
        if count >= num_samples:
            break
        
        inputs, labels = batch[0].to(device), batch[1].to(device)
        
        with torch.no_grad():
            _, interp = model(inputs, return_interpretability=True, return_sae_loss=False)
            sparse_features = interp['sparse_features']  # [B, T, D]
        
        for sample_idx in range(sparse_features.shape[0]):
            if count >= num_samples:
                break
            
            sample_features = sparse_features[sample_idx]  # [T, D]
            T, D = sample_features.shape
            label = 'bonafide' if labels[sample_idx].item() == 0 else 'spoof'
            
            print(f"  样本 {count+1}/{num_samples} (label={label}, T={T})", flush=True)
            
            # 分析每个特征
            active_mask = (sample_features > 0).cpu().numpy()  # [T, D]
            
            for feat_idx in range(D):
                feature_active = active_mask[:, feat_idx]
                
                if not feature_active.any():
                    continue
                
                # 计算连续激活段落
                lifetimes = []
                current_lifetime = 0
                
                for t in range(T):
                    if feature_active[t]:
                        current_lifetime += 1
                    else:
                        if current_lifetime > 0:
                            lifetimes.append(current_lifetime)
                            current_lifetime = 0
                
                if current_lifetime > 0:
                    lifetimes.append(current_lifetime)
                
                # 记录
                if lifetimes:
                    feature_lifetimes[feat_idx].extend(lifetimes)
                    
                    # 保存前几个特征的激活模式作为示例
                    if feat_idx not in feature_activation_patterns and len(feature_activation_patterns) < 20:
                        feature_activation_patterns[feat_idx] = {
                            'activation': feature_active.copy(),
                            'values': sample_features[:, feat_idx].cpu().numpy(),
                            'lifetimes': lifetimes,
                            'label': label,
                            'sample_id': count
                        }
            
            count += 1
    
    # 统计每个特征的lifetime分布
    print(f"\n计算特征统计信息...")
    
    transient_features = []
    persistent_features = []
    feature_stats = {}
    
    for feat_idx in range(dict_size):
        if not feature_lifetimes[feat_idx]:
            continue
        
        lifetimes_array = np.array(feature_lifetimes[feat_idx])
        
        stats = {
            'mean': float(lifetimes_array.mean()),
            'median': float(np.median(lifetimes_array)),
            'std': float(lifetimes_array.std()),
            'min': float(lifetimes_array.min()),
            'max': float(lifetimes_array.max()),
            'count': len(lifetimes_array),  # 出现次数
        }
        
        feature_stats[feat_idx] = stats
        
        # 分类
        if stats['median'] < window_size:
            transient_features.append(feat_idx)
        else:
            persistent_features.append(feat_idx)
    
    print(f"\n特征分类结果:")
    print(f"  瞬态特征: {len(transient_features)} 个")
    print(f"  持久特征: {len(persistent_features)} 个")
    print(f"  未激活特征: {dict_size - len(feature_stats)} 个")
    
    # 显示一些例子
    print(f"\n瞬态特征示例 (前10个):")
    for feat_idx in transient_features[:10]:
        stats = feature_stats[feat_idx]
        print(f"  特征 {feat_idx}: median_lifetime={stats['median']:.1f}, "
              f"mean={stats['mean']:.1f}, count={stats['count']}")
    
    print(f"\n持久特征示例 (前10个):")
    for feat_idx in persistent_features[:10]:
        stats = feature_stats[feat_idx]
        print(f"  特征 {feat_idx}: median_lifetime={stats['median']:.1f}, "
              f"mean={stats['mean']:.1f}, count={stats['count']}")
    
    return {
        'transient_features': transient_features,
        'persistent_features': persistent_features,
        'feature_stats': feature_stats,
        'feature_activation_patterns': feature_activation_patterns,
        'window_size': window_size,
        'num_samples': count
    }


def visualize_feature_patterns(analysis_results, output_dir):
    """
    可视化瞬态和持久特征的激活模式
    """
    os.makedirs(output_dir, exist_ok=True)
    
    patterns = analysis_results['feature_activation_patterns']
    transient_features = set(analysis_results['transient_features'])
    persistent_features = set(analysis_results['persistent_features'])
    window_size = analysis_results['window_size']
    
    # 分别收集瞬态和持久特征的示例
    transient_examples = []
    persistent_examples = []
    
    for feat_idx, pattern_data in patterns.items():
        if feat_idx in transient_features:
            transient_examples.append((feat_idx, pattern_data))
        elif feat_idx in persistent_features:
            persistent_examples.append((feat_idx, pattern_data))
    
    # 1. 绘制瞬态特征示例
    if transient_examples:
        n_examples = min(5, len(transient_examples))
        fig, axes = plt.subplots(n_examples, 1, figsize=(15, 2*n_examples))
        if n_examples == 1:
            axes = [axes]
        
        fig.suptitle(f'瞬态特征激活模式 (lifetime < {window_size})', fontsize=14, fontweight='bold')
        
        for idx, (feat_idx, pattern_data) in enumerate(transient_examples[:n_examples]):
            ax = axes[idx]
            activation = pattern_data['activation']
            values = pattern_data['values']
            lifetimes = pattern_data['lifetimes']
            
            # 绘制激活强度
            time_steps = np.arange(len(values))
            ax.fill_between(time_steps, 0, values, alpha=0.3, color='red')
            ax.plot(time_steps, values, color='darkred', linewidth=2)
            
            # 标记激活区域
            active_regions = np.where(activation)[0]
            if len(active_regions) > 0:
                ax.scatter(active_regions, values[active_regions], 
                          color='red', s=20, alpha=0.6, zorder=5)
            
            median_lt = np.median(lifetimes)
            ax.set_title(f'Feature {feat_idx} | median_lifetime={median_lt:.1f} | '
                        f'lifetimes={lifetimes} | {pattern_data["label"]}')
            ax.set_ylabel('Activation')
            ax.grid(True, alpha=0.3)
            
            if idx == n_examples - 1:
                ax.set_xlabel('Time Steps')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'transient_features_examples.png'), dpi=150)
        plt.close()
        print(f"保存: transient_features_examples.png")
    
    # 2. 绘制持久特征示例
    if persistent_examples:
        n_examples = min(5, len(persistent_examples))
        fig, axes = plt.subplots(n_examples, 1, figsize=(15, 2*n_examples))
        if n_examples == 1:
            axes = [axes]
        
        fig.suptitle(f'持久特征激活模式 (lifetime >= {window_size})', fontsize=14, fontweight='bold')
        
        for idx, (feat_idx, pattern_data) in enumerate(persistent_examples[:n_examples]):
            ax = axes[idx]
            activation = pattern_data['activation']
            values = pattern_data['values']
            lifetimes = pattern_data['lifetimes']
            
            # 绘制激活强度
            time_steps = np.arange(len(values))
            ax.fill_between(time_steps, 0, values, alpha=0.3, color='blue')
            ax.plot(time_steps, values, color='darkblue', linewidth=2)
            
            # 标记激活区域
            active_regions = np.where(activation)[0]
            if len(active_regions) > 0:
                ax.scatter(active_regions, values[active_regions], 
                          color='blue', s=20, alpha=0.6, zorder=5)
            
            median_lt = np.median(lifetimes)
            ax.set_title(f'Feature {feat_idx} | median_lifetime={median_lt:.1f} | '
                        f'lifetimes={lifetimes} | {pattern_data["label"]}')
            ax.set_ylabel('Activation')
            ax.grid(True, alpha=0.3)
            
            if idx == n_examples - 1:
                ax.set_xlabel('Time Steps')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'persistent_features_examples.png'), dpi=150)
        plt.close()
        print(f"保存: persistent_features_examples.png")
    
    # 3. Lifetime分布对比
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 瞬态特征的lifetime分布
    transient_lifetimes = []
    for feat_idx in analysis_results['transient_features']:
        if feat_idx in analysis_results['feature_stats']:
            stats = analysis_results['feature_stats'][feat_idx]
            transient_lifetimes.extend([stats['median']] * int(stats['count']))
    
    if transient_lifetimes:
        axes[0].hist(transient_lifetimes, bins=30, color='red', alpha=0.7, edgecolor='black')
        axes[0].axvline(window_size, color='black', linestyle='--', linewidth=2, 
                       label=f'window_size={window_size}')
        axes[0].set_title(f'瞬态特征 Lifetime 分布 (n={len(analysis_results["transient_features"])})')
        axes[0].set_xlabel('Median Lifetime')
        axes[0].set_ylabel('Frequency')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    
    # 持久特征的lifetime分布
    persistent_lifetimes = []
    for feat_idx in analysis_results['persistent_features']:
        if feat_idx in analysis_results['feature_stats']:
            stats = analysis_results['feature_stats'][feat_idx]
            persistent_lifetimes.extend([stats['median']] * int(stats['count']))
    
    if persistent_lifetimes:
        axes[1].hist(persistent_lifetimes, bins=30, color='blue', alpha=0.7, edgecolor='black')
        axes[1].axvline(window_size, color='black', linestyle='--', linewidth=2,
                       label=f'window_size={window_size}')
        axes[1].set_title(f'持久特征 Lifetime 分布 (n={len(analysis_results["persistent_features"])})')
        axes[1].set_xlabel('Median Lifetime')
        axes[1].set_ylabel('Frequency')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lifetime_distributions.png'), dpi=150)
    plt.close()
    print(f"保存: lifetime_distributions.png")


def main():
    parser = argparse.ArgumentParser(description='可视化瞬态和持久特征')
    parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    parser.add_argument('--database_path', type=str, 
                       default='/data/projects/punim2637/nnliang/Datasets/LA')
    parser.add_argument('--protocols_path', type=str,
                       default='/data/projects/punim2637/nnliang/Datasets/LA/ASVspoof2019_LA_cm_protocols')
    parser.add_argument('--output_dir', type=str, default='transient_analysis_visualization')
    parser.add_argument('--num_samples', type=int, default=20, 
                       help='分析的样本数（建议10-50）')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    print("="*80)
    print("瞬态特征分析和可视化")
    print("="*80)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型
    print("\n[1/4] 加载模型...")
    device = torch.device(args.device)
    
    class ModelArgs:
        pass
    
    model_args = ModelArgs()
    model = Model(
        args=model_args,
        device=device,
        use_sae=True,
        use_sparse_features=True,
        sae_dict_size=4096,
        sae_k=128,
        sae_window_size=8,
    ).to(device)
    
    checkpoint = torch.load(args.model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    print("模型加载成功!")
    
    # 加载数据
    print("\n[2/4] 加载数据...")
    class DataArgs:
        pass
    
    data_args = DataArgs()
    d_label_trn, file_train = genSpoof_list(
        dir_meta=os.path.join(args.protocols_path, 'ASVspoof2019.LA.cm.train.trn.txt'),
        is_train=True, is_eval=False
    )
    
    train_set = Dataset_ASVspoof2019_train(
        args=data_args, list_IDs=file_train, labels=d_label_trn,
        base_dir=os.path.join(args.database_path, 'ASVspoof2019_LA_train/'),
        algo=0
    )
    
    train_loader = DataLoader(train_set, batch_size=4, shuffle=False, num_workers=0)
    print("数据加载成功!")
    
    # 分析特征lifetime
    print("\n[3/4] 分析特征lifetime...")
    analysis_results = analyze_feature_lifetimes(
        model, train_loader, num_samples=args.num_samples, device=device
    )
    
    # 可视化
    print(f"\n[4/4] 生成可视化...")
    visualize_feature_patterns(analysis_results, args.output_dir)
    
    # 保存结果
    print(f"\n保存分析结果...")
    
    # 准备可序列化的数据
    save_data = {
        'transient_features': analysis_results['transient_features'],
        'persistent_features': analysis_results['persistent_features'],
        'feature_stats': {str(k): v for k, v in analysis_results['feature_stats'].items()},
        'window_size': analysis_results['window_size'],
        'num_samples': analysis_results['num_samples'],
        'summary': {
            'num_transient': len(analysis_results['transient_features']),
            'num_persistent': len(analysis_results['persistent_features']),
            'total_active': len(analysis_results['feature_stats']),
            'total_dict_size': 4096
        }
    }
    
    with open(os.path.join(args.output_dir, 'feature_classification.json'), 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"\n结果保存到: {args.output_dir}/")
    print("  - transient_features_examples.png")
    print("  - persistent_features_examples.png")
    print("  - lifetime_distributions.png")
    print("  - feature_classification.json")
    
    print("\n" + "="*80)
    print("总结")
    print("="*80)
    print(f"瞬态特征 (lifetime < {analysis_results['window_size']}): "
          f"{len(analysis_results['transient_features'])} 个")
    print(f"持久特征 (lifetime >= {analysis_results['window_size']}): "
          f"{len(analysis_results['persistent_features'])} 个")
    print(f"比例: {len(analysis_results['transient_features'])/(len(analysis_results['transient_features'])+len(analysis_results['persistent_features'])):.2%} 是瞬态的")
    print("="*80)


if __name__ == '__main__':
    main()
