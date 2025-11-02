"""
Attention Heatmap Visualization for G3 Model
可视化 G3 模型的层次注意力机制
生成类似参考图的多层次注意力热图
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import os
from tqdm import tqdm
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# 设置绘图风格
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def extract_attention_maps(model, audio: torch.Tensor, device: torch.device) -> Dict[str, np.ndarray]:
    """
    从 G3 模型提取多层次注意力图
    
    Args:
        model: G3 模型实例
        audio: 音频张量 (B, T) 或 (B, T, 1)
        device: 设备
    
    Returns:
        包含各层注意力权重的字典
    """
    model.eval()
    audio = audio.to(device)
    
    # 确保输入维度正确
    if audio.dim() == 3:
        audio = audio.squeeze(-1)
    
    with torch.no_grad():
        try:
            # 检查是否是 DataParallel 包装的模型
            if hasattr(model, 'module'):
                ssl_model = model.module.ssl_model
                temporal_attn = model.module.temporal_attn
                intra_attn = model.module.intra_attn
                inter_attn = model.module.inter_attn
                group_refine = model.module.group_refine
                utt_refine = model.module.utt_refine
                group_size = model.module.group_size
            else:
                ssl_model = model.ssl_model
                temporal_attn = model.temporal_attn
                intra_attn = model.intra_attn
                inter_attn = model.inter_attn
                group_refine = model.group_refine
                utt_refine = model.utt_refine
                group_size = model.group_size
            
            # 1. 提取 SSL 特征
            x_ssl_feat, layerResult = ssl_model.extract_feat(audio)
            
            # 2. 获取完整特征 - 导入 getAttenF 函数
            from model_g3_heatmap import getAttenF
            _, fullfeature = getAttenF(layerResult)  # (B, L, T, C)
            
            B, L, T, C = fullfeature.shape
            
            # 3. 时序注意力 (Temporal Attention)
            layer_tokens = fullfeature.contiguous().view(B * L, T, C)
            layer_emb, temporal_attn_weights = temporal_attn(layer_tokens)
            
            layer_emb = layer_emb.view(B, L, C)
            temporal_attn_weights = temporal_attn_weights.view(B, L, T)
            
            # 4. 处理分组
            original_L = L
            if layer_emb.size(1) % group_size != 0:
                pad_size = group_size - (layer_emb.size(1) % group_size)
                layer_emb = F.pad(layer_emb, (0, 0, 0, pad_size), mode='constant', value=0)
            
            # 5. 组内注意力 (Intra-Group Attention)
            groups = torch.split(layer_emb, group_size, dim=1)
            group_vecs = []
            intra_attn_list = []
            
            for g in groups:
                g_vec, intra_attn_weights = intra_attn(g)
                g_vec = group_refine(g_vec)
                group_vecs.append(g_vec)
                intra_attn_list.append(intra_attn_weights)
            
            intra_attn_weights = torch.stack(intra_attn_list, dim=1)  # (B, num_groups, group_size)
            
            # 6. 组间注意力 (Inter-Group Attention)
            group_stack = torch.stack(group_vecs, dim=1)
            utt_emb, inter_attn_weights = inter_attn(group_stack)
            utt_emb = utt_refine(utt_emb)
            
        except Exception as e:
            print(f"❌ Error extracting attention: {e}")
            raise
    
    return {
        'temporal_attn': temporal_attn_weights[:, :original_L, :].cpu().numpy(),
        'intra_attn': intra_attn_weights.cpu().numpy(),
        'inter_attn': inter_attn_weights.cpu().numpy(),
        'num_layers': original_L,
        'num_frames': T,
        'group_size': group_size
    }


def plot_attention_heatmap(
    attention_maps: Dict[str, np.ndarray],
    prediction: int,
    label: int,
    save_path: str,
    sample_idx: int = 0
) -> None:
    """
    绘制多层次注意力热图
    
    Args:
        attention_maps: 注意力权重字典
        prediction: 模型预测类别 (0=bonafide, 1=spoof)
        label: 真实标签
        save_path: 保存路径
        sample_idx: 样本索引
    """
    temporal_attn = attention_maps['temporal_attn'][sample_idx]
    intra_attn = attention_maps['intra_attn'][sample_idx]
    inter_attn = attention_maps['inter_attn'][sample_idx]
    
    L, T = temporal_attn.shape
    num_groups = len(inter_attn)
    group_size = attention_maps['group_size']
    
    # 创建图形布局
    fig = plt.figure(figsize=(22, 14))
    gs = GridSpec(4, 4, figure=fig, hspace=0.35, wspace=0.35)
    
    # ============ 1. 时序注意力热图 (主图) ============
    ax1 = fig.add_subplot(gs[0:2, :3])
    im1 = ax1.imshow(temporal_attn, aspect='auto', cmap='RdYlBu_r', 
                     interpolation='bilinear', vmin=0, vmax=temporal_attn.max())
    ax1.set_title('🔍 Temporal Attention Across Layers', 
                 fontsize=16, fontweight='bold', pad=15)
    ax1.set_ylabel('Layer Index', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Time Frame', fontsize=13, fontweight='bold')
    
    # 添加网格线增强可读性
    ax1.grid(False)
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('Attention Weight', fontsize=11, fontweight='bold')
    
    # ============ 2. 预测结果展示 ============
    ax_pred = fig.add_subplot(gs[0, 3])
    ax_pred.axis('off')
    
    pred_label = 'SPOOF' if prediction == 1 else 'BONAFIDE'
    true_label = 'SPOOF' if label == 1 else 'BONAFIDE'
    is_correct = prediction == label
    
    # 设置边框颜色
    pred_color = '#2ecc71' if is_correct else '#e74c3c'  # 绿色/红色
    
    pred_text = (
        f"🎯 Prediction\n{pred_label}\n\n"
        f"✓ Ground Truth\n{true_label}\n\n"
        f"{'✅ CORRECT' if is_correct else '❌ INCORRECT'}"
    )
    
    bbox_props = dict(
        boxstyle='round,pad=1.2', 
        facecolor='#ecf0f1',
        edgecolor=pred_color, 
        linewidth=4, 
        alpha=0.95
    )
    
    ax_pred.text(0.5, 0.5, pred_text, 
                transform=ax_pred.transAxes,
                fontsize=13, 
                va='center', 
                ha='center',
                fontweight='bold', 
                bbox=bbox_props,
                color='#2c3e50')
    
    # ============ 3. 层级重要性分布 ============
    ax2 = fig.add_subplot(gs[1, 3])
    layer_importance = temporal_attn.sum(axis=1) / temporal_attn.sum()
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, L))
    bars = ax2.barh(range(L), layer_importance, color=colors, 
                    alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax2.set_title('📊 Layer Importance', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Normalized Weight', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Layer', fontsize=10, fontweight='bold')
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3, axis='x', linestyle='--')
    
    # 标注最重要的层
    max_idx = np.argmax(layer_importance)
    ax2.annotate(f'Max: {layer_importance[max_idx]:.3f}', 
                xy=(layer_importance[max_idx], max_idx),
                xytext=(10, 0), textcoords='offset points',
                fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    
    # ============ 4. 时序重要性分布 ============
    ax3 = fig.add_subplot(gs[2, :2])
    temporal_importance = temporal_attn.sum(axis=0) / temporal_attn.sum()
    
    ax3.plot(temporal_importance, linewidth=3, color='#3498db', label='Importance')
    ax3.fill_between(range(T), temporal_importance, alpha=0.4, color='#85c1e9')
    
    ax3.set_title('⏱️ Temporal Importance Distribution', 
                 fontsize=12, fontweight='bold')
    ax3.set_xlabel('Time Frame', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Normalized Weight', fontsize=10, fontweight='bold')
    ax3.grid(True, alpha=0.4, linestyle='--')
    ax3.legend(loc='upper right', fontsize=9)
    
    # 标注峰值
    peaks = np.where(temporal_importance > np.percentile(temporal_importance, 90))[0]
    for peak in peaks[:3]:  # 最多标注3个峰值
        ax3.plot(peak, temporal_importance[peak], 'ro', markersize=8)
        ax3.annotate(f'{temporal_importance[peak]:.3f}',
                    xy=(peak, temporal_importance[peak]),
                    xytext=(0, 10), textcoords='offset points',
                    fontsize=8, ha='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='orange', alpha=0.7))
    
    # ============ 5. 组内注意力热图 ============
    ax4 = fig.add_subplot(gs[2, 2:])
    im4 = ax4.imshow(intra_attn, aspect='auto', cmap='YlOrRd', 
                     interpolation='nearest', vmin=0)
    ax4.set_title('🔗 Intra-Group Attention', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Group Index', fontsize=10, fontweight='bold')
    ax4.set_xlabel('Layer in Group', fontsize=10, fontweight='bold')
    
    # 添加数值标注
    for i in range(min(num_groups, 10)):  # 最多标注10组
        for j in range(group_size):
            if intra_attn[i, j] > 0.01:  # 只标注显著值
                text = ax4.text(j, i, f'{intra_attn[i, j]:.2f}',
                               ha="center", va="center", color="black",
                               fontsize=8, fontweight='bold')
    
    cbar4 = plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    cbar4.set_label('Attention Weight', fontsize=9, fontweight='bold')
    
    # ============ 6. 组间注意力柱状图 ============
    ax5 = fig.add_subplot(gs[3, :2])
    colors_inter = plt.cm.Oranges(np.linspace(0.4, 0.9, num_groups))
    
    bars = ax5.bar(range(num_groups), inter_attn, color=colors_inter, 
                   alpha=0.85, edgecolor='black', linewidth=1.5)
    
    ax5.set_title('🌐 Inter-Group Attention', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Group Index', fontsize=10, fontweight='bold')
    ax5.set_ylabel('Attention Weight', fontsize=10, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # 标注最重要的组
    max_group = np.argmax(inter_attn)
    ax5.annotate(f'Max Group\n{inter_attn[max_group]:.3f}',
                xy=(max_group, inter_attn[max_group]),
                xytext=(0, 15), textcoords='offset points',
                fontsize=9, fontweight='bold',
                ha='center',
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8))
    
    # ============ 7. 统计摘要 ============
    ax_stats = fig.add_subplot(gs[3, 2:])
    ax_stats.axis('off')
    
    stats_text = (
        "📈 Attention Statistics\n"
        "─" * 30 + "\n"
        f"Layers: {L}\n"
        f"Time Frames: {T}\n"
        f"Groups: {num_groups}\n"
        f"Group Size: {group_size}\n\n"
        f"Temporal Entropy: {-np.sum(temporal_importance * np.log(temporal_importance + 1e-10)):.3f}\n"
        f"Layer Diversity: {1 - np.max(layer_importance):.3f}\n"
        f"Top Layer: #{np.argmax(layer_importance)}\n"
        f"Top Group: #{max_group}\n"
    )
    
    bbox_stats = dict(
        boxstyle='round,pad=1',
        facecolor='#f8f9fa',
        edgecolor='#34495e',
        linewidth=2,
        alpha=0.9
    )
    
    ax_stats.text(0.1, 0.5, stats_text,
                 transform=ax_stats.transAxes,
                 fontsize=10,
                 va='center',
                 fontfamily='monospace',
                 bbox=bbox_stats)
    
    # ============ 总标题 ============
    sample_type = "SPOOF" if label == 1 else "BONAFIDE"
    main_title = f'🎵 G3 Model Hierarchical Attention Analysis - {sample_type} Sample'
    plt.suptitle(main_title, fontsize=18, fontweight='bold', y=0.98)
    
    # 保存图像
    plt.savefig(save_path, dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    plt.close()


def analyze_attention_patterns(
    model,
    data_loader,
    device: torch.device,
    num_samples: int = 20,
    save_dir: str = 'attention_heatmaps'
) -> None:
    """
    批量分析并生成注意力热图
    
    Args:
        model: G3 模型
        data_loader: 数据加载器
        device: 设备
        num_samples: 生成样本数
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    sample_count = 0
    
    print(f'\n{"="*70}')
    print('🔬 Analyzing Attention Patterns')
    print(f'{"="*70}')
    print(f'📁 Save Directory: {save_dir}')
    print(f'🎯 Target Samples: {num_samples}')
    print(f'{"="*70}\n')
    
    with tqdm(total=num_samples, desc='Generating heatmaps') as pbar:
        for batch_x, batch_y in data_loader:
            if sample_count >= num_samples:
                break
            
            batch_x = batch_x.to(device)
            batch_y_np = batch_y.cpu().numpy()
            
            # 获取预测 - 修复：正确处理元组输出
            with torch.no_grad():
                model_output = model(batch_x)
                
                # G3 模型返回 (output, contrastive_loss, supcon_loss, embeddings)
                if isinstance(model_output, tuple):
                    output = model_output[0]  # 只取第一个元素 (log probabilities)
                else:
                    output = model_output
                
                predictions = torch.argmax(output, dim=1).cpu().numpy()
            
            # 为每个样本生成热图
            for i in range(len(batch_y)):
                if sample_count >= num_samples:
                    break
                
                try:
                    # 提取注意力图
                    attention_maps = extract_attention_maps(
                        model, 
                        batch_x[i:i+1], 
                        device
                    )
                    
                    # 生成文件名
                    label_name = "spoof" if batch_y_np[i] == 1 else "bonafide"
                    pred_name = "spoof" if predictions[i] == 1 else "bonafide"
                    correct = "✓" if predictions[i] == batch_y_np[i] else "✗"
                    
                    filename = f'sample_{sample_count:03d}_{label_name}_pred_{pred_name}_{correct}.png'
                    save_path = os.path.join(save_dir, filename)
                    
                    # 绘制热图
                    plot_attention_heatmap(
                        attention_maps,
                        predictions[i],
                        batch_y_np[i],
                        save_path,
                        sample_idx=0
                    )
                    
                    sample_count += 1
                    pbar.update(1)
                    
                except Exception as e:
                    print(f'\n⚠️  Warning: Failed to process sample {sample_count}: {e}')
                    import traceback
                    traceback.print_exc()
                    continue
    
    print(f'\n{"="*70}')
    print(f'✅ Analysis Complete!')
    print(f'📊 Generated {sample_count} attention heatmaps')
    print(f'📁 Location: {save_dir}/')
    print(f'{"="*70}\n')


if __name__ == '__main__':
    print("🎨 Attention Heatmap Visualization Module")
    print("This module is designed to be imported by main.py")
    print("Please run main.py with --visualize_attention flag")