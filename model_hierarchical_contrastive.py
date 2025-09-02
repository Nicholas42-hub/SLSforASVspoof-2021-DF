import random
import sys
from typing import Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import fairseq

class SSLModel(nn.Module):
    def __init__(self,device):
        super(SSLModel, self).__init__()
        
        cp_path = '/root/autodl-tmp/SLSforASVspoof-2021-DF/xlsr2_300m.pt'
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
        self.model = model[0]
        self.device=device
        self.out_dim = 1024
        return

    def extract_feat(self, input_data):
        if next(self.model.parameters()).device != input_data.device \
           or next(self.model.parameters()).dtype != input_data.dtype:
            self.model.to(input_data.device, dtype=input_data.dtype)
            self.model.train()

        if input_data.ndim == 3:
            input_tmp = input_data[:, :, 0]
        else:
            input_tmp = input_data
            
        # [batch, length, dim]
        emb = self.model(input_tmp, mask=False, features_only=True)['x']
        layerresult = self.model(input_tmp, mask=False, features_only=True)['layer_results']
        return emb, layerresult

def getAttenF(layerResult):
    poollayerResult = []
    fullf = []
    for layer in layerResult:
        layery = layer[0].transpose(0, 1).transpose(1, 2) #(x,z)  x(201,b,1024) (b,201,1024) (b,1024,201)
        layery = F.adaptive_avg_pool1d(layery, 1) #(b,1024,1)
        layery = layery.transpose(1, 2) # (b,1,1024)
        poollayerResult.append(layery)

        x = layer[0].transpose(0, 1)
        x = x.view(x.size(0), -1,x.size(1), x.size(2))
        fullf.append(x)

    layery = torch.cat(poollayerResult, dim=1)
    fullfeature = torch.cat(fullf, dim=1)
    return layery, fullfeature

class AttnPool(nn.Module):
    def __init__(self, in_dim: int, attn_dim: int = 128):
        super().__init__()
        self.proj = nn.Linear(in_dim, attn_dim)
        self.score = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        # x: (B, T, C)
        e = torch.tanh(self.proj(x))
        s = self.score(e).squeeze(-1)
        if mask is not None:
            s = s.masked_fill(~mask.bool(), float('-inf'))
        a = torch.softmax(s, dim=1)
        out = torch.sum(a.unsqueeze(-1) * x, dim=1)
        return out, a

class ResidualRefine(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.SELU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, in_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor):
        return x + self.net(x)

class DomainRobustContrastiveLoss(nn.Module):
    """
    域鲁棒的对比学习损失 - 提高In-the-Wild泛化性能
    """
    def __init__(self, temperature=0.07, margin=0.1):
        super(DomainRobustContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.margin = margin
        
    def forward(self, features, labels):
        batch_size = features.size(0)
        
        if batch_size < 2:
            return torch.tensor(0.0, device=features.device)
        
        # 轻度归一化，保持特征多样性
        features = F.normalize(features, dim=1) * 0.9  # 减少归一化强度
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # 创建正样本和负样本mask
        labels = labels.view(-1, 1)
        positive_mask = torch.eq(labels, labels.T).float().to(features.device)
        negative_mask = 1.0 - positive_mask
        positive_mask = positive_mask - torch.eye(batch_size).to(features.device)
        
        # 检查正样本对
        positive_pairs = positive_mask.sum(dim=1)
        if positive_pairs.sum() == 0:
            return torch.tensor(0.0, device=features.device)
        
        # 使用margin-based对比损失，更稳定且减少过拟合
        positive_sim = (positive_mask * similarity_matrix).sum(dim=1) / (positive_pairs + 1e-8)
        negative_sim = (negative_mask * similarity_matrix).sum(dim=1) / (negative_mask.sum(dim=1) + 1e-8)
        
        # Margin-based损失：鼓励正样本相似，负样本分离
        margin_loss = torch.clamp(self.margin + negative_sim - positive_sim, min=0.0)
        valid_samples = positive_pairs > 0
        
        if valid_samples.sum() == 0:
            return torch.tensor(0.0, device=features.device)
        
        loss = margin_loss[valid_samples].mean()
        
        # 限制损失范围，防止梯度爆炸和过拟合
        loss = torch.clamp(loss, 0, 1.0)
        
        return loss

class FeatureDiversityLoss(nn.Module):
    """
    特征多样性损失 - 防止学习到数据集特定的模式
    """
    def __init__(self, min_distance=0.1):
        super(FeatureDiversityLoss, self).__init__()
        self.min_distance = min_distance
        
    def forward(self, features):
        batch_size = features.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, device=features.device)
        
        # 计算特征间的余弦距离
        features_norm = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features_norm, features_norm.T)
        
        # 移除对角线（自相似度）
        mask = 1 - torch.eye(batch_size, device=features.device)
        similarities = similarity_matrix * mask
        
        # 惩罚过于相似的特征（防止特征塌陷）
        high_similarity_penalty = torch.clamp(similarities - (1 - self.min_distance), min=0)
        
        # 只对非零相似度计算损失
        valid_pairs = mask.sum()
        if valid_pairs > 0:
            return high_similarity_penalty.sum() / valid_pairs
        return torch.tensor(0.0, device=features.device)

class DomainRobustCombinedLoss(nn.Module):
    """
    域鲁棒的组合损失 - 针对In-the-Wild泛化优化
    """
    def __init__(self, weight=None, temperature=0.07, contrastive_weight=0.1, 
                 diversity_weight=0.05, focal_alpha=0.25, focal_gamma=2.0):
        super(DomainRobustCombinedLoss, self).__init__()
        
        # 使用Focal Loss替代标准CCE，减少简单样本的影响
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.class_weights = weight
        
        # 域鲁棒的对比学习
        self.contrastive_loss = DomainRobustContrastiveLoss(
            temperature=temperature, margin=0.1)
        
        # 特征多样性损失
        self.diversity_loss = FeatureDiversityLoss(min_distance=0.15)
        
        # 权重参数
        self.initial_contrastive_weight = contrastive_weight
        self.diversity_weight = diversity_weight
        
        # 自适应权重调整
        self.step_count = 0
        self.loss_history = {'cce': [], 'contrastive': [], 'diversity': []}
        
    def focal_loss(self, logits, labels):
        """
        Focal Loss实现 - 关注困难样本，提高泛化性
        """
        ce_loss = F.cross_entropy(logits, labels, weight=self.class_weights, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.focal_alpha * (1-pt)**self.focal_gamma * ce_loss
        return focal_loss.mean()
        
    def forward(self, logits, features, labels):
        # 1. Focal Loss（替代标准CCE）
        focal = self.focal_loss(logits, labels)
        
        # 2. 域鲁棒对比损失
        contrastive = self.contrastive_loss(features, labels)
        
        # 3. 特征多样性损失
        diversity = self.diversity_loss(features)
        
        # 更新损失历史
        self.loss_history['cce'].append(focal.item())
        self.loss_history['contrastive'].append(contrastive.item())
        self.loss_history['diversity'].append(diversity.item())
        
        # 保持最近20步的历史
        for key in self.loss_history:
            if len(self.loss_history[key]) > 20:
                self.loss_history[key].pop(0)
        
        # 自适应权重调整 - 基于训练阶段动态调整
        if len(self.loss_history['cce']) > 5:
            avg_focal = sum(self.loss_history['cce'][-5:]) / 5
            avg_contrastive = sum(self.loss_history['contrastive'][-5:]) / 5
            
            # 当focal loss已经很低时，增加正则化项的权重
            if avg_focal < 0.01:  # 训练后期
                contrastive_weight = self.initial_contrastive_weight * 0.5  # 减少对比学习
                diversity_weight = self.diversity_weight * 1.5  # 增加多样性
            elif avg_focal < 0.1:  # 训练中期
                contrastive_weight = self.initial_contrastive_weight * 0.8
                diversity_weight = self.diversity_weight * 1.2
            else:  # 训练早期
                contrastive_weight = self.initial_contrastive_weight
                diversity_weight = self.diversity_weight
        else:
            contrastive_weight = self.initial_contrastive_weight
            diversity_weight = self.diversity_weight
        
        # 组合损失
        total_loss = focal + contrastive_weight * contrastive + diversity_weight * diversity
        
        self.step_count += 1
        
        # 修复：只返回3个值以保持与原来代码的兼容性
        # 将diversity损失合并到contrastive_combined中
        contrastive_combined = contrastive_weight * contrastive + diversity_weight * diversity
        
        return total_loss, focal, contrastive_combined

class ModelHierarchicalContrastive(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.device = device
        self.ssl_model = SSLModel(self.device)
        self.d_model = 1024
        self.group_size = getattr(args, "group_size", 3)  

        # 层次注意力模块
        self.temporal_attn = AttnPool(in_dim=self.d_model, attn_dim=128)
        self.intra_attn = AttnPool(in_dim=self.d_model, attn_dim=128)
        self.group_refine = ResidualRefine(in_dim=self.d_model, hidden=512, dropout=0.15)  # 增加dropout
        self.inter_attn = AttnPool(in_dim=self.d_model, attn_dim=128)
        self.utt_refine = ResidualRefine(in_dim=self.d_model, hidden=512, dropout=0.15)

        # 域鲁棒的对比学习投影头
        self.projection_head = nn.Sequential(
            nn.Linear(self.d_model, 512),
            nn.LayerNorm(512),  # 添加层归一化
            nn.ReLU(),
            nn.Dropout(0.2),  # 增加dropout
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(256, 128)
        )

        # 域鲁棒分类器 - 增加正则化
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),  # 增加dropout
            nn.Linear(self.d_model, 512),
            nn.LayerNorm(512),  # 添加层归一化
            nn.SELU(inplace=True),
            nn.Dropout(p=0.15),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.SELU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(256, 2),
        )
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x, return_features=False, add_noise=False):
        # 可选的输入噪声注入（训练时增强鲁棒性）
        if add_noise and self.training:
            noise_scale = 0.01
            x = x + torch.randn_like(x) * noise_scale
        
        x_ssl_feat, layerResult = self.ssl_model.extract_feat(x.squeeze(-1))
        _, fullfeature = getAttenF(layerResult)  # (B, L, T, C)

        B, L, T, C = fullfeature.shape
        layer_tokens = fullfeature.contiguous().view(B * L, T, C)
        layer_emb, _ = self.temporal_attn(layer_tokens)
        layer_emb = layer_emb.view(B, L, C)

        # 处理分组
        if layer_emb.size(1) % self.group_size != 0:
            pad_size = self.group_size - (layer_emb.size(1) % self.group_size)
            layer_emb = F.pad(layer_emb, (0, 0, 0, pad_size), mode='constant', value=0)

        groups = torch.split(layer_emb, self.group_size, dim=1)
        group_vecs = []
        for g in groups:
            g_vec, _ = self.intra_attn(g)
            g_vec = self.group_refine(g_vec)
            group_vecs.append(g_vec)

        group_stack = torch.stack(group_vecs, dim=1)
        utt_emb, _ = self.inter_attn(group_stack)
        utt_emb = self.utt_refine(utt_emb)

        # 分类输出
        logits = self.classifier(utt_emb)
        output = self.logsoftmax(logits)
        
        if return_features:
            # 域鲁棒的投影特征
            projected_features = self.projection_head(utt_emb)
            return output, projected_features
        
        return output

# 为了兼容原有代码，保持类名映射
class BalancedContrastiveLoss(DomainRobustContrastiveLoss):
    """向后兼容的别名"""
    pass

class AdaptiveCombinedLoss(DomainRobustCombinedLoss):
    """向后兼容的别名"""
    pass

# 保持原来的类名
CombinedLoss = DomainRobustCombinedLoss
AdaptiveCombinedLoss = DomainRobustCombinedLoss

# 添加一个简化版本的损失函数，与你的main代码完全兼容
class CompatibleCombinedLoss(nn.Module):
    """
    与原有代码完全兼容的版本 - 保持3个返回值
    """
    def __init__(self, weight=None, temperature=0.07, contrastive_weight=0.1):
        super(CompatibleCombinedLoss, self).__init__()
        
        # 标准损失函数
        self.cce_loss = nn.CrossEntropyLoss(weight=weight)
        
        # 域鲁棒的对比学习
        self.contrastive_loss = DomainRobustContrastiveLoss(
            temperature=temperature, margin=0.1)
        
        # 特征多样性损失
        self.diversity_loss = FeatureDiversityLoss(min_distance=0.15)
        
        # 权重参数
        self.contrastive_weight = contrastive_weight
        self.diversity_weight = 0.02  # 较小的多样性权重
        
    def forward(self, logits, features, labels):
        # 1. 标准CCE损失
        cce = self.cce_loss(logits, labels)
        
        # 2. 域鲁棒对比损失
        contrastive = self.contrastive_loss(features, labels)
        
        # 3. 特征多样性损失
        diversity = self.diversity_loss(features)
        
        # 组合对比学习相关损失
        combined_contrastive = self.contrastive_weight * contrastive + self.diversity_weight * diversity
        
        # 总损失
        total_loss = cce + combined_contrastive
        
        # 返回3个值：总损失、CCE损失、组合的对比损失
        return total_loss, cce, combined_contrastive

# 更新类名映射以保持兼容性
CombinedLoss = CompatibleCombinedLoss
AdaptiveCombinedLoss = CompatibleCombinedLoss