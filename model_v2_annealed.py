import random
import sys
from typing import Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import fairseq
import math

# 复用基础组件 (SSLModel, getAttenF, AttnPool, ResidualRefine, SupConLoss, FeatureDiversityLoss)
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
            
        emb = self.model(input_tmp, mask=False, features_only=True)['x']
        layerresult = self.model(input_tmp, mask=False, features_only=True)['layer_results']
        return emb, layerresult

def getAttenF(layerResult):
    poollayerResult = []
    fullf = []
    for layer in layerResult:
        layery = layer[0].transpose(0, 1).transpose(1, 2)
        layery = F.adaptive_avg_pool1d(layery, 1)
        layery = layery.transpose(1, 2)
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

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.3, contrast_mode='all', base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        device = features.device

        if len(features.shape) < 3:
            features = features.unsqueeze(1)

        batch_size = features.shape[0]
        if labels is not None:
            labels = labels.contiguous().view(-1, 1)
            mask = torch.eq(labels, labels.T).float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        
        if self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            anchor_feature = features[:, 0]
            anchor_count = 1

        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class FeatureDiversityLoss(nn.Module):
    def __init__(self, min_distance=0.15):
        super(FeatureDiversityLoss, self).__init__()
        self.min_distance = min_distance
        
    def forward(self, features):
        batch_size = features.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, device=features.device)
        
        features_norm = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features_norm, features_norm.T)
        
        mask = 1 - torch.eye(batch_size, device=features.device)
        similarities = similarity_matrix * mask
        
        high_similarity_penalty = torch.clamp(similarities - (1 - self.min_distance), min=0)
        
        valid_pairs = mask.sum()
        if valid_pairs > 0:
            return high_similarity_penalty.sum() / valid_pairs
        return torch.tensor(0.0, device=features.device)

class DynamicAnnealedCombinedLoss(nn.Module):
    """
    方案二：Dynamic Annealed Lambda - 防止早期崩塌
    λ(t) = min(t / T_anneal, 1.0) * λ_max
    """
    def __init__(self, weight=None, temperature=0.3, lambda_max=0.1, T_anneal=5):
        super(DynamicAnnealedCombinedLoss, self).__init__()
        
        # 标准损失函数
        self.cce_loss = nn.CrossEntropyLoss(weight=weight)
        
        # Temperature-scaled SupConLoss
        self.contrastive_loss = SupConLoss(temperature=temperature)
        
        # 特征多样性损失
        self.diversity_loss = FeatureDiversityLoss(min_distance=0.15)
        
        # Dynamic annealing参数
        self.lambda_max = lambda_max
        self.T_anneal = T_anneal
        self.current_epoch = 0
        self.diversity_weight = 0.02
        
        # 训练统计
        self.training_stats = []
        
    def update_epoch(self, epoch):
        """更新当前epoch用于动态调整lambda"""
        self.current_epoch = epoch
        
    def get_current_lambda(self):
        """计算当前的lambda值 - λ(t) = min(t / T_anneal, 1.0) * λ_max"""
        lambda_t = min(self.current_epoch / self.T_anneal, 1.0) * self.lambda_max
        return lambda_t
        
    def forward(self, logits, features, labels):
        batch_size = features.size(0)
        
        # 1. 标准CCE损失
        cce = self.cce_loss(logits, labels)
        
        # 2. Temperature-scaled contrastive loss
        contrastive = self.contrastive_loss(features, labels)
        
        # 3. Batch size归一化
        normalized_contrastive = contrastive / batch_size
        
        # 4. 特征多样性损失
        diversity = self.diversity_loss(features)
        
        # 5. Dynamic annealed lambda
        current_lambda = self.get_current_lambda()
        
        # 组合对比学习相关损失（使用动态权重）
        combined_contrastive = current_lambda * normalized_contrastive + self.diversity_weight * diversity
        
        # 总损失
        total_loss = cce + combined_contrastive
        
        # 记录训练统计
        self.training_stats.append({
            'epoch': self.current_epoch,
            'lambda': current_lambda,
            'cce': cce.item(),
            'contrastive': normalized_contrastive.item(),
            'diversity': diversity.item()
        })
        
        # 保持最近100步统计
        if len(self.training_stats) > 100:
            self.training_stats.pop(0)
        
        return total_loss, cce, combined_contrastive
    
    def get_training_info(self):
        """获取训练信息用于监控"""
        return {
            'current_epoch': self.current_epoch,
            'current_lambda': self.get_current_lambda(),
            'lambda_max': self.lambda_max,
            'T_anneal': self.T_anneal,
            'annealing_progress': min(self.current_epoch / self.T_anneal, 1.0) * 100
        }

class ModelHierarchicalContrastive(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.device = device
        self.ssl_model = SSLModel(self.device)
        self.d_model = 1024
        self.group_size = getattr(args, "group_size", 3)  

        self.temporal_attn = AttnPool(in_dim=self.d_model, attn_dim=128)
        self.intra_attn = AttnPool(in_dim=self.d_model, attn_dim=128)
        self.group_refine = ResidualRefine(in_dim=self.d_model, hidden=512, dropout=0.15)
        self.inter_attn = AttnPool(in_dim=self.d_model, attn_dim=128)
        self.utt_refine = ResidualRefine(in_dim=self.d_model, hidden=512, dropout=0.15)

        self.projection_head = nn.Sequential(
            nn.Linear(self.d_model, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(256, 128)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(self.d_model, 512),
            nn.LayerNorm(512),
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
        if add_noise and self.training:
            noise_scale = 0.01
            x = x + torch.randn_like(x) * noise_scale
        
        x_ssl_feat, layerResult = self.ssl_model.extract_feat(x.squeeze(-1))
        _, fullfeature = getAttenF(layerResult)

        B, L, T, C = fullfeature.shape
        layer_tokens = fullfeature.contiguous().view(B * L, T, C)
        layer_emb, _ = self.temporal_attn(layer_tokens)
        layer_emb = layer_emb.view(B, L, C)

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

        logits = self.classifier(utt_emb)
        output = self.logsoftmax(logits)
        
        if return_features:
            projected_features = self.projection_head(utt_emb)
            return output, projected_features
        
        return output

# 类名映射
CombinedLoss = DynamicAnnealedCombinedLoss