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

class BalancedContrastiveLoss(nn.Module):
    """
    平衡的对比学习损失 - 自动调整尺度
    """
    def __init__(self, temperature=0.07):
        super(BalancedContrastiveLoss, self).__init__()
        self.temperature = temperature
        
    def forward(self, features, labels):
        batch_size = features.size(0)
        
        if batch_size < 2:
            return torch.tensor(0.0, device=features.device)
        
        # L2归一化
        features = F.normalize(features, dim=1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # 创建正样本mask
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features.device)
        mask = mask - torch.eye(batch_size).to(features.device)
        
        # 检查正样本对
        positive_pairs = mask.sum(dim=1)
        if positive_pairs.sum() == 0:
            return torch.tensor(0.0, device=features.device)
        
        # 数值稳定性
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()
        
        # 计算对比损失
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)
        
        # 只对有正样本的样本计算损失
        valid_mask = positive_pairs > 0
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=features.device)
            
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (positive_pairs + 1e-8)
        mean_log_prob_pos = mean_log_prob_pos[valid_mask]
        
        loss = -mean_log_prob_pos.mean()
        
        # 将损失缩放到与CCE相似的范围 (0-1)
        loss = torch.clamp(loss, 0, 10) / 10
        
        return loss

class AdaptiveCombinedLoss(nn.Module):
    """
    自适应组合损失 - 动态平衡两个损失
    """
    def __init__(self, weight=None, temperature=0.07, contrastive_weight=0.1):
        super(AdaptiveCombinedLoss, self).__init__()
        self.cce_loss = nn.CrossEntropyLoss(weight=weight)
        self.contrastive_loss = BalancedContrastiveLoss(temperature=temperature)
        self.initial_contrastive_weight = contrastive_weight
        self.step_count = 0
        self.cce_history = []
        self.contrastive_history = []
        
    def forward(self, logits, features, labels):
        # 计算两个损失
        cce = self.cce_loss(logits, labels)
        contrastive = self.contrastive_loss(features, labels)
        
        # 记录历史
        self.cce_history.append(cce.item())
        self.contrastive_history.append(contrastive.item())
        
        # 保持最近20步的历史
        if len(self.cce_history) > 20:
            self.cce_history.pop(0)
            self.contrastive_history.pop(0)
        
        # 动态调整权重
        if len(self.cce_history) > 5:
            avg_cce = sum(self.cce_history[-5:]) / 5
            avg_contrastive = sum(self.contrastive_history[-5:]) / 5
            
            # 根据损失比例调整权重
            if avg_contrastive > 0:
                ratio = avg_cce / avg_contrastive
                # 目标是让两个损失贡献相当
                adaptive_weight = self.initial_contrastive_weight * ratio * 2
                adaptive_weight = torch.clamp(torch.tensor(adaptive_weight), 0.01, 1.0).item()
            else:
                adaptive_weight = self.initial_contrastive_weight
        else:
            adaptive_weight = self.initial_contrastive_weight
        
        # 组合损失
        total_loss = cce + adaptive_weight * contrastive
        
        self.step_count += 1
        
        return total_loss, cce, contrastive

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
        self.group_refine = ResidualRefine(in_dim=self.d_model, hidden=512, dropout=0.1)
        self.inter_attn = AttnPool(in_dim=self.d_model, attn_dim=128)
        self.utt_refine = ResidualRefine(in_dim=self.d_model, hidden=512, dropout=0.1)

        # 对比学习投影头 - 简化版本
        self.projection_head = nn.Sequential(
            nn.Linear(self.d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128)
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(self.d_model, 256),
            nn.SELU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(256, 2),
        )
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x, return_features=False):
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
            # 投影特征用于对比学习
            projected_features = self.projection_head(utt_emb)
            return output, projected_features
        
        return output

# 为了兼容，保持原来的类名
CombinedLoss = AdaptiveCombinedLoss