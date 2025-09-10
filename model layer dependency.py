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

class AdaptiveGroupingModule(nn.Module):
    """依赖驱动的自适应分组模块"""
    def __init__(self, num_layers: int, d_model: int, num_groups: int = 4):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_groups = num_groups
        
        # 学习软分组矩阵 S ∈ R^{L×G}
        self.group_logits = nn.Parameter(torch.randn(num_layers, num_groups))
        
        # 层依赖建模：计算层间相似度
        self.dependency_proj = nn.Linear(d_model, 256)
        
        # 组内注意力
        self.intra_group_attn = AttnPool(in_dim=d_model, attn_dim=128)
        
        # 组间注意力
        self.inter_group_attn = AttnPool(in_dim=d_model, attn_dim=128)
        
        # 分组细化
        self.group_refine = ResidualRefine(in_dim=d_model, hidden=512, dropout=0.1)
        
    def compute_layer_dependencies(self, layer_embs):
        """计算层间依赖关系"""
        # layer_embs: (B, L, C)
        B, L, C = layer_embs.shape
        
        # 投影到低维空间计算相似度
        proj_embs = self.dependency_proj(layer_embs)  # (B, L, 256)
        
        # 计算层间余弦相似度
        normalized = F.normalize(proj_embs, p=2, dim=-1)
        similarity = torch.bmm(normalized, normalized.transpose(1, 2))  # (B, L, L)
        
        return similarity
    
    def compute_soft_grouping(self, layer_dependencies):
        """基于层依赖计算软分组矩阵"""
        B = layer_dependencies.shape[0]
        
        # 基础分组logits
        base_logits = self.group_logits.unsqueeze(0).expand(B, -1, -1)  # (B, L, G)
        
        # 使用层依赖调整分组
        # 相似的层应该被分到同一组
        dep_mean = layer_dependencies.mean(dim=-1, keepdim=True)  # (B, L, 1)
        dependency_bias = dep_mean.expand(-1, -1, self.num_groups) * 0.1
        
        adjusted_logits = base_logits + dependency_bias
        
        # 软分组矩阵 (温度控制软硬程度)
        soft_groups = F.softmax(adjusted_logits / 0.5, dim=-1)  # (B, L, G)
        
        return soft_groups
    
    def apply_regularization(self, soft_groups, layer_dependencies):
        """应用正则化约束"""
        B, L, G = soft_groups.shape
        
        # 1. 低熵正则（鼓励明确分组）
        entropy = -torch.sum(soft_groups * torch.log(soft_groups + 1e-8), dim=-1)  # (B, L)
        entropy_loss = entropy.mean()
        
        # 2. 组间正交正则（鼓励多样性）
        group_centers = torch.bmm(soft_groups.transpose(1, 2), layer_dependencies)  # (B, G, L)
        group_centers_norm = F.normalize(group_centers, p=2, dim=-1)
        orthogonal_matrix = torch.bmm(group_centers_norm, group_centers_norm.transpose(1, 2))  # (B, G, G)
        identity = torch.eye(G, device=soft_groups.device).unsqueeze(0).expand(B, -1, -1)
        orthogonal_loss = F.mse_loss(orthogonal_matrix, identity)
        
        # 3. 层序平滑正则（鼓励连续分组）
        smoothness_loss = 0
        for g in range(G):
            group_weights = soft_groups[:, :, g]  # (B, L)
            # 相邻层的分组权重应该平滑变化
            diff = group_weights[:, 1:] - group_weights[:, :-1]
            smoothness_loss += torch.mean(diff ** 2)
        smoothness_loss /= G
        
        total_reg_loss = 0.1 * entropy_loss + 0.1 * orthogonal_loss + 0.05 * smoothness_loss
        
        return total_reg_loss
    
    def forward(self, layer_embs):
        """
        Args:
            layer_embs: (B, L, C) - 层表示
        Returns:
            group_output: (B, C) - 最终分组表示
            soft_groups: (B, L, G) - 软分组矩阵
            reg_loss: 正则化损失
        """
        B, L, C = layer_embs.shape
        
        # 1. 计算层依赖
        layer_dependencies = self.compute_layer_dependencies(layer_embs)
        
        # 2. 计算软分组
        soft_groups = self.compute_soft_grouping(layer_dependencies)
        
        # 3. 计算正则化损失
        reg_loss = self.apply_regularization(soft_groups, layer_dependencies)
        
        # 4. 软分组聚合
        group_representations = []
        for g in range(self.num_groups):
            # 使用软权重聚合该组的层表示
            group_weights = soft_groups[:, :, g].unsqueeze(-1)  # (B, L, 1)
            weighted_layers = layer_embs * group_weights  # (B, L, C)
            
            # 组内注意力聚合
            group_repr, _ = self.intra_group_attn(weighted_layers)  # (B, C)
            group_repr = self.group_refine(group_repr)
            group_representations.append(group_repr)
        
        # 5. 组间注意力
        group_stack = torch.stack(group_representations, dim=1)  # (B, G, C)
        final_repr, _ = self.inter_group_attn(group_stack)  # (B, C)
        
        return final_repr, soft_groups, reg_loss

class Model(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.device = device
        self.ssl_model = SSLModel(self.device)
        self.d_model = 1024
        self.num_groups = getattr(args, "num_groups", 4)
        
        # 时序注意力（处理时间维度）
        self.temporal_attn = AttnPool(in_dim=self.d_model, attn_dim=128)
        
        # 自适应分组模块
        # 假设XLSR模型有24层
        self.adaptive_grouping = AdaptiveGroupingModule(
            num_layers=24, 
            d_model=self.d_model, 
            num_groups=self.num_groups
        )
        
        # 最终细化
        self.utt_refine = ResidualRefine(in_dim=self.d_model, hidden=512, dropout=0.1)

        # 分类器
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.SELU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(self.d_model, 256),
            nn.BatchNorm1d(256),
            nn.SELU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, 2),
        )
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # 1. SSL特征提取
        x_ssl_feat, layerResult = self.ssl_model.extract_feat(x.squeeze(-1))
        _, fullfeature = getAttenF(layerResult)  # (B, L, T, C)

        B, L, T, C = fullfeature.shape
        
        # 2. 时序维度聚合
        layer_tokens = fullfeature.contiguous().view(B * L, T, C)
        layer_emb, _ = self.temporal_attn(layer_tokens)  # (B*L, C)
        layer_emb = layer_emb.view(B, L, C)  # (B, L, C)

        # 3. 依赖驱动的自适应分组
        utt_emb, soft_groups, reg_loss = self.adaptive_grouping(layer_emb)
        
        # 4. 最终细化
        utt_emb = self.utt_refine(utt_emb)

        # 5. 分类
        logits = self.classifier(utt_emb)
        output = self.logsoftmax(logits)
        
        # 返回输出和正则化损失（用于训练时的loss计算）
        return {
            'output': output,
            'reg_loss': reg_loss,
            'soft_groups': soft_groups  # 可用于可视化分组热力图
        }
    
    def get_grouping_heatmap(self, x):
        """获取分组热力图用于可视化"""
        with torch.no_grad():
            result = self.forward(x)
            return result['soft_groups']  # (B, L, G)

