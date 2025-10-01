import random
import sys
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import fairseq
# 添加mamba导入
from mamba_ssm import Mamba

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

class Model(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.device = device
        self.ssl_model = SSLModel(self.device)
        self.d_model = 1024
        self.group_size = getattr(args, "group_size", 3)  

        # ✅ 添加层门控 - 自适应选择重要的SSL层
        self.layer_gate = nn.Sequential(
            nn.Linear(self.d_model, 64),  # 保守的隐藏层大小
            nn.SELU(inplace=True),        # 保持与现有架构一致
            nn.Dropout(0.1),              # 轻度正则化
            nn.Linear(64, 1),
            nn.Sigmoid()                  # 输出0-1的门控权重
        )

        self.temporal_attn = AttnPool(in_dim=self.d_model, attn_dim=128)
        self.intra_attn = AttnPool(in_dim=self.d_model, attn_dim=128)
        self.group_refine = ResidualRefine(in_dim=self.d_model, hidden=512, dropout=0.1)
        
        # ✅ 替换inter_attn为Mamba
        self.mamba_block = Mamba(
            d_model=self.d_model,      # Model dimension
            d_state=16,                # SSM state expansion factor
            d_conv=4,                  # Local convolution width  
            expand=2,                  # Block expansion factor
        )
        
        # 添加一个线性层来聚合序列
        self.sequence_aggregator = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.SELU(inplace=True),
            nn.AdaptiveAvgPool1d(1)  # 将序列维度聚合为1
        )
        
        self.utt_refine = ResidualRefine(in_dim=self.d_model, hidden=512, dropout=0.1)

        # Add more regularization to prevent overfitting
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.SELU(inplace=True),
            nn.Dropout(p=0.2),  # Increased dropout
            nn.Linear(self.d_model, 256),
            nn.BatchNorm1d(256),  # Add batch normalization
            nn.SELU(inplace=True),
            nn.Dropout(p=0.2),  # Increased dropout
            nn.Linear(256, 2),
        )
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x_ssl_feat, layerResult = self.ssl_model.extract_feat(x.squeeze(-1))
        _, fullfeature = getAttenF(layerResult)  # (B, L, T, C)

        B, L, T, C = fullfeature.shape
        layer_tokens = fullfeature.contiguous().view(B * L, T, C)
        layer_emb, _ = self.temporal_attn(layer_tokens)
        layer_emb = layer_emb.view(B, L, C)

        # ✅ 应用层门控 - 对每层特征进行加权
        gate_weights = self.layer_gate(layer_emb)  # (B, L, 1)
        layer_emb = layer_emb * gate_weights       # 元素级乘法

        groups = torch.split(layer_emb, self.group_size, dim=1)
        group_vecs = []
        for g in groups:
            g_vec, _ = self.intra_attn(g)
            g_vec = self.group_refine(g_vec)
            group_vecs.append(g_vec)

        group_stack = torch.stack(group_vecs, dim=1)  # (B, num_groups, C)
        
        # ✅ 使用Mamba替代inter_attn处理组间关系
        # Mamba期望输入形状为(B, L, D)，这里group_stack已经是正确的形状
        mamba_out = self.mamba_block(group_stack)  # (B, num_groups, C)
        
        # 聚合序列维度得到utterance级别的embedding
        mamba_out_transposed = mamba_out.transpose(1, 2)  # (B, C, num_groups)
        utt_emb = self.sequence_aggregator(mamba_out_transposed).squeeze(-1)  # (B, C)
        
        utt_emb = self.utt_refine(utt_emb)

        logits = self.classifier(utt_emb)
        output = self.logsoftmax(logits)
        return output