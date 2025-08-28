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

class Model(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.device = device
        self.ssl_model = SSLModel(self.device)
        self.d_model = 1024
        self.L = 24
        self.group_size = getattr(args, "group_size", 3)

        # ✅ 1. 位置编码 - 帮助模型理解层级信息
        self.layer_pos = nn.Parameter(torch.zeros(1, self.L, self.d_model))
        
        # ✅ 2. 层门控 - 自适应选择重要层
        self.layer_gate = nn.Sequential(
            nn.Linear(self.d_model, 128), 
            nn.GELU(), 
            nn.Linear(128, 1), 
            nn.Sigmoid()
        )

        # ✅ 3. 轻量局部卷积 - 提取局部特征
        self.local_conv = nn.Conv1d(self.d_model, self.d_model, kernel_size=5, padding=2, groups=16)
        
        self.temporal_attn = AttnPool(in_dim=self.d_model, attn_dim=128)
        self.intra_attn = AttnPool(in_dim=self.d_model, attn_dim=128)
        self.group_refine = ResidualRefine(in_dim=self.d_model, hidden=512, dropout=0.1)
        self.inter_attn = AttnPool(in_dim=self.d_model, attn_dim=128)
        self.utt_refine = ResidualRefine(in_dim=self.d_model, hidden=512, dropout=0.1)

        # ✅ 4. 改进的分类器设计
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(self.d_model, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 2),
        )

    def forward(self, x, pad_mask=None):
        x_ssl_feat, layerResult = self.ssl_model.extract_feat(x.squeeze(-1))
        
        # ✅ 5. 更清晰的特征重构逻辑
        fullf = []
        for layer in layerResult:
            x_l = layer[0].transpose(0,1)         # (B,T,C)
            x_l = x_l.unsqueeze(1)                # (B,1,T,C)
            fullf.append(x_l)
        fullfeature = torch.cat(fullf, dim=1)     # (B,L,T,C)

        B, L, T, C = fullfeature.shape
        
        # ✅ 6. 局部卷积 + 时序注意力（支持mask）
        layer_tokens = fullfeature.reshape(B*L, T, C)
        tokens = self.local_conv(layer_tokens.transpose(1,2)).transpose(1,2)
        layer_emb, _ = self.temporal_attn(tokens, 
                                          mask=None if pad_mask is None else pad_mask.repeat_interleave(L, dim=0))
        layer_emb = layer_emb.view(B, L, C)

        # ✅ 7. 层位置编码与门控
        layer_emb = layer_emb + self.layer_pos[:, :L, :]
        gate = self.layer_gate(layer_emb)
        layer_emb = layer_emb * gate

        # 分组处理
        groups = torch.split(layer_emb, self.group_size, dim=1)
        group_vecs = []
        for g in groups:
            g_vec, _ = self.intra_attn(g)
            g_vec = self.group_refine(g_vec)
            group_vecs.append(g_vec)

        group_stack = torch.stack(group_vecs, dim=1)
        utt_emb, _ = self.inter_attn(group_stack)
        utt_emb = self.utt_refine(utt_emb)

        # ✅ 8. 直接输出logits（不使用LogSoftmax）
        logits = self.classifier(utt_emb)
        output = self.logsoftmax(logits)
        return output

