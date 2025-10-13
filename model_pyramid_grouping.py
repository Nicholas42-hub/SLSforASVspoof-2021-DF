import torch
import torch.nn as nn
import torch.nn.functional as F
import fairseq
from typing import List, Tuple, Optional
import math


class SSLModel(nn.Module):
    """SSL feature extractor wrapper"""
    def __init__(self, cp_path: str, device: torch.device):
        super().__init__()
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
            
        result = self.model(input_tmp, mask=False, features_only=True)
        emb = result['x']
        layer_results = result['layer_results']
        return emb, layer_results



class PositionalEncoding(nn.Module):
    """Learnable positional encoding for layer ordering"""
    def __init__(self, num_layers: int, d_model: int):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.randn(1, num_layers, d_model) * 0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        if L != self.pos_emb.size(1):
            pos_emb = self.pos_emb[:, :L, :]
        else:
            pos_emb = self.pos_emb
        return x + pos_emb


class EfficientAttnPool(nn.Module):
    def __init__(self, in_dim: int, attn_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.scale = attn_dim ** -0.5
        self.q = nn.Linear(in_dim, attn_dim)
        self.k = nn.Linear(in_dim, attn_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        q = self.q(x.mean(dim=1, keepdim=True))  # [B, 1, attn_dim]
        k = self.k(x)  # [B, T, attn_dim]
        
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        scores = scores.squeeze(1)
        
        if mask is not None:
            scores = scores.masked_fill(~mask.bool(), float('-inf'))
        
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.sum(attn.unsqueeze(-1) * x, dim=1)
        return out, attn


class AdaptiveGrouping(nn.Module):
    """
    🔧 简化版自适应分组
    - 移除温度 clamp
    - 使用 Gumbel-Softmax 进行更锐利的分组
    """
    def __init__(self, num_groups: int, d_model: int):
        super().__init__()
        self.num_groups = num_groups
        # 🔧 使用正交初始化，确保初始分组更清晰
        centers = torch.empty(num_groups, d_model)
        nn.init.orthogonal_(centers)
        self.group_centers = nn.Parameter(centers)
        # 🔧 温度初始化为 0.5（更激进）
        self.temperature = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, layer_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, D = layer_emb.shape
        
        # Compute similarity
        layer_emb_norm = F.normalize(layer_emb, dim=-1)
        centers_norm = F.normalize(self.group_centers, dim=-1)
        sim = torch.matmul(layer_emb_norm, centers_norm.T)  # [B, L, num_groups]
        
        # 🔧 移除 clamp，允许温度自由学习
        # 使用 Gumbel-Softmax 进行更锐利的分组
        if self.training:
            # Gumbel noise for sharp assignment during training
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(sim) + 1e-8) + 1e-8)
            assignments = F.softmax((sim + gumbel_noise) / self.temperature.abs(), dim=-1)
        else:
            assignments = F.softmax(sim / self.temperature.abs(), dim=-1)
        
        # Weighted aggregation
        grouped = torch.matmul(assignments.transpose(1, 2), layer_emb)
        
        return grouped, assignments


class ResidualAttention(nn.Module):
    """Attention block with layer normalization"""
    def __init__(self, in_dim: int, attn_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.attn = EfficientAttnPool(in_dim, attn_dim, dropout)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        normed = self.norm(x)
        pooled, weights = self.attn(normed)
        return pooled, weights


class SimplifiedTemporalConv(nn.Module):
    """🔧 简化版时序卷积 - 单个深度可分离卷积"""
    def __init__(self, in_dim: int):
        super().__init__()
        # Depthwise separable conv (更高效)
        self.depthwise = nn.Conv1d(in_dim, in_dim, kernel_size=3, padding=1, groups=in_dim)
        self.pointwise = nn.Conv1d(in_dim, in_dim, kernel_size=1)
        self.norm = nn.LayerNorm(in_dim)
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, C]"""
        x_t = x.transpose(1, 2)  # [B, C, T]
        out = self.depthwise(x_t)
        out = self.pointwise(out)
        out = out.transpose(1, 2)  # [B, T, C]
        return self.activation(self.norm(out))


class AdaptiveGroupingHierarchicalModel(nn.Module):
    """
    🔧 优化后的自适应分组层次模型
    主要改进:
    1. 减少分组数 (8 -> 4)
    2. 简化时序处理
    3. 添加 warmup 机制
    4. 更强的残差连接
    """
    def __init__(self, args, device: torch.device):
        super().__init__()
        self.device = device
        
        # SSL model
        cp_path = getattr(args, 'ssl_checkpoint', 
                         '/root/autodl-tmp/SLSforASVspoof-2021-DF/xlsr2_300m.pt')
        self.ssl_model = SSLModel(cp_path, device)
        self.d_model = self.ssl_model.out_dim
        
        # 🔧 Configuration: 减少分组数
        self.num_groups = getattr(args, "num_groups", 4)  # 从 8 改为 4
        self.use_multiscale = getattr(args, "use_multiscale", True)
        
        # 🔧 简化时序处理
        if self.use_multiscale:
            self.temporal_conv = SimplifiedTemporalConv(self.d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(num_layers=30, d_model=self.d_model)
        
        # Hierarchical attention
        self.temporal_attn = ResidualAttention(self.d_model, attn_dim=256, dropout=0.1)
        
        # 🔧 Adaptive grouping with improved initialization
        self.adaptive_grouping = AdaptiveGrouping(
            num_groups=self.num_groups, 
            d_model=self.d_model
        )
        
        self.intra_attn = ResidualAttention(self.d_model, attn_dim=256, dropout=0.1)
        self.inter_attn = ResidualAttention(self.d_model, attn_dim=256, dropout=0.1)
        
        # 🔧 更简单的 refinement
        self.group_refine = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        
        self.utt_refine = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        
        # Pre-classifier normalization
        self.pre_classifier_norm = nn.LayerNorm(self.d_model)
        
        # 🔧 Classifier with stronger dropout
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, 512),
            nn.GELU(),
            nn.Dropout(0.3),  # 从 0.2 增加到 0.3
            nn.Linear(512, 2),
        )
        
        # Weight initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, return_attention: bool = False):
        assert x.ndim in [2, 3], f"Expected 2D or 3D input, got {x.ndim}D"
        
        # Extract SSL features
        _, layer_results = self.ssl_model.extract_feat(x.squeeze(-1))
        
        # Convert to tensors
        layer_features = []
        for hidden, _ in layer_results:
            layer_features.append(hidden.transpose(0, 1))  # [T, B, D] -> [B, T, D]
        
        layer_stack = torch.stack(layer_features, dim=1)  # [B, L, T, D]
        B, L, T, D = layer_stack.shape
        
        # ====== Level 1: Temporal Processing ======
        if self.use_multiscale:
            # 🔧 处理所有层，但使用更高效的卷积
            layer_emb_list = []
            for l in range(L):
                # 时序建模
                temporal_feat = self.temporal_conv(layer_stack[:, l])  # [B, T, D]
                # 池化
                pooled, _ = self.temporal_attn(temporal_feat)  # [B, D]
                layer_emb_list.append(pooled)
            layer_emb = torch.stack(layer_emb_list, dim=1)  # [B, L, D]
        else:
            # 直接池化（最快）
            layer_tokens = layer_stack.reshape(B * L, T, D)
            layer_pooled, _ = self.temporal_attn(layer_tokens)
            layer_emb = layer_pooled.reshape(B, L, D)
        
        # Add positional encoding
        layer_emb = self.pos_encoding(layer_emb)  # [B, L, D]
        
        # ====== Level 2: Adaptive Grouping ======
        grouped_emb, group_assignments = self.adaptive_grouping(layer_emb)  # [B, num_groups, D]
        
        # Intra-group attention
        group_vecs = []
        for g_idx in range(self.num_groups):
            g_vec, _ = self.intra_attn(grouped_emb[:, g_idx:g_idx+1, :])
            # 🔧 更强的残差连接
            g_vec = g_vec + 0.5 * self.group_refine(g_vec)  # 缩放残差
            group_vecs.append(g_vec)
        
        # ====== Level 3: Inter-group Attention ======
        group_stack = torch.stack(group_vecs, dim=1)  # [B, num_groups, D]
        utt_emb, inter_weights = self.inter_attn(group_stack)
        # 🔧 更强的残差连接
        utt_emb = utt_emb + 0.5 * self.utt_refine(utt_emb)
        
        # ====== Classification ======
        utt_emb = self.pre_classifier_norm(utt_emb)
        logits = self.classifier(utt_emb)
        output = F.log_softmax(logits, dim=-1)
        
        if return_attention:
            attention_dict = {
                'group_assignments': group_assignments,
                'inter_weights': inter_weights,
                'num_layers': L,
                'num_groups': self.num_groups,
                'temperature': self.adaptive_grouping.temperature.item(),  # 🔧 监控温度
            }
            return output, attention_dict
        
        return output








