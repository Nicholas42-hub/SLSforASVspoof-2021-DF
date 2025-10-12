"""
Hierarchical Layer Attention Model (Simplified)

核心思路:
1. XLS-R提取25层特征
2. Multi-head扩展: 25层 → 100 tokens
3. 三阶段Transformer降维: 100 → 50 → 25
4. 时间+层特征融合
5. 分类

Author: Lzlo
Date: October 2025
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import fairseq


class SSLModel(nn.Module):
    """XLS-R特征提取器"""
    def __init__(self, model_path, device):
        super().__init__()
        
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_path])
        self.model = model[0]
        self.device = device
        self.out_dim = 1024
        self.num_layers = None  # 动态检测
        
        # 冻结SSL参数
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    def extract_feat(self, input_data):
        if next(self.model.parameters()).device != input_data.device:
            self.model.to(input_data.device, dtype=input_data.dtype)
        
        self.model.eval()
        input_tmp = input_data[:, :, 0] if input_data.ndim == 3 else input_data
        
        with torch.no_grad():
            result = self.model(input_tmp, mask=False, features_only=True)
            layer_results = result['layer_results']
        
        if self.num_layers is None:
            self.num_layers = len(layer_results)
            print(f"✅ Detected {self.num_layers} XLS-R layers")
        
        return [layer[0] for layer in layer_results]  # List of [T, B, D]


class MultiHeadExpansion(nn.Module):
    """
    多头层扩展: 25层 → 100 tokens
    每个头学习不同的层间关系
    """
    def __init__(self, in_dim=1024, out_dim=256, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        
        # 每个头独立投影
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            )
            for _ in range(num_heads)
        ])
        
    def forward(self, layer_hiddens: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            layer_hiddens: List[Tensor], each [T, B, 1024], length=25
        Returns:
            tokens: [B, 100, 256]
        """
        # Stack: [25, T, B, D]
        stacked = torch.stack(layer_hiddens, dim=0)
        
        # Time pooling: [25, T, B, D] → [25, B, D]
        pooled = stacked.mean(dim=1)
        
        # Multi-head projection
        head_outs = []
        for head in self.heads:
            out = head(pooled)  # [25, B, 256]
            head_outs.append(out)
        
        # Stack: [4, 25, B, 256]
        multi_head = torch.stack(head_outs, dim=0)
        
        # Reshape: [100, B, 256]
        nh, L, B, D = multi_head.shape
        multi_head = multi_head.reshape(nh * L, B, D)
        
        # Transpose: [B, 100, 256]
        return multi_head.permute(1, 0, 2)


class AttnPooling(nn.Module):
    """注意力池化: [B, T, D] → [B, D]"""
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.query = nn.Parameter(torch.randn(1, 1, dim))
        
    def forward(self, x):
        """x: [B, T, D] → [B, D]"""
        B = x.size(0)
        q = self.query.expand(B, -1, -1)  # [B, 1, D]
        out, _ = self.attn(q, x, x)  # [B, 1, D]
        return out.squeeze(1)  # [B, D]


class HierarchicalLayerModel(nn.Module):
    """
    简化版层次化注意力模型
    
    Pipeline:
    1. 25层XLS-R → Multi-head扩展 → 100 tokens (256d)
    2. Stage1: 100 → 50 tokens (512d)
    3. Stage2: 50 → 25 tokens (1024d)
    4. 时间特征提取 + 融合
    5. 分类
    """
    
    def __init__(self, args, device):
        super().__init__()
        self.device = device
        
        # 超参数
        self.embed_dim = getattr(args, 'embed_dim', 256)
        self.num_heads_expand = getattr(args, 'num_heads_expand', 4)
        
        # SSL模型
        ssl_path = getattr(args, 'ssl_path', 
                          '/root/autodl-tmp/SLSforASVspoof-2021-DF/xlsr2_300m.pt')
        self.ssl = SSLModel(ssl_path, device)
        
        # === 层扩展 ===
        self.expand = MultiHeadExpansion(
            in_dim=1024,
            out_dim=self.embed_dim,
            num_heads=self.num_heads_expand  # 25 → 100
        )
        
        # === 三阶段降维 ===
        # Stage 1: 100 tokens, dim=256
        self.stage1 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.embed_dim,
                nhead=4,
                dim_feedforward=self.embed_dim * 4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=2
        )
        
        # 🔧 修复: 添加维度投影层 256 → 512
        self.proj1to2 = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 2),
            nn.LayerNorm(self.embed_dim * 2),
            nn.GELU()
        )
        
        # Stage 2: 50 tokens, dim=512
        self.stage2 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.embed_dim * 2,
                nhead=8,
                dim_feedforward=self.embed_dim * 8,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=2
        )
        
        # 🔧 修复: 添加维度投影层 512 → 1024
        self.proj2to3 = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.embed_dim * 4),
            nn.LayerNorm(self.embed_dim * 4),
            nn.GELU()
        )
        
        # Stage 3: 25 tokens, dim=1024
        self.stage3 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.embed_dim * 4,
                nhead=16,
                dim_feedforward=self.embed_dim * 16,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=2
        )
        
        # === 时间特征提取 ===
        self.time_proj = nn.Linear(1024, self.embed_dim)
        self.time_pool = AttnPooling(self.embed_dim, num_heads=4)
        
        # === 特征融合 ===
        fusion_dim = self.embed_dim * 4 + self.embed_dim  # 1024 + 256 = 1280
        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 512)
        )
        
        # === 分类器 ===
        self.classifier = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )
        
        self.logsoftmax = nn.LogSoftmax(dim=1)
    
    def extract_time_feat(self, hiddens: List[torch.Tensor]) -> torch.Tensor:
        """
        从中间层提取时间特征
        Args:
            hiddens: List of [T, B, 1024]
        Returns:
            time_feat: [B, 256]
        """
        mid_idx = len(hiddens) // 2
        mid_layer = hiddens[mid_idx]  # [T, B, 1024]
        
        # [T, B, D] → [B, T, D]
        mid_layer = mid_layer.permute(1, 0, 2)
        
        # 投影并池化
        projected = self.time_proj(mid_layer)  # [B, T, 256]
        pooled = self.time_pool(projected)  # [B, 256]
        
        return pooled
    
    def forward(self, x):
        """
        Args:
            x: [B, T] or [B, T, 1]
        Returns:
            log_probs: [B, 2]
        """
        # 提取所有层
        hiddens = self.ssl.extract_feat(x.squeeze(-1))  # List of [T, B, 1024]
        
        # === 层特征处理 ===
        # Multi-head扩展: 25层 → 100 tokens
        layer_tokens = self.expand(hiddens)  # [B, 100, 256]
        
        # Stage 1: 100 tokens, dim=256
        x1 = self.stage1(layer_tokens)  # [B, 100, 256]
        
        # 降采样: 100 → 50 tokens (在序列维度)
        x1_pool = F.adaptive_avg_pool1d(
            x1.transpose(1, 2), 50
        ).transpose(1, 2)  # [B, 50, 256]
        
        # 🔧 修复: 先升维 256 → 512，再进入 stage2
        x1_up = self.proj1to2(x1_pool)  # [B, 50, 512]
        
        # Stage 2: 50 tokens, dim=512
        x2 = self.stage2(x1_up)  # [B, 50, 512]
        
        # 降采样: 50 → 25 tokens
        x2_pool = F.adaptive_avg_pool1d(
            x2.transpose(1, 2), 25
        ).transpose(1, 2)  # [B, 25, 512]
        
        # 🔧 修复: 先升维 512 → 1024，再进入 stage3
        x2_up = self.proj2to3(x2_pool)  # [B, 25, 1024]
        
        # Stage 3: 25 tokens, dim=1024
        x3 = self.stage3(x2_up)  # [B, 25, 1024]
        
        # 全局池化
        layer_feat = x3.mean(dim=1)  # [B, 1024]
        
        # === 时间特征 ===
        time_feat = self.extract_time_feat(hiddens)  # [B, 256]
        
        # === 融合 ===
        combined = torch.cat([layer_feat, time_feat], dim=-1)  # [B, 1280]
        fused = self.fusion(combined)  # [B, 512]
        
        # === 分类 ===
        logits = self.classifier(fused)  # [B, 2]
        
        return self.logsoftmax(logits)


# Alias
Model = HierarchicalLayerModel


def get_model(args, device):
    return HierarchicalLayerModel(args, device)


# ===== 测试代码 =====
if __name__ == "__main__":
    import argparse
    
    args = argparse.Namespace()
    args.embed_dim = 256
    args.num_heads_expand = 4
    args.ssl_path = '/root/autodl-tmp/SLSforASVspoof-2021-DF/xlsr2_300m.pt'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = get_model(args, device).to(device)
    
    # 测试
    x = torch.randn(2, 64000).to(device)
    out = model(x)
    print(f"✅ Output: {out.shape}")  # [2, 2]
    
    # 参数量
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Parameters: {params:,}")