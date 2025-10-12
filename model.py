"""
Fusion Model: Combining Layer-wise Gamma Fusion + Pyramid Hierarchical Structure

核心思想:
1. 保留 Layer-wise 的 gamma 加权融合（论文核心贡献）
2. 引入 Pyramid 的自适应多尺度层次结构
3. 结合两者优势，避免各自的缺陷

Author: Lzlo
Date: October 2025
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import fairseq


class SSLModel(nn.Module):
    """XLS-R SSL Feature Extractor"""
    def __init__(self, model_path, device):  # 🔧 Fix: Correct parameter order
        super(SSLModel, self).__init__()
        
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_path])
        self.model = model[0]
        self.device = device
        self.out_dim = 1024
        
        # 🔧 Dynamic detection of actual layer count
        self.num_layers = None  # Lazy initialization
        
        # Freeze SSL model parameters
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    def extract_feat(self, input_data):
        if next(self.model.parameters()).device != input_data.device \
           or next(self.model.parameters()).dtype != input_data.dtype:
            self.model.to(input_data.device, dtype=input_data.dtype)
        
        self.model.eval()
        
        if input_data.ndim == 3:
            input_tmp = input_data[:, :, 0]
        else:
            input_tmp = input_data
        
        with torch.no_grad():
            result = self.model(input_tmp, mask=False, features_only=True)
            emb = result['x']
            layer_results = result['layer_results']
        
        # 🔧 Set actual layer count on first run
        if self.num_layers is None:
            self.num_layers = len(layer_results)
            print(f"✅ Detected {self.num_layers} layers in XLS-R model")
        
        return emb, layer_results
    
    def get_all_layer_hiddens(self, input_data) -> List[torch.Tensor]:
        """Extract hidden states from all layers"""
        _, layer_results = self.extract_feat(input_data)
        hiddens = [layer[0] for layer in layer_results]
        return hiddens


class AttnPool(nn.Module):
    """Attention Pooling Module"""
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
    """Residual Refinement Block"""
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


class GammaLayerFusion(nn.Module):
    """
    Layer-wise Gamma Fusion Module (from paper)
    可学习的层权重，端到端优化
    
    🔧 修复: 动态适应实际层数
    """
    def __init__(self, num_layers: int = 25):
        super().__init__()
        self.num_layers = num_layers
        # 可学习的gamma参数
        self.gamma = nn.Parameter(torch.randn(num_layers))
    
    def forward(self, hiddens: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            hiddens: List of [T, B, D], length = actual_num_layers
        Returns:
            fused: [B, T, D] - gamma加权融合后的特征
        """
        actual_num_layers = len(hiddens)
        T, B, D = hiddens[0].shape
        
        # 🔧 修复: 动态调整 gamma 参数以匹配实际层数
        if actual_num_layers != self.num_layers:
            print(f"⚠️  Warning: Expected {self.num_layers} layers but got {actual_num_layers}")
            print(f"   Adjusting gamma weights to match actual layers...")
            # 使用前 actual_num_layers 个 gamma 权重
            gamma_weights = self.gamma[:actual_num_layers]
        else:
            gamma_weights = self.gamma
        
        # Gamma加权融合
        gamma_prob = F.softmax(gamma_weights, dim=0)  # [L]
        stacked_hiddens = torch.stack(hiddens, dim=0)  # [L, T, B, D]
        fused = torch.einsum('l, ltbd -> tbd', gamma_prob, stacked_hiddens)  # [T, B, D]
        
        # [T, B, D] → [B, T, D]
        fused = fused.permute(1, 0, 2)
        
        return fused


class AdaptivePyramidPool(nn.Module):
    """
    自适应金字塔池化
    在时间维度上建立多尺度表示，避免在层维度分组
    """
    def __init__(self, in_dim: int, num_scales: int = 3):
        super().__init__()
        self.num_scales = num_scales
        self.in_dim = in_dim
        
        # 多尺度池化 - 不同的时间窗口大小
        self.pool_sizes = [2 ** (i + 1) for i in range(num_scales)]
        
        # 每个尺度的attention
        self.scale_attns = nn.ModuleList([
            AttnPool(in_dim=in_dim, attn_dim=128) 
            for _ in range(num_scales)
        ])
        
        # 尺度融合
        self.scale_fusion = nn.Linear(in_dim * num_scales, in_dim)
        self.scale_norm = nn.LayerNorm(in_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D]
        Returns:
            multi_scale: [B, D] - 多尺度融合特征
        """
        B, T, D = x.shape
        scale_feats = []
        
        for pool_size, attn in zip(self.pool_sizes, self.scale_attns):
            # 自适应窗口划分
            if T < pool_size:
                feat, _ = attn(x)
            else:
                num_windows = T // pool_size
                remainder = T % pool_size
                
                if remainder > 0:
                    pad_size = pool_size - remainder
                    x_padded = F.pad(x, (0, 0, 0, pad_size))
                    num_windows = (T + pad_size) // pool_size
                else:
                    x_padded = x
                
                x_windowed = x_padded.view(B, num_windows, pool_size, D)
                
                window_feats = []
                for i in range(num_windows):
                    window = x_windowed[:, i, :, :]
                    w_feat, _ = attn(window)
                    window_feats.append(w_feat)
                
                feat = torch.stack(window_feats, dim=1).mean(dim=1)
            
            scale_feats.append(feat)
        
        multi_scale = torch.cat(scale_feats, dim=-1)
        multi_scale = self.scale_fusion(multi_scale)
        multi_scale = self.scale_norm(multi_scale)
        
        return multi_scale


class FusionBestModel(nn.Module):
    """
    融合最佳模型
    
    🔧 修复: 动态适应XLS-R实际层数
    """
    
    def __init__(self, args, device):
        super().__init__()
        self.device = device
        
        # 🔧 Fix: Initialize attributes from args FIRST
        self.low_dim = getattr(args, 'low_dim', 256)
        self.num_scales = getattr(args, 'num_scales', 3)
        
        # Load SSL model
        cp_path = '/root/autodl-tmp/SLSforASVspoof-2021-DF/xlsr2_300m.pt'
        self.ssl_model = SSLModel(cp_path, device)  # 🔧 Correct argument order
        self.LL = self.ssl_model.out_dim
        self.d_model = self.LL  # 1024
        
        # 🔧 Will be set dynamically on first forward pass
        self.num_layers = 25  # Initial estimate
        
        # Layer-wise Gamma Fusion
        self.gamma_fusion = GammaLayerFusion(num_layers=self.num_layers)
        
        # Projection
        self.proj = nn.Linear(self.d_model, self.low_dim)
        self.proj_norm = nn.LayerNorm(self.low_dim)
        
        # Adaptive Pyramid
        self.pyramid_pool = AdaptivePyramidPool(
            in_dim=self.low_dim, 
            num_scales=self.num_scales
        )
        
        # Refinement
        self.refine = ResidualRefine(
            in_dim=self.low_dim, 
            hidden=self.low_dim * 2, 
            dropout=0.1
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.low_dim),
            nn.Linear(self.low_dim, 128),
            nn.SELU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, 2),
        )
        self.logsoftmax = nn.LogSoftmax(dim=1)
        
        # Track if layers have been adjusted
        self._layers_adjusted = False
    
    def forward(self, x):
        """
        Args:
            x: input audio [B, T] or [B, T, 1]
        Returns:
            output: log probabilities [B, 2]
        """
        # Extract all layer features
        hiddens = self.ssl_model.get_all_layer_hiddens(x.squeeze(-1))
        
        # 🔧 Adjust gamma layer count on first forward pass
        if not self._layers_adjusted and self.ssl_model.num_layers is not None:
            actual_layers = self.ssl_model.num_layers
            if actual_layers != self.num_layers:
                print(f"🔧 Adjusting model for {actual_layers} layers (was {self.num_layers})")
                self.num_layers = actual_layers
                # Reinitialize gamma_fusion to match actual layers
                self.gamma_fusion = GammaLayerFusion(num_layers=actual_layers).to(self.device)
            self._layers_adjusted = True
        
        # Gamma Layer Fusion
        fused = self.gamma_fusion(hiddens)  # [B, T, 1024]
        
        # Projection
        fused_proj = self.proj(fused)
        fused_proj = self.proj_norm(fused_proj)
        
        # Adaptive Pyramid Pooling
        multi_scale_emb = self.pyramid_pool(fused_proj)
        
        # Refinement
        refined_emb = self.refine(multi_scale_emb)
        
        # Classification
        logits = self.classifier(refined_emb)
        output = self.logsoftmax(logits)
        
        return output


# Alias for compatibility
Model = FusionBestModel


def get_model(args, device):
    """Factory function to create the fusion model"""
    return FusionBestModel(args, device)