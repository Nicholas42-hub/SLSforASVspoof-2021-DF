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
        self.device = device
        self.out_dim = self.model.cfg.encoder_embed_dim  # Dynamic dimension detection
        
        # Freeze SSL model
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    @torch.no_grad()
    def extract_feat(self, input_data: torch.Tensor) -> Tuple[torch.Tensor, List[Tuple]]:
        """Extract features from all transformer layers"""
        self.model.eval()
        
        # Handle different input shapes
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
        """
        Args:
            x: [B, L, D]
        Returns:
            x with positional encoding: [B, L, D]
        """
        B, L, D = x.shape
        # Dynamically handle variable number of layers
        if L != self.pos_emb.size(1):
            # Interpolate or slice positional embeddings
            pos_emb = self.pos_emb[:, :L, :]
        else:
            pos_emb = self.pos_emb
        return x + pos_emb


class EfficientAttnPool(nn.Module):
    """
    Efficient attention pooling with:
    - Scaled dot-product attention
    - Entropy regularization support
    - Optional dropout
    """
    def __init__(self, in_dim: int, attn_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.scale = attn_dim ** -0.5
        self.q = nn.Linear(in_dim, attn_dim)
        self.k = nn.Linear(in_dim, attn_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, T, C]
            mask: [B, T] optional attention mask
        Returns:
            out: [B, C] pooled features
            attn: [B, T] attention weights
        """
        q = self.q(x.mean(dim=1, keepdim=True))  # [B, 1, attn_dim]
        k = self.k(x)  # [B, T, attn_dim]
        
        # Scaled attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, 1, T]
        scores = scores.squeeze(1)  # [B, T]
        
        if mask is not None:
            scores = scores.masked_fill(~mask.bool(), float('-inf'))
        
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.sum(attn.unsqueeze(-1) * x, dim=1)  # [B, C]
        return out, attn


class MultiScaleFeatureFusion(nn.Module):
    """
    Multi-scale feature fusion to preserve temporal information
    - Uses dilated convolutions at different rates
    - Replaces aggressive temporal pooling
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.scales = nn.ModuleList([
            nn.Conv1d(in_dim, out_dim // 4, kernel_size=3, dilation=1, padding=1),
            nn.Conv1d(in_dim, out_dim // 4, kernel_size=3, dilation=2, padding=2),
            nn.Conv1d(in_dim, out_dim // 4, kernel_size=3, dilation=4, padding=4),
            nn.Conv1d(in_dim, out_dim // 4, kernel_size=3, dilation=8, padding=8),
        ])
        self.norm = nn.LayerNorm(out_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, C]
        Returns:
            out: [B, T, C] multi-scale fused features
        """
        x_t = x.transpose(1, 2)  # [B, C, T]
        features = [conv(x_t) for conv in self.scales]
        out = torch.cat(features, dim=1)  # [B, C, T]
        out = out.transpose(1, 2)  # [B, T, C]
        return self.norm(out)


class AdaptiveGrouping(nn.Module):
    """
    Learnable grouping mechanism instead of fixed splitting
    - Uses soft assignment based on layer similarities
    - Temperature-controlled softmax for assignment sharpness
    """
    def __init__(self, num_groups: int, d_model: int):
        super().__init__()
        self.num_groups = num_groups
        self.group_centers = nn.Parameter(torch.randn(num_groups, d_model))
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, layer_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            layer_emb: [B, L, D]
        Returns:
            grouped: [B, num_groups, D] grouped layer representations
            assignments: [B, L, num_groups] soft assignment weights
        """
        B, L, D = layer_emb.shape
        
        # Compute similarity to group centers
        layer_emb_norm = F.normalize(layer_emb, dim=-1)
        centers_norm = F.normalize(self.group_centers, dim=-1)
        sim = torch.matmul(layer_emb_norm, centers_norm.T)  # [B, L, num_groups]
        
        # Soft assignment with temperature
        assignments = F.softmax(sim / self.temperature.clamp(min=0.1), dim=-1)  # [B, L, num_groups]
        
        # Weighted aggregation
        grouped = torch.matmul(assignments.transpose(1, 2), layer_emb)  # [B, num_groups, D]
        
        return grouped, assignments


class ResidualAttention(nn.Module):
    """Attention block with layer normalization"""
    def __init__(self, in_dim: int, attn_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.attn = EfficientAttnPool(in_dim, attn_dim, dropout)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, T, D] - can represent temporal, layer, or group dimension
        Returns:
            out: [B, D] pooled representation
            weights: [B, T] attention weights
        """
        normed = self.norm(x)
        pooled, weights = self.attn(normed)
        return pooled, weights


class LightweightTemporalConv(nn.Module):
    """轻量级时序卷积 - 只用一个膨胀卷积"""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        # 只保留一个中等感受野的卷积
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size=3, dilation=2, padding=2)
        self.norm = nn.LayerNorm(out_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_t = x.transpose(1, 2)  # [B, C, T]
        out = self.conv(x_t)  # [B, C, T]
        out = out.transpose(1, 2)  # [B, T, C]
        return self.norm(out)


class AdaptiveGroupingHierarchicalModel(nn.Module):
    """
    Improved hierarchical attention model with:
    1. Multi-scale temporal feature extraction
    2. Learnable adaptive grouping
    3. Residual connections throughout
    4. Dynamic positional encoding for layer ordering
    5. Entropy regularization for attention sparsity
    """
    def __init__(self, args, device: torch.device):
        super().__init__()
        self.device = device
        
        # SSL model with configurable path
        cp_path = getattr(args, 'ssl_checkpoint', 
                         '/root/autodl-tmp/SLSforASVspoof-2021-DF/xlsr2_300m.pt')
        self.ssl_model = SSLModel(cp_path, device)
        self.d_model = self.ssl_model.out_dim
        
        # Configuration
        self.num_groups = getattr(args, "num_groups", 8)  # Learnable groups instead of fixed size
        self.use_multiscale = getattr(args, "use_multiscale", False)
        
        # Multi-scale temporal fusion (optional)
        if self.use_multiscale:
            self.temporal_fusion = LightweightTemporalConv(self.d_model, self.d_model)
        
        # Positional encoding for layer ordering (support up to 30 layers)
        self.pos_encoding = PositionalEncoding(num_layers=30, d_model=self.d_model)
        
        # Hierarchical attention layers with residual connections
        self.temporal_attn = ResidualAttention(self.d_model, attn_dim=128, dropout=0.1)
        
        # Adaptive grouping (no need to specify num_layers)
        self.adaptive_grouping = AdaptiveGrouping(
            num_groups=self.num_groups, 
            d_model=self.d_model
        )
        
        self.intra_attn = ResidualAttention(self.d_model, attn_dim=128, dropout=0.1)
        self.inter_attn = ResidualAttention(self.d_model, attn_dim=128, dropout=0.1)
        
        # Feature refinement with residual
        self.group_refine = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.d_model * 2, self.d_model),
        )
        
        self.utt_refine = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.d_model * 2, self.d_model),
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 2),
        )
        
        # For training stability
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights with careful scaling"""
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def compute_attention_entropy(self, attn_weights: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy of attention weights for regularization
        Higher entropy = more uniform attention (potentially bad)
        Lower entropy = more focused attention
        """
        eps = 1e-8
        entropy = -(attn_weights * torch.log(attn_weights + eps)).sum(dim=-1).mean()
        return entropy
    
    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """
        Args:
            x: [B, T] or [B, T, 1] input waveform
            return_attention: whether to return attention weights for visualization
        Returns:
            output: [B, 2] - log probabilities
            (optional) attention_dict: dictionary of attention weights
        """
        # Validate input
        assert x.ndim in [2, 3], f"Expected 2D or 3D input, got {x.ndim}D"
        
        # Extract SSL features
        _, layer_results = self.ssl_model.extract_feat(x.squeeze(-1))
        
        # Convert layer results to tensors
        layer_features = []
        for hidden, _ in layer_results:
            # hidden: [T, B, D] -> [B, T, D]
            layer_features.append(hidden.transpose(0, 1))
        
        # Stack: [B, L, T, D]
        layer_stack = torch.stack(layer_features, dim=1)
        B, L, T, D = layer_stack.shape
        
        # ====== Level 1: Temporal Processing ======
        if self.use_multiscale:
            # 只在前几层和最后几层使用多尺度
            layer_temporal = []
            key_layers = [0, L//4, L//2, 3*L//4, L-1]  # 只处理5个关键层
            
            for l in range(L):
                if l in key_layers:
                    fused = self.temporal_fusion(layer_stack[:, l])
                    pooled, _ = self.temporal_attn(fused)
                else:
                    # 其他层直接池化
                    pooled, _ = self.temporal_attn(layer_stack[:, l])
                layer_temporal.append(pooled)
            layer_emb = torch.stack(layer_temporal, dim=1)
        else:
            # 全部直接池化（最快）
            layer_tokens = layer_stack.reshape(B * L, T, D)
            layer_pooled, temporal_weights = self.temporal_attn(layer_tokens)
            layer_emb = layer_pooled.reshape(B, L, D)
        
        # Add positional encoding (dynamically handles L layers)
        layer_emb = self.pos_encoding(layer_emb)  # [B, L, D]
        
        # ====== Level 2: Adaptive Grouping ======
        grouped_emb, group_assignments = self.adaptive_grouping(layer_emb)  # [B, num_groups, D]
        
        # Intra-group attention (process each group)
        group_vecs = []
        for g_idx in range(self.num_groups):
            g_vec, intra_w = self.intra_attn(grouped_emb[:, g_idx:g_idx+1, :])
            g_vec = g_vec + self.group_refine(g_vec)  # Residual connection
            group_vecs.append(g_vec)
        
        # ====== Level 3: Inter-group Attention ======
        group_stack = torch.stack(group_vecs, dim=1)  # [B, num_groups, D]
        utt_emb, inter_weights = self.inter_attn(group_stack)
        utt_emb = utt_emb + self.utt_refine(utt_emb)  # Residual connection
        
        # ====== Classification ======
        logits = self.classifier(utt_emb)
        output = F.log_softmax(logits, dim=-1)
        
        if return_attention:
            attention_dict = {
                'group_assignments': group_assignments,  # [B, L, num_groups]
                'inter_weights': inter_weights,  # [B, num_groups]
                'num_layers': L,  # 实际层数
                'num_groups': self.num_groups,  # 分组数
            }
            return output, attention_dict
        
        return output
    
    def get_regularization_loss(self) -> torch.Tensor:
        """
        Compute regularization loss for attention sparsity
        Call this during training to encourage focused attention
        Note: Currently returns zero; modify forward to cache weights if needed
        """
        return torch.tensor(0.0, device=self.device)





