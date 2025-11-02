import torch
import torch.nn as nn
import torch.nn.functional as F
import fairseq
import math


class SSLModel(nn.Module):
    """
    Self-supervised learning model wrapper for XLSR
    """
    def __init__(self, device):
        super(SSLModel, self).__init__()
        
        cp_path = '/root/autodl-tmp/SLSforASVspoof-2021-DF/xlsr2_300m.pt'
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
        self.model = model[0]
        self.device = device
        self.out_dim = 1024
        
    def extract_feat(self, input_data):
        """
        Extract features from SSL model
        
        Args:
            input_data: input audio tensor
            
        Returns:
            emb: final layer embeddings
            layer_results: intermediate layer results
        """
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


def getAttenF(layerResult):
    """
    Extract layer features from SSL model output
    
    Args:
        layerResult: list of layer outputs from SSL model
        
    Returns:
        layery: pooled layer representations [B, L, C]
        fullfeature: full layer features [B, L, T, C]
    """
    poollayerResult = []
    fullf = []
    
    for layer in layerResult:
        # Pooled representation
        layery = layer[0].transpose(0, 1).transpose(1, 2)  # (B, C, T)
        layery = F.adaptive_avg_pool1d(layery, 1)  # (B, C, 1)
        layery = layery.transpose(1, 2)  # (B, 1, C)
        poollayerResult.append(layery)
        
        # Full feature
        x = layer[0].transpose(0, 1)  # (B, T, C)
        x = x.view(x.size(0), -1, x.size(1), x.size(2))  # (B, 1, T, C)
        fullf.append(x)
    
    layery = torch.cat(poollayerResult, dim=1)  # (B, L, C)
    fullfeature = torch.cat(fullf, dim=1)  # (B, L, T, C)
    return layery, fullfeature


class TemporalBiasedAttention(nn.Module):
    """
    Temporal attention with early-segment bias
    Based on empirical findings that early segments are most informative
    """
    def __init__(self, embed_dim=1024, num_heads=8, early_bias=2.0, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.early_bias = early_bias
        
        # Learnable query token for pooling
        self.query_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # K, V projections
        self.kv = nn.Linear(embed_dim, embed_dim * 2, bias=True)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Forward pass with temporal bias
        
        Args:
            x: [B, T, C] temporal features
            mask: [B, T] optional attention mask
            
        Returns:
            pooled: [B, C] pooled representation
            attn_weights: [B, T] attention weights
        """
        B, T, C = x.shape
        
        # Generate K, V
        kv = self.kv(x).reshape(B, T, 2, self.num_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        
        # Generate Q from learnable token
        query = self.query_token.expand(B, -1, -1)
        q = self.q_proj(query).reshape(B, 1, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)
        
        # Calculate attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, 1, T]
        
        # Apply temporal bias favoring early positions
        temporal_bias = self._create_temporal_bias(T, device=x.device)
        attn = attn + temporal_bias.view(1, 1, 1, T)
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        # Weighted pooling
        pooled = (attn @ v).squeeze(2).reshape(B, C)
        pooled = self.proj(pooled)
        pooled = self.proj_drop(pooled)
        
        # Average attention weights across heads
        attn_weights = attn.squeeze(2).mean(dim=1)
        return pooled, attn_weights
    
    def _create_temporal_bias(self, length, device):
        """
        Create temporal bias with exponential decay
        Early positions receive higher bias scores
        
        Args:
            length: temporal sequence length
            device: torch device
            
        Returns:
            bias: [length] bias scores
        """
        positions = torch.arange(length, device=device).float()
        bias = self.early_bias * torch.exp(-positions / (length / 3))
        return bias


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for Transformer
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: [B, L, C] input tensor
        Returns:
            [B, L, C] tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    """
    Standard Transformer encoder layer with multi-head self-attention and FFN
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        
        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Args:
            src: [B, L, C] input tensor
            src_mask: optional attention mask
            src_key_padding_mask: optional padding mask
            
        Returns:
            output: [B, L, C] transformed tensor
            attn_weights: [B, num_heads, L, L] attention weights
        """
        # Self-attention with residual connection
        src2, attn_weights = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=True,
            average_attn_weights=False
        )
        src = src + self.dropout(src2)
        src = self.norm1(src)
        
        # FFN with residual connection
        src2 = self.ffn(src)
        src = src + src2
        src = self.norm2(src)
        
        return src, attn_weights


class TransformerEncoder(nn.Module):
    """
    Stack of Transformer encoder layers
    """
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            type(encoder_layer)(
                encoder_layer.self_attn.embed_dim,
                encoder_layer.self_attn.num_heads,
                encoder_layer.ffn[0].out_features,
                encoder_layer.dropout.p
            ) for _ in range(num_layers)
        ])
        self.num_layers = num_layers
        
    def forward(self, src, mask=None, src_key_padding_mask=None):
        """
        Args:
            src: [B, L, C] input tensor
            mask: optional attention mask
            src_key_padding_mask: optional padding mask
            
        Returns:
            output: [B, L, C] encoded tensor
            all_attn_weights: list of attention weights from each layer
        """
        output = src
        all_attn_weights = []
        
        for layer in self.layers:
            output, attn_weights = layer(
                output, 
                src_mask=mask, 
                src_key_padding_mask=src_key_padding_mask
            )
            all_attn_weights.append(attn_weights)
        
        return output, all_attn_weights


class MultiHeadAttentionPooling(nn.Module):
    """
    Multi-head attention pooling using learnable query token
    """
    def __init__(self, embed_dim=1024, num_heads=8, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Learnable query token
        self.query_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # K, V projections
        self.kv = nn.Linear(embed_dim, embed_dim * 2, bias=True)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Forward pass
        
        Args:
            x: [B, N, C] input features
            mask: [B, N] optional mask
            
        Returns:
            pooled: [B, C] pooled output
            attn_weights: [B, N] attention weights
        """
        B, N, C = x.shape
        
        # Generate K, V
        kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        
        # Generate Q from learnable token
        query = self.query_token.expand(B, -1, -1)
        q = self.q_proj(query).reshape(B, 1, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)
        
        # Calculate attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, 1, N]
        
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        # Pooling
        pooled = (attn @ v).squeeze(2).reshape(B, C)
        pooled = self.proj(pooled)
        pooled = self.proj_drop(pooled)
        
        # Average attention weights across heads
        attn_weights = attn.squeeze(2).mean(dim=1)
        
        return pooled, attn_weights


class DomainRobustContrastiveLoss(nn.Module):
    """
    Domain-robust contrastive learning loss for improved generalization
    """
    def __init__(self, temperature=0.07, margin=0.1):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        
    def forward(self, features, labels):
        """
        Calculate contrastive loss
        
        Args:
            features: [B, D] feature embeddings
            labels: [B] class labels
            
        Returns:
            loss: scalar contrastive loss
        """
        batch_size = features.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, device=features.device)
        
        # Light normalization to preserve diversity
        features = F.normalize(features, dim=1) * 0.9
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create positive and negative masks
        labels = labels.view(-1, 1)
        positive_mask = torch.eq(labels, labels.T).float().to(features.device)
        negative_mask = 1.0 - positive_mask
        positive_mask = positive_mask - torch.eye(batch_size).to(features.device)
        
        # Check for positive pairs
        positive_pairs = positive_mask.sum(dim=1)
        if positive_pairs.sum() == 0:
            return torch.tensor(0.0, device=features.device)
        
        # Margin-based contrastive loss
        positive_sim = (positive_mask * similarity_matrix).sum(dim=1) / (positive_pairs + 1e-8)
        negative_sim = (negative_mask * similarity_matrix).sum(dim=1) / (negative_mask.sum(dim=1) + 1e-8)
        
        margin_loss = torch.clamp(self.margin + negative_sim - positive_sim, min=0.0)
        valid_samples = positive_pairs > 0
        
        if valid_samples.sum() == 0:
            return torch.tensor(0.0, device=features.device)
        
        loss = margin_loss[valid_samples].mean()
        return torch.clamp(loss, 0, 1.0)


class FocalLoss(nn.Module):
    """
    Focal loss for handling class imbalance
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        Calculate focal loss
        
        Args:
            inputs: [B, C] logits
            targets: [B] class labels
            
        Returns:
            loss: scalar focal loss
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p = torch.exp(-ce_loss)
        focal_loss = (1 - p) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                if isinstance(self.alpha, list):
                    alpha = torch.tensor(self.alpha, device=inputs.device)
                else:
                    alpha = self.alpha
                alpha_t = alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CombinedLossFlexible(nn.Module):
    """
    Combined loss function supporting both Cross-Entropy and Focal Loss
    with domain-robust contrastive learning
    """
    def __init__(self, loss_type='focal', weight=None, alpha=None, gamma=2.0, 
                 temperature=0.07, contrastive_weight=0.1):
        super().__init__()
        self.loss_type = loss_type
        
        if loss_type == 'ce':
            self.primary_loss = nn.CrossEntropyLoss(weight=weight)
        elif loss_type == 'focal':
            self.primary_loss = FocalLoss(alpha=alpha, gamma=gamma)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}. Choose 'ce' or 'focal'")
        
        self.contrastive_loss = DomainRobustContrastiveLoss(
            temperature=temperature, 
            margin=0.1
        )
        self.contrastive_weight = contrastive_weight
        
    def forward(self, logits, features, labels):
        """
        Calculate combined loss
        
        Args:
            logits: [B, C] classification logits
            features: [B, D] contrastive features
            labels: [B] class labels
            
        Returns:
            total_loss: total combined loss
            primary: primary classification loss
            combined_contrastive: weighted contrastive loss
        """
        primary = self.primary_loss(logits, labels)
        contrastive = self.contrastive_loss(features, labels)
        combined_contrastive = self.contrastive_weight * contrastive
        total_loss = primary + combined_contrastive
        
        return total_loss, primary, combined_contrastive


class ModelTransformerHierarchical(nn.Module):
    """
    Transformer-based hierarchical model:
    1. Stage 1: Temporal attention with early-segment bias (保持不变)
    2. Stage 2-3: Standard Transformer encoder替代分组注意力机制
    3. Stage 4: Skip connection from layer 23 (保持不变)
    
    Architecture improvements:
    - Replaced grouped attention with standard Transformer layers
    - Added sinusoidal positional encoding
    - Multi-layer self-attention for better layer interaction modeling
    - Final pooling for utterance-level representation
    """
    def __init__(self, args, device):
        super().__init__()
        self.device = device
        self.ssl_model = SSLModel(device)
        self.d_model = 1024
        self.num_heads = getattr(args, "num_heads", 8)
        self.num_transformer_layers = getattr(args, "num_transformer_layers", 4)
        self.dim_feedforward = getattr(args, "dim_feedforward", 2048)
        
        print("="*60)
        print("Initializing Transformer-based Hierarchical Model")
        print("="*60)
        print(f"Configuration:")
        print(f"  - Number of attention heads: {self.num_heads}")
        print(f"  - Number of Transformer layers: {self.num_transformer_layers}")
        print(f"  - Model dimension: {self.d_model}")
        print(f"  - Feedforward dimension: {self.dim_feedforward}")
        print(f"\nArchitectural design:")
        print(f"  - Stage 1: Temporal attention with early-segment bias")
        print(f"  - Stage 2-3: {self.num_transformer_layers}-layer Transformer encoder")
        print(f"  - Stage 4: Skip connection from layer 23 (40.8% contribution)")
        print(f"  - Final: Multi-head attention pooling")
        print("="*60)
        
        # Stage 1: Temporal attention with early-segment bias
        self.temporal_attn = TemporalBiasedAttention(
            embed_dim=self.d_model,
            num_heads=self.num_heads,
            early_bias=2.0,
            dropout=0.1
        )
        
        # Stage 2-3: Transformer encoder (替代分组注意力)
        self.positional_encoding = PositionalEncoding(
            d_model=self.d_model,
            max_len=100,  # 假设最多24层 + padding
            dropout=0.1
        )
        
        encoder_layer = TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=0.1
        )
        
        self.transformer_encoder = TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=self.num_transformer_layers
        )
        
        # Final pooling layer (替代inter-group attention)
        self.final_pooling = MultiHeadAttentionPooling(
            embed_dim=self.d_model,
            num_heads=self.num_heads,
            dropout=0.1
        )
        
        # Refinement after pooling
        self.utt_refine = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, 512),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(512, self.d_model),
            nn.Dropout(0.15)
        )
        
        # Skip connection from layer 23 (most important layer)
        self.skip_layer_idx = 23
        self.skip_projection = nn.Linear(self.d_model, self.d_model)
        self.skip_gate = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.Sigmoid()
        )
        
        # Projection head for contrastive learning
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
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(self.d_model, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(p=0.15),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, 2),
        )
        self.logsoftmax = nn.LogSoftmax(dim=1)
        
        # Storage for attention weights (interpretability)
        self.attention_weights = {}
    
    def forward(self, x, return_features=False, return_interpretability=False):
        """
        Forward pass
        
        Args:
            x: input audio tensor
            return_features: whether to return projection features
            return_interpretability: whether to return interpretability info
            
        Returns:
            output: log-softmax predictions
            projected_features: (optional) features for contrastive loss
            interpretations: (optional) interpretability dict
        """
        # Extract SSL features
        x_ssl_feat, layerResult = self.ssl_model.extract_feat(x.squeeze(-1))
        _, fullfeature = getAttenF(layerResult)  # (B, L, T, C)
        B, L, T, C = fullfeature.shape
        
        # Stage 1: Temporal attention with early-segment bias
        layer_tokens = fullfeature.contiguous().view(B * L, T, C)
        layer_emb, temporal_attn = self.temporal_attn(layer_tokens)
        layer_emb = layer_emb.view(B, L, C)  # [B, L, C]
        
        if return_interpretability:
            self.attention_weights['temporal'] = temporal_attn.view(B, L, T).detach()
        
        # Extract skip connection from layer 23
        if self.skip_layer_idx < L:
            skip_feature = layer_emb[:, self.skip_layer_idx, :].clone()
            skip_feature = self.skip_projection(skip_feature)
        else:
            skip_feature = layer_emb[:, -1, :].clone()
            skip_feature = self.skip_projection(skip_feature)
        
        # Stage 2-3: Transformer encoder (替代原来的分组注意力)
        # Add positional encoding
        layer_emb_pos = self.positional_encoding(layer_emb)  # [B, L, C]
        
        # Apply Transformer encoder
        transformer_output, all_attn_weights = self.transformer_encoder(layer_emb_pos)  # [B, L, C]
        
        if return_interpretability:
            # Store all transformer attention weights
            self.attention_weights['transformer_layers'] = [
                attn.detach() for attn in all_attn_weights
            ]
        
        # Final pooling to get utterance-level representation
        utt_emb, pooling_attn = self.final_pooling(transformer_output)  # [B, C]
        
        if return_interpretability:
            self.attention_weights['final_pooling'] = pooling_attn.detach()
        
        # Refinement
        utt_emb = utt_emb + self.utt_refine(utt_emb)
        
        # Stage 4: Gated skip connection from layer 23
        combined = torch.cat([utt_emb, skip_feature], dim=1)
        gate = self.skip_gate(combined)
        utt_emb = gate * utt_emb + (1 - gate) * skip_feature
        
        if return_interpretability:
            self.attention_weights['skip_gate'] = gate.detach()
        
        # Classification
        logits = self.classifier(utt_emb)
        output = self.logsoftmax(logits)
        
        # Return based on flags
        if return_interpretability:
            interpretations = self._compute_interpretations(
                temporal_attn, all_attn_weights, pooling_attn, B, L, T
            )
            if return_features:
                projected_features = self.projection_head(utt_emb)
                return output, projected_features, interpretations
            return output, interpretations
        
        if return_features:
            projected_features = self.projection_head(utt_emb)
            return output, projected_features
        
        return output
    
    def _compute_interpretations(self, temporal_attn, transformer_attn_list, pooling_attn, B, L, T):
        """
        Compute interpretability metrics for Transformer-based model
        
        Args:
            temporal_attn: temporal attention weights
            transformer_attn_list: list of transformer layer attention weights
            pooling_attn: final pooling attention weights
            B: batch size
            L: number of layers
            T: temporal length
            
        Returns:
            dict containing interpretability information
        """
        # Layer importance from final pooling
        layer_importance = pooling_attn  # [B, L]
        
        # Temporal importance
        temporal_attn_reshaped = temporal_attn.view(B, L, T)
        temporal_importance = self._compute_temporal_importance(
            temporal_attn_reshaped, layer_importance
        )
        
        # Generate text explanations
        explanations = self._generate_explanations(layer_importance, temporal_importance)
        
        return {
            'layer_importance': layer_importance,
            'temporal_importance': temporal_importance,
            'attention_weights': self.attention_weights,
            'text_explanations': explanations
        }
    
    def _compute_temporal_importance(self, temporal_attn, layer_importance):
        """
        Compute temporal importance weighted by layer importance
        
        Args:
            temporal_attn: [B, L, T] temporal attention weights
            layer_importance: [B, L] layer importance scores
            
        Returns:
            temporal_importance: [B, T] temporal importance scores
        """
        B, L, T = temporal_attn.shape
        layer_importance_expanded = layer_importance.unsqueeze(2)  # [B, L, 1]
        weighted_temporal_attn = temporal_attn * layer_importance_expanded
        temporal_importance = weighted_temporal_attn.sum(dim=1)  # [B, T]
        # Normalize
        temporal_importance = temporal_importance / (temporal_importance.sum(dim=1, keepdim=True) + 1e-8)
        return temporal_importance
    
    def _generate_explanations(self, layer_importance, temporal_importance):
        """
        Generate human-readable text explanations
        
        Args:
            layer_importance: [B, L] layer importance scores
            temporal_importance: [B, T] temporal importance scores
            
        Returns:
            explanations: list of explanation strings
        """
        B = layer_importance.size(0)
        explanations = []
        
        for i in range(B):
            layer_imp = layer_importance[i]
            temporal_imp = temporal_importance[i]
            
            # Top 3 important layers
            top_layers = torch.topk(layer_imp, k=min(3, len(layer_imp)))
            top_layer_indices = top_layers.indices.cpu().numpy()
            top_layer_values = top_layers.values.cpu().numpy()
            
            # Temporal distribution
            T = temporal_imp.size(0)
            early = temporal_imp[:T//3].sum().item()
            middle = temporal_imp[T//3:2*T//3].sum().item()
            late = temporal_imp[2*T//3:].sum().item()
            
            # Build explanation string
            explanation = "Detection decision based on:\n"
            explanation += f"- Key layers: "
            explanation += f"{top_layer_indices[0]}({top_layer_values[0]:.1%}), "
            explanation += f"{top_layer_indices[1]}({top_layer_values[1]:.1%}), "
            explanation += f"{top_layer_indices[2]}({top_layer_values[2]:.1%})\n"
            
            # Sort temporal regions by importance
            temporal_regions = [('early', early), ('middle', middle), ('late', late)]
            temporal_regions.sort(key=lambda x: x[1], reverse=True)
            explanation += f"- Temporal focus: "
            explanation += f"{temporal_regions[0][0]}({temporal_regions[0][1]:.1%}), "
            explanation += f"{temporal_regions[1][0]}({temporal_regions[1][1]:.1%})"
            
            explanations.append(explanation)
        
        return explanations