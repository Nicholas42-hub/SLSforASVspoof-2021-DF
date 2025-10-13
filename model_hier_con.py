import torch
import torch.nn as nn
import torch.nn.functional as F
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
            
        result = self.model(input_tmp, mask=False, features_only=True)
        emb = result['x']
        layer_results = result['layer_results']
        return emb, layer_results


def getAttenF(layerResult):
    """提取层特征"""
    poollayerResult = []
    fullf = []
    
    for layer in layerResult:
        x = layer[0].transpose(0, 1)  # (B, T, C)
        
        # Pooled representation
        layery = F.adaptive_avg_pool1d(x.transpose(1, 2), 1).transpose(1, 2)
        poollayerResult.append(layery)
        
        # Full feature
        fullf.append(x.unsqueeze(1))
    
    layery = torch.cat(poollayerResult, dim=1)
    fullfeature = torch.cat(fullf, dim=1)
    return layery, fullfeature


class AttnPool(nn.Module):
    def __init__(self, in_dim: int, attn_dim: int = 128):
        super().__init__()
        self.proj = nn.Linear(in_dim, attn_dim)
        self.score = nn.Linear(attn_dim, 1, bias=False)
        self.layer_norm = nn.LayerNorm(in_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        x = self.layer_norm(x)
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
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, in_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor):
        return x + self.net(x)


class DomainRobustContrastiveLoss(nn.Module):
    """域鲁棒的对比学习损失"""
    def __init__(self, temperature=0.07, margin=0.1):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        
    def forward(self, features, labels):
        batch_size = features.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, device=features.device)
        
        features = F.normalize(features, dim=1) * 0.9
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        labels = labels.view(-1, 1)
        positive_mask = torch.eq(labels, labels.T).float().to(features.device)
        negative_mask = 1.0 - positive_mask
        positive_mask = positive_mask - torch.eye(batch_size).to(features.device)
        
        positive_pairs = positive_mask.sum(dim=1)
        if positive_pairs.sum() == 0:
            return torch.tensor(0.0, device=features.device)
        
        positive_sim = (positive_mask * similarity_matrix).sum(dim=1) / (positive_pairs + 1e-8)
        negative_sim = (negative_mask * similarity_matrix).sum(dim=1) / (negative_mask.sum(dim=1) + 1e-8)
        
        margin_loss = torch.clamp(self.margin + negative_sim - positive_sim, min=0.0)
        valid_samples = positive_pairs > 0
        
        if valid_samples.sum() == 0:
            return torch.tensor(0.0, device=features.device)
        
        loss = margin_loss[valid_samples].mean()
        return torch.clamp(loss, 0, 1.0)


class CompatibleCombinedLoss(nn.Module):
    """兼容的组合损失函数"""
    def __init__(self, weight=None, temperature=0.07, contrastive_weight=0.1):
        super().__init__()
        self.cce_loss = nn.CrossEntropyLoss(weight=weight)
        self.contrastive_loss = DomainRobustContrastiveLoss(temperature=temperature, margin=0.1)
        self.contrastive_weight = contrastive_weight
        
    def forward(self, logits, features, labels):
        cce = self.cce_loss(logits, labels)
        contrastive = self.contrastive_loss(features, labels)
        combined_contrastive = self.contrastive_weight * contrastive
        total_loss = cce + combined_contrastive
        return total_loss, cce, combined_contrastive


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha: Weighting factor in range (0,1) to balance positive/negative examples
                   or a list of weights for each class. If None, alpha = 1
            gamma: Focusing parameter for modulating loss, gamma >= 0
            reduction: 'none' | 'mean' | 'sum'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (B, C) logits
            targets: (B,) class labels
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p = torch.exp(-ce_loss)  # pt
        focal_loss = (1 - p) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                # alpha is a list/tensor of weights for each class
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


class CombinedLossWithFocal(nn.Module):
    """
    组合损失函数：Focal Loss + Contrastive Loss
    """
    def __init__(self, alpha=None, gamma=2.0, temperature=0.07, contrastive_weight=0.1):
        """
        Args:
            alpha: Class weights for Focal Loss. Can be:
                   - None: no weighting
                   - float: weighting factor for positive class
                   - list: [weight_class0, weight_class1]
            gamma: Focusing parameter for Focal Loss
            temperature: Temperature for contrastive loss
            contrastive_weight: Weight for contrastive loss
        """
        super().__init__()
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma, reduction='mean')
        self.contrastive_loss = DomainRobustContrastiveLoss(temperature=temperature, margin=0.1)
        self.contrastive_weight = contrastive_weight
        
    def forward(self, logits, features, labels):
        """
        Args:
            logits: (B, 2) classification logits
            features: (B, D) projected features for contrastive learning
            labels: (B,) ground truth labels
        """
        focal = self.focal_loss(logits, labels)
        contrastive = self.contrastive_loss(features, labels)
        combined_contrastive = self.contrastive_weight * contrastive
        total_loss = focal + combined_contrastive
        
        return total_loss, focal, combined_contrastive


class CombinedLossFlexible(nn.Module):
    """
    灵活的组合损失函数：支持 CE/Focal + Contrastive
    """
    def __init__(self, loss_type='focal', weight=None, alpha=None, gamma=2.0, 
                 temperature=0.07, contrastive_weight=0.1):
        """
        Args:
            loss_type: 'ce' for CrossEntropy or 'focal' for Focal Loss
            weight: Class weights for CE loss
            alpha: Class weights for Focal Loss
            gamma: Focusing parameter for Focal Loss
            temperature: Temperature for contrastive loss
            contrastive_weight: Weight for contrastive loss
        """
        super().__init__()
        self.loss_type = loss_type
        
        if loss_type == 'ce':
            self.primary_loss = nn.CrossEntropyLoss(weight=weight)
        elif loss_type == 'focal':
            self.primary_loss = FocalLoss(alpha=alpha, gamma=gamma)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}. Choose 'ce' or 'focal'")
        
        self.contrastive_loss = DomainRobustContrastiveLoss(temperature=temperature, margin=0.1)
        self.contrastive_weight = contrastive_weight
        
    def forward(self, logits, features, labels):
        """
        Args:
            logits: (B, 2) classification logits
            features: (B, D) projected features
            labels: (B,) ground truth labels
        """
        primary = self.primary_loss(logits, labels)
        contrastive = self.contrastive_loss(features, labels)
        combined_contrastive = self.contrastive_weight * contrastive
        total_loss = primary + combined_contrastive
        
        return total_loss, primary, combined_contrastive


class ModelHierarchicalContrastive(nn.Module):
    """
    带可解释性的层次注意力模型
    """
    def __init__(self, args, device):
        super().__init__()
        self.device = device
        self.ssl_model = SSLModel(device)
        self.d_model = 1024
        self.group_size = getattr(args, "group_size", 3)
        
        # 层次注意力模块
        self.temporal_attn = AttnPool(in_dim=self.d_model, attn_dim=128)
        self.intra_attn = AttnPool(in_dim=self.d_model, attn_dim=128)
        self.group_refine = ResidualRefine(in_dim=self.d_model, hidden=512, dropout=0.15)
        self.inter_attn = AttnPool(in_dim=self.d_model, attn_dim=128)
        self.utt_refine = ResidualRefine(in_dim=self.d_model, hidden=512, dropout=0.15)
        
        # 对比学习投影头
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
        
        # 分类器
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
        
        # 存储attention weights
        self.attention_weights = {}

    def forward(self, x, return_features=False, return_interpretability=False):
        """
        前向传播
        Args:
            x: 输入音频 (B, T)
            return_features: 是否返回对比学习特征
            return_interpretability: 是否返回可解释性信息
        """
        x_ssl_feat, layerResult = self.ssl_model.extract_feat(x.squeeze(-1))
        _, fullfeature = getAttenF(layerResult)  # (B, L, T, C)

        B, L, T, C = fullfeature.shape
        
        # 1. Temporal Attention
        layer_tokens = fullfeature.contiguous().view(B * L, T, C)
        layer_emb, temporal_attn = self.temporal_attn(layer_tokens)
        layer_emb = layer_emb.view(B, L, C)
        
        # 保存temporal attention
        temporal_attn = temporal_attn.view(B, L, T)
        if return_interpretability:
            self.attention_weights['temporal'] = temporal_attn.detach()
        
        # 处理分组
        if layer_emb.size(1) % self.group_size != 0:
            pad_size = self.group_size - (layer_emb.size(1) % self.group_size)
            layer_emb = F.pad(layer_emb, (0, 0, 0, pad_size), mode='constant', value=0)
        
        # 2. Intra-group Attention
        groups = torch.split(layer_emb, self.group_size, dim=1)
        group_vecs = []
        intra_attn_weights = []
        
        for g in groups:
            g_vec, intra_attn = self.intra_attn(g)
            g_vec = self.group_refine(g_vec)
            group_vecs.append(g_vec)
            intra_attn_weights.append(intra_attn)
        
        if return_interpretability:
            self.attention_weights['intra_group'] = torch.stack(
                [a.detach() for a in intra_attn_weights], dim=1
            )
        
        # 3. Inter-group Attention
        group_stack = torch.stack(group_vecs, dim=1)
        utt_emb, inter_attn = self.inter_attn(group_stack)
        utt_emb = self.utt_refine(utt_emb)
        
        if return_interpretability:
            self.attention_weights['inter_group'] = inter_attn.detach()
        
        # 分类
        logits = self.classifier(utt_emb)
        output = self.logsoftmax(logits)
        
        # 返回值处理
        if return_interpretability:
            interpretations = self._compute_interpretations(
                temporal_attn, intra_attn_weights, inter_attn, B, L, T
            )
            if return_features:
                projected_features = self.projection_head(utt_emb)
                return output, projected_features, interpretations
            return output, interpretations
        
        if return_features:
            projected_features = self.projection_head(utt_emb)
            return output, projected_features
        
        return output
    
    def _compute_interpretations(self, temporal_attn, intra_attn_list, inter_attn, B, L, T):
        """计算可解释性指标"""
        # 1. 层级重要性
        layer_importance = self._compute_layer_importance(intra_attn_list, inter_attn, L)
        
        # 2. 时序重要性
        temporal_importance = self._compute_temporal_importance(temporal_attn, layer_importance)
        
        # 3. 生成文本解释
        explanations = self._generate_explanations(layer_importance, temporal_importance)
        
        return {
            'layer_importance': layer_importance,  # (B, L)
            'temporal_importance': temporal_importance,  # (B, T)
            'attention_weights': self.attention_weights,
            'text_explanations': explanations
        }
    
    def _compute_layer_importance(self, intra_attn_list, inter_attn, L):
        """计算每层的重要性分数"""
        B = inter_attn.size(0)
        num_groups = len(intra_attn_list)
        
        layer_importance = []
        for group_idx in range(num_groups):
            group_weight = inter_attn[:, group_idx:group_idx+1]  # (B, 1)
            intra_weight = intra_attn_list[group_idx]  # (B, group_size)
            layer_weight = group_weight * intra_weight  # (B, group_size)
            layer_importance.append(layer_weight)
        
        layer_importance = torch.cat(layer_importance, dim=1)[:, :L]
        
        # 归一化
        layer_importance = layer_importance / (layer_importance.sum(dim=1, keepdim=True) + 1e-8)
        return layer_importance
    
    def _compute_temporal_importance(self, temporal_attn, layer_importance):
        """计算时序重要性"""
        B, L, T = temporal_attn.shape
        layer_importance_expanded = layer_importance.unsqueeze(2)  # (B, L, 1)
        weighted_temporal_attn = temporal_attn * layer_importance_expanded  # (B, L, T)
        temporal_importance = weighted_temporal_attn.sum(dim=1)  # (B, T)
        
        # 归一化
        temporal_importance = temporal_importance / (temporal_importance.sum(dim=1, keepdim=True) + 1e-8)
        return temporal_importance
    
    def _generate_explanations(self, layer_importance, temporal_importance):
        """生成文本解释"""
        B = layer_importance.size(0)
        explanations = []
        
        for i in range(B):
            layer_imp = layer_importance[i]
            temporal_imp = temporal_importance[i]
            
            # 最重要的3层
            top_layers = torch.topk(layer_imp, k=min(3, len(layer_imp)))
            top_layer_indices = top_layers.indices.cpu().numpy()
            top_layer_values = top_layers.values.cpu().numpy()
            
            # 时序区域分析
            T = temporal_imp.size(0)
            early = temporal_imp[:T//3].sum().item()
            middle = temporal_imp[T//3:2*T//3].sum().item()
            late = temporal_imp[2*T//3:].sum().item()
            
            # 生成解释
            explanation = "Detection decision based on:\n"
            explanation += f"- Key layers: {top_layer_indices[0]}({top_layer_values[0]:.1%}), "
            explanation += f"{top_layer_indices[1]}({top_layer_values[1]:.1%}), "
            explanation += f"{top_layer_indices[2]}({top_layer_values[2]:.1%})\n"
            
            temporal_regions = [('early', early), ('middle', middle), ('late', late)]
            temporal_regions.sort(key=lambda x: x[1], reverse=True)
            explanation += f"- Temporal focus: {temporal_regions[0][0]}({temporal_regions[0][1]:.1%}), "
            explanation += f"{temporal_regions[1][0]}({temporal_regions[1][1]:.1%})"
            
            explanations.append(explanation)
        
        return explanations


# 保持兼容性
CombinedLoss = CompatibleCombinedLoss  # 或者改为 CombinedLossWithFocal
Model = ModelHierarchicalContrastive