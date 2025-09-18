import random
import sys
from typing import Union, Optional, Tuple
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
import seaborn as sns

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import fairseq

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning Loss with temperature scaling and InfoNCE"""
    def __init__(self, temperature=0.3, contrast_mode='all', base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None, return_per_class=False):
        """
        Args:
            features: hidden vector of shape [bsz, n_views, f_dim] or [bsz, f_dim]
            labels: ground truth of shape [bsz]
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
            return_per_class: whether to return per-class loss statistics
        Returns:
            A loss scalar and optionally per-class statistics.
        """
        device = features.device

        if len(features.shape) < 3:
            features = features.unsqueeze(1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        if return_per_class and labels is not None:
            # Compute per-class statistics
            per_class_stats = {}
            unique_labels = torch.unique(labels.squeeze())
            
            for label in unique_labels:
                label_mask = (labels.squeeze() == label)
                if label_mask.sum() > 0:
                    label_loss = loss.view(anchor_count, batch_size)[0, label_mask].mean()
                    per_class_stats[f'class_{label.item()}'] = label_loss.item()
            
            return loss, per_class_stats

        return loss

class LocalPatchContrastLoss(nn.Module):
    """Local patch contrast within audio segments"""
    def __init__(self, temperature=0.1, patch_size=32, overlap=0.3, sample_ratio=0.5, max_negatives=128):
        super().__init__()
        self.temperature = temperature
        self.patch_size = patch_size
        self.overlap = overlap
        self.sample_ratio = sample_ratio
        self.max_negatives = max_negatives  # Add this missing attribute
        
    def create_patches(self, features, patch_size, overlap):
        """Create overlapping patches from temporal features"""
        B, T, C = features.shape
        stride = int(patch_size * (1 - overlap))
        
        patches = []
        for i in range(0, T - patch_size + 1, stride):
            patch = features[:, i:i+patch_size, :]  # (B, patch_size, C)
            patch_emb = patch.mean(dim=1)  # Global average pooling (B, C)
            patches.append(patch_emb)
        
        if len(patches) == 0:
            # If sequence too short, use the whole sequence
            patches = [features.mean(dim=1)]
            
        return torch.stack(patches, dim=1)  # (B, num_patches, C)
    
    def forward(self, features, labels):
        device = features.device
        B, T, C = features.shape
        
        # 随机采样部分样本以减少计算
        if self.sample_ratio < 1.0:
            sample_size = max(1, int(B * self.sample_ratio))
            indices = torch.randperm(B)[:sample_size]
            features = features[indices]
            labels = labels[indices]
            B = features.shape[0]
        
        # 创建补丁
        patches = self.create_patches(features, self.patch_size, self.overlap)
        num_patches = patches.shape[1]
        
        if num_patches < 2:
            return torch.tensor(0.0, device=device)
        
        # 向量化计算，避免嵌套循环
        all_anchors = []
        all_positives = []
        
        # 收集所有anchor-positive对
        for b in range(B):
            for i in range(num_patches - 1):
                all_anchors.append(patches[b, i])
                all_positives.append(patches[b, i + 1])
        
        if len(all_anchors) == 0:
            return torch.tensor(0.0, device=device)
        
        # 批量处理
        anchors = torch.stack(all_anchors)  # (N, C)
        positives = torch.stack(all_positives)  # (N, C)
        
        # 创建负样本池（随机采样以控制计算量）
        neg_pool = []
        for b in range(B):
            for i in range(num_patches):
                neg_pool.append(patches[b, i])
        
        neg_pool = torch.stack(neg_pool)  # (B*num_patches, C)
        
        # 限制负样本数量
        if len(neg_pool) > self.max_negatives:
            indices = torch.randperm(len(neg_pool))[:self.max_negatives]
            neg_pool = neg_pool[indices]
        
        # 批量计算相似度
        anchors = F.normalize(anchors, p=2, dim=1)
        positives = F.normalize(positives, p=2, dim=1)
        neg_pool = F.normalize(neg_pool, p=2, dim=1)
        
        # 计算所有相似度
        pos_sims = torch.sum(anchors * positives, dim=1) / self.temperature  # (N,)
        neg_sims = torch.matmul(anchors, neg_pool.T) / self.temperature  # (N, num_neg)
        
        # InfoNCE损失
        logits = torch.cat([pos_sims.unsqueeze(1), neg_sims], dim=1)  # (N, 1+num_neg)
        labels_loss = torch.zeros(logits.size(0), dtype=torch.long, device=device)
        loss = F.cross_entropy(logits, labels_loss)
        
        return loss

class DataAugmentContrastLoss(nn.Module):
    """Data augmentation-based contrastive learning using RawBoost as positive pairs"""
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, original_features, augmented_features, labels):
        """
        Args:
            original_features: (B, C) original embeddings
            augmented_features: (B, C) augmented embeddings (from RawBoost)
            labels: (B,) labels
        """
        device = original_features.device
        B, C = original_features.shape
        
        # Normalize features
        original_features = F.normalize(original_features, p=2, dim=1)
        augmented_features = F.normalize(augmented_features, p=2, dim=1)
        
        total_loss = 0.0
        
        for i in range(B):
            anchor = original_features[i]  # Original sample
            positive = augmented_features[i]  # Augmented version as positive
            
            # Get negatives: all other samples (both original and augmented)
            negatives = []
            for j in range(B):
                if j != i:
                    negatives.append(original_features[j])
                    negatives.append(augmented_features[j])
            
            if len(negatives) == 0:
                continue
                
            negatives = torch.stack(negatives)  # (2*(B-1), C)
            
            # Compute similarities
            pos_sim = torch.dot(anchor, positive) / self.temperature
            neg_sims = torch.matmul(negatives, anchor) / self.temperature
            
            # InfoNCE loss
            logits = torch.cat([pos_sim.unsqueeze(0), neg_sims])
            loss = -F.log_softmax(logits, dim=0)[0]
            
            total_loss += loss
            
        return total_loss / B

class EmbeddingVisualizer:
    """Visualization and analysis of embeddings for real vs fake separation"""
    
    @staticmethod
    def plot_embeddings(embeddings, labels, method='tsne', save_path=None, title="Embedding Visualization"):
        """
        Plot embeddings using t-SNE or UMAP
        Args:
            embeddings: (N, C) numpy array of embeddings
            labels: (N,) numpy array of labels (0=real, 1=fake)
            method: 'tsne' or 'umap'
            save_path: path to save the plot
            title: plot title
        """
        plt.figure(figsize=(10, 8))
        
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)//4))
        elif method == 'umap':
            reducer = umap.UMAP(n_components=2, random_state=42)
        else:
            raise ValueError("Method must be 'tsne' or 'umap'")
        
        # Reduce dimensionality
        reduced_embeddings = reducer.fit_transform(embeddings)
        
        # Create scatter plot
        colors = ['blue', 'red']
        class_names = ['Real', 'Fake']
        
        for i, (color, name) in enumerate(zip(colors, class_names)):
            mask = labels == i
            if np.any(mask):
                plt.scatter(reduced_embeddings[mask, 0], reduced_embeddings[mask, 1], 
                           c=color, label=name, alpha=0.6, s=30)
        
        plt.legend()
        plt.title(f"{title} ({method.upper()})")
        plt.xlabel(f"{method.upper()}-1")
        plt.ylabel(f"{method.upper()}-2")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    @staticmethod
    def compute_separation_metrics(embeddings, labels):
        """
        Compute separation metrics for real vs fake embeddings
        """
        real_embeddings = embeddings[labels == 0]
        fake_embeddings = embeddings[labels == 1]
        
        if len(real_embeddings) == 0 or len(fake_embeddings) == 0:
            return {}
        
        # Compute centroids
        real_centroid = np.mean(real_embeddings, axis=0)
        fake_centroid = np.mean(fake_embeddings, axis=0)
        
        # Inter-class distance (distance between centroids)
        inter_class_dist = np.linalg.norm(real_centroid - fake_centroid)
        
        # Intra-class distances (average distance within each class)
        real_intra_dist = np.mean([np.linalg.norm(emb - real_centroid) for emb in real_embeddings])
        fake_intra_dist = np.mean([np.linalg.norm(emb - fake_centroid) for emb in fake_embeddings])
        avg_intra_dist = (real_intra_dist + fake_intra_dist) / 2
        
        # Silhouette-like ratio
        separation_ratio = inter_class_dist / (avg_intra_dist + 1e-8)
        
        return {
            'inter_class_distance': inter_class_dist,
            'real_intra_distance': real_intra_dist,
            'fake_intra_distance': fake_intra_dist,
            'separation_ratio': separation_ratio
        }

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
        
        # Contrastive learning strategy
        self.contrastive_strategy = getattr(args, "contrastive_strategy", "hierarchical")  # hierarchical, local_patch, data_augment

        self.temporal_attn = AttnPool(in_dim=self.d_model, attn_dim=128)
        self.intra_attn = AttnPool(in_dim=self.d_model, attn_dim=128)
        self.group_refine = ResidualRefine(in_dim=self.d_model, hidden=512, dropout=0.1)
        self.inter_attn = AttnPool(in_dim=self.d_model, attn_dim=128)
        self.utt_refine = ResidualRefine(in_dim=self.d_model, hidden=512, dropout=0.1)

        # Add projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(self.d_model, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            L2Norm(dim=1)
        )

        # Add more regularization to prevent overfitting
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
        
        # Initialize different contrastive losses
        temp = getattr(args, "contrastive_temperature", 0.1)
        self.hierarchical_contrastive_loss = SupConLoss(temperature=temp)
        # Fix: Add max_negatives parameter
        self.local_patch_loss = LocalPatchContrastLoss(
            temperature=temp, 
            patch_size=32, 
            overlap=0.3, 
            sample_ratio=0.5, 
            max_negatives=128
        )
        self.data_augment_loss = DataAugmentContrastLoss(temperature=temp)
        
        # Gradient scaling
        self.contrastive_gradient_scale = 0.5
        
        # For storing embeddings for visualization
        self.stored_embeddings = []
        self.stored_labels = []

    def forward(self, x, labels=None, x_augmented=None, return_contrastive=False, store_embeddings=False):
        # Process original input
        x_ssl_feat, layerResult = self.ssl_model.extract_feat(x.squeeze(-1))
        _, fullfeature = getAttenF(layerResult)  # (B, L, T, C)

        B, L, T, C = fullfeature.shape
        layer_tokens = fullfeature.contiguous().view(B * L, T, C)
        layer_emb, _ = self.temporal_attn(layer_tokens)
        layer_emb = layer_emb.view(B, L, C)

        groups = torch.split(layer_emb, self.group_size, dim=1)
        group_vecs = []
        for g in groups:
            g_vec, _ = self.intra_attn(g)
            g_vec = self.group_refine(g_vec)
            group_vecs.append(g_vec)

        group_stack = torch.stack(group_vecs, dim=1)
        utt_emb, _ = self.inter_attn(group_stack)
        utt_emb = self.utt_refine(utt_emb)

        # Store embeddings for visualization if requested
        if store_embeddings and labels is not None:
            self.stored_embeddings.append(utt_emb.detach().cpu().numpy())
            self.stored_labels.append(labels.cpu().numpy())

        # Classification logits
        logits = self.classifier(utt_emb)
        output = self.logsoftmax(logits)
        
        if return_contrastive and labels is not None:
            contrastive_loss = torch.tensor(0.0, device=self.device)
            per_class_stats = {}
            
            if self.contrastive_strategy == "hierarchical":
                # Original hierarchical contrastive learning
                contrastive_features = self.projection_head(utt_emb)
                contrastive_loss, per_class_stats = self.hierarchical_contrastive_loss(
                    contrastive_features.unsqueeze(1), labels, return_per_class=True)
                
            elif self.contrastive_strategy == "local_patch":
                # Local patch contrastive learning
                # Use the original fullfeature tensor which has the correct shape
                # Reshape fullfeature from (B, L, T, C) to (B, L*T, C) for temporal processing
                temporal_features = fullfeature.view(B, L * T, C)  # Fixed reshaping
                contrastive_loss = self.local_patch_loss(temporal_features, labels)
                
            elif self.contrastive_strategy == "data_augment" and x_augmented is not None:
                # Data augmentation contrastive learning
                # Process augmented input
                x_aug_ssl_feat, aug_layerResult = self.ssl_model.extract_feat(x_augmented.squeeze(-1))
                _, aug_fullfeature = getAttenF(aug_layerResult)
                
                aug_layer_tokens = aug_fullfeature.contiguous().view(B * L, T, C)
                aug_layer_emb, _ = self.temporal_attn(aug_layer_tokens)
                aug_layer_emb = aug_layer_emb.view(B, L, C)

                aug_groups = torch.split(aug_layer_emb, self.group_size, dim=1)
                aug_group_vecs = []
                for g in aug_groups:
                    g_vec, _ = self.intra_attn(g)
                    g_vec = self.group_refine(g_vec)
                    aug_group_vecs.append(g_vec)

                aug_group_stack = torch.stack(aug_group_vecs, dim=1)
                aug_utt_emb, _ = self.inter_attn(aug_group_stack)
                aug_utt_emb = self.utt_refine(aug_utt_emb)
                
                # Project both original and augmented embeddings
                original_features = self.projection_head(utt_emb)
                augmented_features = self.projection_head(aug_utt_emb)
                
                contrastive_loss = self.data_augment_loss(original_features, augmented_features, labels)
            
            # Apply gradient scaling and normalization
            scaled_contrastive = contrastive_loss * self.contrastive_gradient_scale
            normalized_contrastive = scaled_contrastive / B
            
            return output, normalized_contrastive, per_class_stats
        
        return output
    
    def get_embeddings_for_visualization(self):
        """Get stored embeddings for visualization"""
        if len(self.stored_embeddings) == 0:
            return None, None
        
        embeddings = np.concatenate(self.stored_embeddings, axis=0)
        labels = np.concatenate(self.stored_labels, axis=0)
        
        return embeddings, labels
    
    def clear_stored_embeddings(self):
        """Clear stored embeddings"""
        self.stored_embeddings = []
        self.stored_labels = []

# Add L2Norm layer for projection head
class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super(L2Norm, self).__init__()
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim)

# Register the L2Norm module
nn.L2Norm = L2Norm