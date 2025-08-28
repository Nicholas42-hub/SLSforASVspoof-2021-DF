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

class ContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss for binary classification
    """
    def __init__(self, temperature=0.1):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        
    def forward(self, features, labels):
        """
        Args:
            features: [batch_size, feature_dim] - L2 normalized features
            labels: [batch_size] - binary labels (0 or 1)
        """
        batch_size = features.size(0)
        
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create positive mask (same class)
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features.device)
        
        # Remove diagonal elements
        mask = mask - torch.eye(batch_size).to(features.device)
        
        # For numerical stability
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()
        
        # Compute log probabilities
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)
        
        # Compute mean of positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        
        # Loss
        loss = -mean_log_prob_pos.mean()
        
        return loss

class CombinedLoss(nn.Module):
    """
    Combined Cross-Entropy and Contrastive Loss
    """
    def __init__(self, weight=None, temperature=0.1, contrastive_weight=0.1):
        super(CombinedLoss, self).__init__()
        self.cce_loss = nn.CrossEntropyLoss(weight=weight)
        self.contrastive_loss = ContrastiveLoss(temperature=temperature)
        self.contrastive_weight = contrastive_weight
        
    def forward(self, logits, features, labels):
        """
        Args:
            logits: [batch_size, num_classes] - classification logits
            features: [batch_size, feature_dim] - features for contrastive learning
            labels: [batch_size] - ground truth labels
        """
        # Cross-entropy loss
        cce = self.cce_loss(logits, labels)
        
        # Contrastive loss
        contrastive = self.contrastive_loss(features, labels)
        
        # Combined loss
        total_loss = cce + self.contrastive_weight * contrastive
        
        return total_loss, cce, contrastive

class ModelHierarchicalContrastive(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.device = device
        self.ssl_model = SSLModel(self.device)
        self.d_model = 1024
        self.group_size = getattr(args, "group_size", 3)  

        self.temporal_attn = AttnPool(in_dim=self.d_model, attn_dim=128)
        self.intra_attn = AttnPool(in_dim=self.d_model, attn_dim=128)
        self.group_refine = ResidualRefine(in_dim=self.d_model, hidden=512, dropout=0.1)
        self.inter_attn = AttnPool(in_dim=self.d_model, attn_dim=128)
        self.utt_refine = ResidualRefine(in_dim=self.d_model, hidden=512, dropout=0.1)

        # Add projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(self.d_model, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128)
        )

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

    def forward(self, x, return_features=False):
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

        # Classification output
        logits = self.classifier(utt_emb)
        output = self.logsoftmax(logits)
        
        if return_features:
            # Project features for contrastive learning
            projected_features = self.projection_head(utt_emb)
            return output, projected_features
        
        return output