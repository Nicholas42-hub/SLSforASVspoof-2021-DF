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
        
        cp_path = 'xlsr2_300m.pt'
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

        
        if True:
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

class ModelContrastive(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.device = device
        self.ssl_model = SSLModel(self.device)
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)
        self.fc0 = nn.Linear(1024, 1)
        self.sig = nn.Sigmoid()
        self.fc1 = nn.Linear(22847, 1024)
        
        # Add projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        self.fc3 = nn.Linear(1024, 2)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x, return_features=False):
        x_ssl_feat, layerResult = self.ssl_model.extract_feat(x.squeeze(-1)) #layerresult = [(x,z),24个] x(201,1,1024) z(1,201,201)
        y0, fullfeature = getAttenF(layerResult)
        y0 = self.fc0(y0)
        y0 = self.sig(y0)
        y0 = y0.view(y0.shape[0], y0.shape[1], y0.shape[2], -1)
        fullfeature = fullfeature * y0
        fullfeature = torch.sum(fullfeature, 1)
        fullfeature = fullfeature.unsqueeze(dim=1)
        x = self.first_bn(fullfeature)
        x = self.selu(x)
        x = F.max_pool2d(x, (3, 3))
        x = torch.flatten(x, 1)
        features = self.fc1(x)  # Store features before final classification
        features = self.selu(features)
        
        # Classification output
        logits = self.fc3(features)
        logits = self.selu(logits)
        output = self.logsoftmax(logits)
        
        if return_features:
            # Project features for contrastive learning
            projected_features = self.projection_head(features)
            return output, projected_features
        
        return output