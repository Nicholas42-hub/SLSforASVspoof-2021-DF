import torch
import torch.nn as nn
import torch.nn.functional as F

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
            features: [batch_size, feature_dim]
            labels: [batch_size]
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