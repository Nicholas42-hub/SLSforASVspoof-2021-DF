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
    def __init__(self, in_dim: int, attn_dim: int = 128, use_temporal_bias: bool = False):
        super().__init__()
        self.proj = nn.Linear(in_dim, attn_dim)
        self.score = nn.Linear(attn_dim, 1, bias=False)
        self.use_temporal_bias = use_temporal_bias

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        # x: (B, T, C)
        e = torch.tanh(self.proj(x))
        s = self.score(e).squeeze(-1)  # (B, T)
        
        # Add temporal bias to emphasize beginning and end of audio
        if self.use_temporal_bias:
            T = s.size(1)
            # Create a U-shaped bias: high at start and end, lower in middle
            # Using a quadratic function: bias = -((t - T/2) / (T/2))^2 + 1
            # This creates values from ~1.0 at edges to ~0.0 in middle
            t_indices = torch.arange(T, dtype=torch.float32, device=s.device)
            t_normalized = (t_indices - T/2) / (T/2)  # Normalize to [-1, 1]
            temporal_bias = 1.0 - t_normalized ** 2  # U-shape: 1.0 at edges, 0.0 in middle
            # Scale the bias (adjust the multiplier to control emphasis strength)
            temporal_bias = temporal_bias * 2.0  # Increase emphasis on edges
            s = s + temporal_bias.unsqueeze(0)  # Broadcast to (B, T)
        
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
    def __init__(self, temperature=0.07, margin=0.5):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        
    def forward(self, embeddings, labels):
        """
        Contrastive loss to separate spoof and real audio embeddings
        Args:
            embeddings: (B, D) normalized embeddings
            labels: (B,) binary labels (0=real, 1=spoof)
        """
        if embeddings.size(0) < 2:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
            
        # Normalize embeddings with numerical stability
        embeddings = F.normalize(embeddings, p=2, dim=1, eps=1e-8)
        
        # Compute pairwise cosine similarity
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # Clamp similarity to prevent overflow
        similarity_matrix = torch.clamp(similarity_matrix, min=-10, max=10)
        
        # Create positive and negative masks
        labels = labels.unsqueeze(1)
        positive_mask = (labels == labels.T).float()
        negative_mask = (labels != labels.T).float()
        
        # Remove self-similarity
        batch_size = embeddings.size(0)
        mask = torch.eye(batch_size, device=embeddings.device).bool()
        positive_mask.masked_fill_(mask, 0)
        
        # Check if we have positive pairs
        if torch.sum(positive_mask) == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        # Compute contrastive loss with numerical stability
        exp_sim = torch.exp(similarity_matrix)
        
        # Positive loss: minimize distance between same class samples
        pos_sim = similarity_matrix * positive_mask
        pos_loss = -torch.sum(pos_sim * positive_mask) / (torch.sum(positive_mask) + 1e-8)
        
        # Negative loss: maximize distance between different class samples  
        neg_sim = torch.log(torch.sum(exp_sim * negative_mask, dim=1) + 1e-8)
        neg_loss = torch.mean(neg_sim)
        
        total_loss = pos_loss + neg_loss
        
        # Safety check for NaN
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        return total_loss

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning Loss"""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        Args:
            features: (B, D) normalized feature embeddings
            labels: (B,) ground truth labels
        """
        device = features.device
        batch_size = features.shape[0]
        
        if batch_size < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Normalize features with numerical stability
        features = F.normalize(features, p=2, dim=1, eps=1e-8)
        
        # Compute similarity matrix
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T), self.temperature)
        
        # Clamp for numerical stability
        anchor_dot_contrast = torch.clamp(anchor_dot_contrast, min=-10, max=10)
        
        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        # Create masks
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # Remove diagonal elements
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1,
            torch.arange(batch_size).view(-1, 1).to(device), 0)
        mask = mask * logits_mask
        
        # Check if we have positive pairs
        if torch.sum(mask) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Compute log probabilities
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        
        # Compute mean of log-likelihood over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
        
        # Loss
        loss = -mean_log_prob_pos.mean()
        
        # Safety check for NaN
        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        return loss

class Model(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.device = device
        self.ssl_model = SSLModel(self.device)
        self.d_model = 1024
        self.group_size = getattr(args, "group_size", 3)  
        
        # Reduce contrastive learning weights for stability
        self.use_contrastive = getattr(args, "use_contrastive", True)
        self.contrastive_weight = getattr(args, "contrastive_weight", 0.1)  # Reduced from 0.3
        self.supcon_weight = getattr(args, "supcon_weight", 0.1)  # Reduced from 0.2

        self.temporal_attn = AttnPool(in_dim=self.d_model, attn_dim=128, use_temporal_bias=True)
        self.intra_attn = AttnPool(in_dim=self.d_model, attn_dim=128)
        self.group_refine = ResidualRefine(in_dim=self.d_model, hidden=512, dropout=0.1)
        self.inter_attn = AttnPool(in_dim=self.d_model, attn_dim=128)
        self.utt_refine = ResidualRefine(in_dim=self.d_model, hidden=512, dropout=0.1)
        
        # Store attention weights for visualization and analysis
        self.attention_weights = {
            'temporal': None,
            'intra': None,
            'inter': None
        }

        # Improved projection head with layer normalization
        self.projection_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
        )

        # Improved classifier with better regularization
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.SELU(inplace=True),
            nn.Dropout(p=0.15),
            nn.Linear(self.d_model, 256),
            nn.LayerNorm(256),
            nn.SELU(inplace=True),
            nn.Dropout(p=0.15),
            nn.Linear(256, 2),
        )
        
        # Initialize contrastive loss functions with more stable parameters
        self.contrastive_loss = ContrastiveLoss(temperature=0.1)  # Higher temperature for stability
        self.supcon_loss = SupConLoss(temperature=0.1)

        # Initialize weights properly
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for better training stability"""
        for module in [self.projection_head, self.classifier]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, x, labels=None, return_attention=False):
        try:
            x_ssl_feat, layerResult = self.ssl_model.extract_feat(x.squeeze(-1))
            _, fullfeature = getAttenF(layerResult)  # (B, L, T, C)

            B, L, T, C = fullfeature.shape
            layer_tokens = fullfeature.contiguous().view(B * L, T, C)
            layer_emb, temporal_attn = self.temporal_attn(layer_tokens)
            layer_emb = layer_emb.view(B, L, C)
            
            # Store temporal attention weights (B*L, T) -> (B, L, T)
            if return_attention:
                self.attention_weights['temporal'] = temporal_attn.view(B, L, T).detach()

            groups = torch.split(layer_emb, self.group_size, dim=1)
            group_vecs = []
            intra_attns = []
            for g in groups:
                g_vec, intra_attn = self.intra_attn(g)
                g_vec = self.group_refine(g_vec)
                group_vecs.append(g_vec)
                if return_attention:
                    intra_attns.append(intra_attn.detach())
            
            # Store intra-group attention weights
            if return_attention and len(intra_attns) > 0:
                self.attention_weights['intra'] = torch.stack(intra_attns, dim=1)  # (B, num_groups, group_size)

            group_stack = torch.stack(group_vecs, dim=1)
            utt_emb, inter_attn = self.inter_attn(group_stack)
            utt_emb = self.utt_refine(utt_emb)
            
            # Store inter-group attention weights
            if return_attention:
                self.attention_weights['inter'] = inter_attn.detach()  # (B, num_groups)

            # Add gradient clipping to embeddings
            if self.training:
                utt_emb = torch.clamp(utt_emb, min=-10, max=10)

            # Get classification logits
            logits = self.classifier(utt_emb)
            log_probs = F.log_softmax(logits, dim=1)
            
            # Don't apply logsoftmax here - let loss function handle it
            output = log_probs  # Return log probabilities
            
            # Compute both contrastive losses if labels are provided and training
            contrastive_loss = None
            supcon_loss = None
            
            if self.training and labels is not None and self.use_contrastive and B > 1:
                # Project embeddings for contrastive learning
                projected_emb = self.projection_head(utt_emb)
                
                # Compute both contrastive losses
                contrastive_loss = self.contrastive_loss(projected_emb, labels)
                supcon_loss = self.supcon_loss(projected_emb, labels)
            
            return output, contrastive_loss, supcon_loss, utt_emb
            
        except Exception as e:
            print(f"Error in model forward pass: {e}")
            # Return safe default values
            batch_size = x.size(0)
            output = torch.zeros(batch_size, 2, device=x.device, requires_grad=True)
            utt_emb = torch.zeros(batch_size, self.d_model, device=x.device, requires_grad=True)
            return output, None, None, utt_emb

    def get_embeddings(self, x):
        """Extract embeddings for analysis"""
        with torch.no_grad():
            output, _, _, embeddings = self.forward(x)
            return embeddings

    def compute_total_loss(self, classification_loss, contrastive_loss, supcon_loss):
        """Compute the combined loss with all three components"""
        total_loss = classification_loss
        
        if contrastive_loss is not None and not torch.isnan(contrastive_loss):
            total_loss += self.contrastive_weight * contrastive_loss
            
        if supcon_loss is not None and not torch.isnan(supcon_loss):
            total_loss += self.supcon_weight * supcon_loss
            
        return total_loss
    
    def aggregate_attention_importance(self):
        """
        Aggregate attention weights across all levels to compute overall temporal importance.
        Returns a single importance distribution over time frames.
        
        Returns:
            aggregated_importance: (B, T) tensor representing temporal importance for each sample
        """
        if self.attention_weights['temporal'] is None:
            raise ValueError("No attention weights available. Run forward with return_attention=True first.")
        
        # Get temporal attention: (B, L, T)
        temporal_attn = self.attention_weights['temporal']
        B, L, T = temporal_attn.shape
        
        # Method 1: Average across layers to get temporal importance
        # This gives equal weight to all layers
        temporal_importance = temporal_attn.mean(dim=1)  # (B, T)
        
        # Optional: If you want to weight by intra and inter attention
        # This gives more sophisticated importance by considering the hierarchical structure
        if self.attention_weights['intra'] is not None and self.attention_weights['inter'] is not None:
            intra_attn = self.attention_weights['intra']  # (B, num_groups, group_size)
            inter_attn = self.attention_weights['inter']  # (B, num_groups)
            
            # Reshape intra attention to layer level: (B, L)
            # Each layer's importance is weighted by its group's inter-attention and intra-attention
            num_groups = intra_attn.shape[1]
            layer_importance = []
            
            for g_idx in range(num_groups):
                group_size = intra_attn.shape[2] if g_idx < num_groups else L % self.group_size
                for l_idx in range(group_size):
                    # Layer importance = inter_attn * intra_attn
                    layer_imp = inter_attn[:, g_idx] * intra_attn[:, g_idx, l_idx]
                    layer_importance.append(layer_imp)
            
            layer_importance = torch.stack(layer_importance, dim=1)  # (B, L)
            
            # Normalize layer importance
            layer_importance = layer_importance / (layer_importance.sum(dim=1, keepdim=True) + 1e-8)
            
            # Weighted temporal importance using layer importance
            temporal_importance_weighted = (temporal_attn * layer_importance.unsqueeze(-1)).sum(dim=1)  # (B, T)
            
            return temporal_importance_weighted
        
        return temporal_importance
    
    def get_class_aggregated_importance(self, dataloader, device):
        """
        Compute aggregated attention importance separately for spoof and bonafide samples.
        Aggregates across ALL samples (not individual samples).
        
        Args:
            dataloader: DataLoader containing samples
            device: torch device
            
        Returns:
            dict with keys 'spoof' and 'bonafide', each containing aggregated importance (T,)
            representing the average temporal importance across all samples of that class
        """
        self.eval()
        spoof_temporal_attns = []  # Collect raw temporal attention
        bonafide_temporal_attns = []
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, dict):
                    x = batch['feature'].to(device)
                    labels = batch['label'].to(device)
                else:
                    x, labels = batch[0].to(device), batch[1].to(device)
                
                # Forward pass with attention collection
                _ = self.forward(x, return_attention=True)
                
                # Get temporal attention for this batch: (B, L, T)
                temporal_attn = self.attention_weights['temporal']
                
                # Separate by class and collect
                for i in range(temporal_attn.shape[0]):
                    # Average across layers for this sample: (L, T) -> (T,)
                    sample_importance = temporal_attn[i].mean(dim=0).cpu().numpy()
                    
                    if int(labels[i]) == 1:  # bonafide
                        bonafide_temporal_attns.append(sample_importance)
                    else:  # spoof
                        spoof_temporal_attns.append(sample_importance)
        
        # Aggregate across all samples: average all samples' importance
        result = {}
        if len(spoof_temporal_attns) > 0:
            # Stack all spoof samples and take mean: (num_samples, T) -> (T,)
            result['spoof'] = np.mean(np.stack(spoof_temporal_attns, axis=0), axis=0)
        if len(bonafide_temporal_attns) > 0:
            # Stack all bonafide samples and take mean: (num_samples, T) -> (T,)
            result['bonafide'] = np.mean(np.stack(bonafide_temporal_attns, axis=0), axis=0)
        
        return result