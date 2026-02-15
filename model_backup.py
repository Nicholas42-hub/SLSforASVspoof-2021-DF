import random
import sys
from typing import Union, Optional
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Add the fairseq path to Python path
sys.path.insert(0, "/data/gpfs/projects/punim2637/nnliang/SLSforASVspoof-2021-DF/fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1")

# Now import fairseq (should work with hydra_init disabled)
import fairseq
import fairseq.checkpoint_utils


@torch.no_grad()
def geometric_median(points: torch.Tensor, max_iter: int = 100, tol: float = 1e-5):
    """Compute the geometric median of points. Used for initializing decoder bias."""
    # Initialize our guess as the mean of the points
    guess = points.mean(dim=0)
    prev = torch.zeros_like(guess)

    # Weights for iteratively reweighted least squares
    weights = torch.ones(len(points), device=points.device)

    for _ in range(max_iter):
        prev = guess

        # Compute the weights
        weights = 1 / torch.norm(points - guess, dim=1)

        # Normalize the weights
        weights /= weights.sum()

        # Compute the new geometric median
        guess = (weights.unsqueeze(1) * points).sum(dim=0)

        # Early stopping condition
        if torch.norm(guess - prev) < tol:
            break

    return guess


class AutoEncoderTopK(nn.Module):
    """
    Top-K Sparse Autoencoder implementation from https://arxiv.org/abs/2406.04093
    """

    def __init__(self, activation_dim: int, dict_size: int, k: int):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size

        assert isinstance(k, int) and k > 0, f"k={k} must be a positive integer"
        self.register_buffer("k", torch.tensor(k, dtype=torch.int))
        self.register_buffer("threshold", torch.tensor(-1.0, dtype=torch.float32))

        self.decoder = nn.Linear(dict_size, activation_dim, bias=False)
        # Initialize decoder with unit norm
        self.decoder.weight.data = self.decoder.weight.data / torch.norm(
            self.decoder.weight.data, dim=0, keepdim=True
        )

        self.encoder = nn.Linear(activation_dim, dict_size)
        self.encoder.weight.data = self.decoder.weight.T.clone()
        self.encoder.bias.data.zero_()

        self.b_dec = nn.Parameter(torch.zeros(activation_dim))

    def encode(
        self, x: torch.Tensor, return_topk: bool = False, use_threshold: bool = False
    ):
        post_relu_feat_acts_BF = F.relu(self.encoder(x - self.b_dec))

        if use_threshold:
            encoded_acts_BF = post_relu_feat_acts_BF * (
                post_relu_feat_acts_BF > self.threshold
            )
            if return_topk:
                post_topk = post_relu_feat_acts_BF.topk(self.k, sorted=False, dim=-1)
                return (
                    encoded_acts_BF,
                    post_topk.values,
                    post_topk.indices,
                    post_relu_feat_acts_BF,
                )
            else:
                return encoded_acts_BF

        post_topk = post_relu_feat_acts_BF.topk(self.k, sorted=False, dim=-1)

        tops_acts_BK = post_topk.values
        top_indices_BK = post_topk.indices

        buffer_BF = torch.zeros_like(post_relu_feat_acts_BF)
        encoded_acts_BF = buffer_BF.scatter_(
            dim=-1, index=top_indices_BK, src=tops_acts_BK
        )

        if return_topk:
            return encoded_acts_BF, tops_acts_BK, top_indices_BK, post_relu_feat_acts_BF
        else:
            return encoded_acts_BF

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x) + self.b_dec

    def forward(self, x: torch.Tensor, output_features: bool = False):
        encoded_acts_BF = self.encode(x)
        x_hat_BD = self.decode(encoded_acts_BF)
        if not output_features:
            return x_hat_BD
        else:
            return x_hat_BD, encoded_acts_BF

    @staticmethod
    def from_pretrained(path, k: Optional[int] = None, device=None):
        """Load a pretrained autoencoder from a file."""
        state_dict = torch.load(path)
        dict_size, activation_dim = state_dict["encoder.weight"].shape

        if k is None:
            k = state_dict["k"].item()
        elif "k" in state_dict and k != state_dict["k"].item():
            raise ValueError(f"k={k} != {state_dict['k'].item()}=state_dict['k']")

        autoencoder = AutoEncoderTopK(activation_dim, dict_size, k)
        autoencoder.load_state_dict(state_dict)
        if device is not None:
            autoencoder.to(device)
        return autoencoder

class SSLModel(nn.Module):
    def __init__(self, device):
        super(SSLModel, self).__init__()
        
        cp_path = 'xlsr2_300m.pt'
        
        try:
            # Load model with strict=False to ignore missing keys
            model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
                [cp_path], 
                strict=False
            )
            
        except Exception as e:
            print(f"Error loading model: {e}")
            # Fallback to a different model
            print("Trying fallback model...")
            cp_path = 'xlsr_53_56k.pt'
            try:
                model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path], strict=False)
            except Exception as e2:
                print(f"Fallback model also failed: {e2}")
                raise RuntimeError(f"Could not load any model. Original error: {e}, Fallback error: {e2}")
        
        self.model = model[0]
        self.device = device
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

import torch
import torch.nn as nn
import torch.nn.functional as F
import fairseq
try:
    from fairseq import checkpoint_utils
except Exception as e:  # pragma: no cover
    checkpoint_utils = None
    _FAIRSEQ_IMPORT_ERROR = e

class SSLModel(nn.Module):
    def __init__(self,device,cp_path):
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
        return emb
    
class SparseAutoencoder(nn.Module):

    def __init__(self,input_dim=1024, hidden_dim=2048,sparsity_coef=1e-3, k=256):
        super(SparseAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sparsity_coef = sparsity_coef
        self.k = k
        self.encoder = nn.Linear(input_dim, hidden_dim,bias=True)

        self.decoder_bias = nn.Parameter(torch.zeros(input_dim))

        nn.init.kaiming_uniform_(self.encoder.weight, a=0, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.encoder.bias)
        
        self.register_buffer('feature_activation_count', torch.zeros(hidden_dim))
        self.register_buffer('total_samples', torch.tensor(0))

    
    def encode(self,x):
        encoded = self.encoder(x)
        encoded = torch.clamp(encoded,min=-10,max=10)
        latent = self._topk_activation(encoded, self.k)
        if self.training:
            self._update_activation_states(latent)
        return latent
    def _topk_activation(self,x,k):
        topk_values,topk_indices =torch.topk(x,k,dim=-1)
        sparse_latent = torch.zeros_like(x)
        sparse_latent.scatter_(-1,topk_indices,topk_values)
        
        sparse_latent = F.relu(sparse_latent)
        return sparse_latent
    def _update_activation_states(self,latent):
        with torch.no_grad():
            active= (latent>0).float().sum(dim=0)
            self.feature_activation_count+=active
            self.total_samples+=latent.shape[0]
    def get_dead_neuron_ratio(self) -> float:

        if self.total_samples == 0:
            return 0.0
        activation_freq = self.feature_activation_count / self.total_samples
        dead_ratio = (activation_freq == 0).float().mean().item()
        return dead_ratio
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        reconstruction = F.linear(latent,self.encoder.weight.t(),self.decoder_bias)
        return reconstruction

    def forward(self,x):
        B,T,C = x.shape
        x_flat = x.reshape(B*T,C)
        latent = self.encode(x_flat)

        reconstruction = self.decode(latent)

        recon_loss = F.mse_loss(reconstruction,x_flat,reduction='mean')

        recon_loss = torch.clamp(recon_loss,max=100)

        sparsity_loss = torch.mean(torch.abs(latent))
        sparsity_loss = torch.clamp(sparsity_loss,max=10)

        sae_loss = recon_loss + self.sparsity_coef * sparsity_loss

        # Calculate active ratio
        active_ratio = (latent > 0).float().mean().item()

        if torch.isnan(sae_loss) or torch.isinf(sae_loss):
            sae_loss = torch.tensor(0.0, device=x.device, requires_grad=True)
            recon_loss = torch.tensor(0.0, device=x.device)
            sparsity_loss = torch.tensor(0.0, device=x.device)

        loss_dict = {
            'sae_recon': recon_loss.item() if not torch.isnan(recon_loss) else 0.0,
            'sae_sparsity': sparsity_loss.item() if not torch.isnan(sparsity_loss) else 0.0,
            'sae_total': sae_loss.item() if not torch.isnan(sae_loss) else 0.0,
            'sae_active_ratio': active_ratio,
            'sae_dead_ratio': self.get_dead_neuron_ratio()
        }

        reconstruction = reconstruction.reshape(B,T,C)
        latent = latent.reshape(B,T,-1)

        return reconstruction, latent, sae_loss, loss_dict


class ModelSAE(nn.Module):
    def __init__(
        self,
        args,
        device,
        cp_path: str = '/root/autodl-tmp/SLSforASVspoof-2021-DF/xlsr2_300m.pt',
        input_dim: int = 1024,
    ):
        super(ModelSAE, self).__init__()
        self.sae_weight = float(getattr(args, 'sae_weight', 0.05))
        hidden_dim = int(getattr(args, 'sae_hidden_dim', 2048))
        k = int(getattr(args, 'sae_topk', 256)) 
        sparsity_coef = float(getattr(args, 'sae_sparsity_coef', 1e-3))

        self.d_model = int(input_dim)
        self.ssl_model = SSLModel(device=device, cp_path=cp_path)
        self.sae = SparseAutoencoder(input_dim=self.d_model, hidden_dim=hidden_dim,k=k,sparsity_coef=sparsity_coef)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, 256),
            nn.ReLU(),
            
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, input_data: torch.Tensor, labels=None):
        with torch.no_grad():
            x_ssl = self.ssl_model.extract_feat(input_data)

        x_recon, _, sae_loss, sae_loss_dict = self.sae(x_ssl)
        x_pooled = self.pool(x_recon.transpose(1, 2)).squeeze(-1)
        logits = self.classifier(x_pooled)
        output = F.log_softmax(logits, dim=-1)
        return output, sae_loss, sae_loss_dict

    def compute_total_loss(self, classification_loss: torch.Tensor, sae_loss: torch.Tensor) -> torch.Tensor:
        if not isinstance(sae_loss, torch.Tensor):
            return classification_loss
        return classification_loss + (self.sae_weight * sae_loss)


# Backward-compatible alias
Model = ModelSAE