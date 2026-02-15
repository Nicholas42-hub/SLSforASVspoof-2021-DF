import sys
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add the fairseq path to Python path
sys.path.insert(0, "/data/gpfs/projects/punim2637/nnliang/SLSforASVspoof-2021-DF/fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1")

# Disable Hydra to avoid OmegaConf version conflicts
import os
os.environ['HYDRA_FULL_ERROR'] = '1'
try:
    from hydra import initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    GlobalHydra.instance().clear()
except:
    pass

import fairseq
import fairseq.checkpoint_utils

@torch.no_grad()
def geometric_median(points: torch.Tensor, max_iter: int = 100, tol: float = 1e-5):
    """Compute the geometric median of points. Used for initializing decoder bias."""
    guess = points.mean(dim=0)
    prev = torch.zeros_like(guess)
    weights = torch.ones(len(points), device=points.device)

    for _ in range(max_iter):
        prev = guess
        weights = 1 / torch.norm(points - guess, dim=1)
        weights /= weights.sum()
        guess = (weights.unsqueeze(1) * points).sum(dim=0)
        
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

        # Decoder: dict_size -> activation_dim
        self.decoder = nn.Linear(dict_size, activation_dim, bias=False)
        self.decoder.weight.data = self.decoder.weight.data / torch.norm(
            self.decoder.weight.data, dim=0, keepdim=True
        )

        # Encoder: activation_dim -> dict_size
        self.encoder = nn.Linear(activation_dim, dict_size)
        self.encoder.weight.data = self.decoder.weight.T.clone()
        self.encoder.bias.data.zero_()

        self.b_dec = nn.Parameter(torch.zeros(activation_dim))

    def encode(self, x: torch.Tensor):
        """Encode input with TopK sparsity."""
        post_relu_feat_acts = F.relu(self.encoder(x - self.b_dec))

        # Select top-k activations
        topk_values, topk_indices = post_relu_feat_acts.topk(self.k, sorted=False, dim=-1)

        # Create sparse representation
        buffer = torch.zeros_like(post_relu_feat_acts)
        encoded_acts = buffer.scatter_(dim=-1, index=topk_indices, src=topk_values)

        return encoded_acts

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Decode sparse representation back to original dimension."""
        return self.decoder(x) + self.b_dec

    def forward(self, x: torch.Tensor):
        """Forward pass: encode and decode."""
        encoded = self.encode(x)
        reconstructed = self.decode(encoded)
        return reconstructed, encoded

    @staticmethod
    def from_pretrained(path, k: Optional[int] = None, device=None):
        """Load a pretrained autoencoder from a file."""
        state_dict = torch.load(path)
        dict_size, activation_dim = state_dict["encoder.weight"].shape

        if k is None:
            k = state_dict["k"].item()

        autoencoder = AutoEncoderTopK(activation_dim, dict_size, k)
        autoencoder.load_state_dict(state_dict)
        if device is not None:
            autoencoder.to(device)
        return autoencoder

class SSLModel(nn.Module):
    """SSL feature extractor using wav2vec 2.0 / XLS-R model."""
    
    def __init__(self, device, cp_path='xlsr2_300m.pt'):
        super(SSLModel, self).__init__()
        
        try:
            model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
                [cp_path], strict=False
            )
        except Exception as e:
            print(f"Error loading model {cp_path}: {e}")
            print("Trying fallback model...")
            cp_path = 'xlsr_53_56k.pt'
            model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
                [cp_path], strict=False
            )
        
        self.model = model[0]
        self.device = device
        self.out_dim = 1024

    def extract_feat(self, input_data):
        """Extract SSL features from audio input."""
        if next(self.model.parameters()).device != input_data.device:
            self.model.to(input_data.device, dtype=input_data.dtype)
            self.model.train()

        if input_data.ndim == 3:
            input_tmp = input_data[:, :, 0]
        else:
            input_tmp = input_data
            
        # Extract features [batch, length, dim]
        emb = self.model(input_tmp, mask=False, features_only=True)['x']
        return emb


class Model(nn.Module):
    """
    Audio Deepfake Detection Model with TopK SAE.
    Combines SSL features with TopK Sparse Autoencoder for improved representation.
    """
    
    def __init__(
        self,
        args,
        device,
        cp_path: str = 'xlsr2_300m.pt',
        use_sae: bool = True,
        use_sparse_features: bool = True,  # New: use sparse features or reconstructed features
        sae_dict_size: int = 4096,
        sae_k: int = 128,
        sae_weight: float = 0.1
    ):
        super(Model, self).__init__()
        self.device = device
        self.use_sae = use_sae
        self.use_sparse_features = use_sparse_features
        self.sae_weight = sae_weight
        
        # SSL feature extractor
        self.ssl_model = SSLModel(device=device, cp_path=cp_path)
        
        # TopK SAE
        if self.use_sae:
            self.sae = AutoEncoderTopK(
                activation_dim=1024,  # SSL output dimension
                dict_size=sae_dict_size,
                k=sae_k
            )
            input_dim = sae_dict_size if use_sparse_features else 1024
        else:
            input_dim = 1024
        
        # Pooling and classifier
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )
        
        # For interpretability analysis
        self.last_sparse_features = None
        self.last_feature_indices = None

    def forward(self, input_data: torch.Tensor, return_sae_loss: bool = True, return_interpretability: bool = False):
        """
        Forward pass.
        
        Args:
            input_data: Audio input tensor
            return_sae_loss: Whether to return SAE reconstruction loss (default: True)
            return_interpretability: Whether to return interpretability information
            
        Returns:
            output: Log softmax classification output
            sae_loss: (optional) SAE reconstruction loss if return_sae_loss=True
            interp_dict: (optional) interpretability info if return_interpretability=True
        """
        # Extract SSL features (trainable - no torch.no_grad())
        x_ssl = self.ssl_model.extract_feat(input_data)  # [B, T, 1024]
        
        sae_loss = None
        interp_dict = None
        
        if self.use_sae:
            # Flatten for SAE
            B, T, C = x_ssl.shape
            x_flat = x_ssl.reshape(B * T, C)
            
            # Apply TopK SAE
            x_recon, encoded = self.sae(x_flat)
            
            # Calculate reconstruction loss when requested (both train and eval)
            if return_sae_loss:
                sae_loss = F.mse_loss(x_recon, x_flat)
            
            # Choose between sparse features or reconstructed features
            if self.use_sparse_features:
                # Use sparse encoding (better interpretability)
                x = encoded.reshape(B, T, -1)  # [B, T, dict_size]
            else:
                # Use reconstructed features
                x = x_recon.reshape(B, T, C)  # [B, T, 1024]
            
            # Save for interpretability analysis
            if return_interpretability:
                self.last_sparse_features = encoded.reshape(B, T, -1)
                # Find activated feature indices at each time step
                active_mask = (encoded > 0).reshape(B, T, -1)
                self.last_feature_indices = active_mask
        else:
            x = x_ssl
        
        # Pool and classify
        x_pooled = self.pool(x.transpose(1, 2)).squeeze(-1)  # [B, dim]
        logits = self.classifier(x_pooled)  # [B, 2]
        output = F.log_softmax(logits, dim=-1)
        
        # Build interpretability information
        if return_interpretability and self.use_sae:
            interp_dict = self.get_interpretability_info(x_pooled)
        
        # Return results
        if return_interpretability:
            if return_sae_loss:
                return output, sae_loss, interp_dict
            return output, interp_dict
        elif return_sae_loss:
            return output, sae_loss  # sae_loss may be None during eval, but always return 2 values
        return output
    
    def get_interpretability_info(self, pooled_features):
        """
        Get interpretability information
        
        Returns:
            dict: Contains feature activation patterns, top features, and other info
        """
        if self.last_sparse_features is None:
            return None
        
        B, T, D = self.last_sparse_features.shape
        
        # Average feature activation across all time steps for each sample
        avg_activation = self.last_sparse_features.mean(dim=1)  # [B, D]
        
        # Find top-20 most important features for each sample
        topk_values, topk_indices = avg_activation.topk(k=min(20, D), dim=-1)
        
        # Calculate sparsity
        sparsity = (self.last_sparse_features > 0).float().mean(dim=[1, 2])  # [B]
        
        # Feature activation frequency (across time dimension)
        activation_freq = (self.last_sparse_features > 0).float().mean(dim=1)  # [B, D]
        
        return {
            'avg_activation': avg_activation,  # [B, D] Average activation values
            'top20_features': topk_indices,    # [B, 20] Top 20 most important features
            'top20_values': topk_values,       # [B, 20] Corresponding activation values
            'sparsity': sparsity,              # [B] Sparsity of each sample
            'activation_freq': activation_freq, # [B, D] Feature activation frequency over time
            'sparse_features': self.last_sparse_features,  # [B, T, D] Complete sparse features
        }
    
    def compute_total_loss(self, classification_loss: torch.Tensor, sae_loss: torch.Tensor = None):
        """Compute total loss including SAE reconstruction loss."""
        if sae_loss is None or not self.use_sae:
            return classification_loss
        return classification_loss + (self.sae_weight * sae_loss)
    
    @torch.no_grad()
    def analyze_feature_importance(self, dataloader, num_samples=100):
        """
        Analyze feature importance: compare feature activations between bonafide and spoofed audio
        
        Args:
            dataloader: Data loader
            num_samples: Number of samples to analyze
        
        Returns:
            dict: Feature statistics for bonafide and spoofed audio
        """
        self.eval()
        
        bonafide_features = []  # Bonafide audio
        spoof_features = []     # Spoofed audio
        
        count = 0
        for batch in dataloader:
            if count >= num_samples:
                break
            
            inputs, labels = batch[0].to(self.device), batch[1].to(self.device)
            
            # Get interpretability information
            output, interp = self(inputs, return_interpretability=True)
            
            # Separate bonafide and spoofed samples
            for i in range(len(labels)):
                if labels[i] == 1:  # Assume 1 is bonafide, 0 is spoof
                    bonafide_features.append(interp['avg_activation'][i])
                else:
                    spoof_features.append(interp['avg_activation'][i])
            
            count += len(labels)
        
        # Statistical analysis
        bonafide_features = torch.stack(bonafide_features)  # [N, D]
        spoof_features = torch.stack(spoof_features)        # [M, D]
        
        # Calculate average activation for each feature in both classes
        bonafide_mean = bonafide_features.mean(dim=0)
        spoof_mean = spoof_features.mean(dim=0)
        
        # Find most discriminative features
        diff = torch.abs(bonafide_mean - spoof_mean)
        discriminative_features = diff.topk(50)
        
        return {
            'bonafide_mean_activation': bonafide_mean,
            'spoof_mean_activation': spoof_mean,
            'most_discriminative_features': discriminative_features.indices,
            'discriminative_scores': discriminative_features.values,
            'bonafide_only_features': (bonafide_mean > spoof_mean * 2).nonzero().squeeze(),
            'spoof_only_features': (spoof_mean > bonafide_mean * 2).nonzero().squeeze(),
        }