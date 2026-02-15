import sys
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import fairseq
import fairseq.checkpoint_utils
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
    With window-based TopK selection for temporal coherence.
    """

    def __init__(self, activation_dim: int, dict_size: int, k: int, window_size: int = 1):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.window_size = window_size

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

    def encode(self, x: torch.Tensor, temporal_dim: Optional[int] = None):
        """Encode input with window-based TopK sparsity.
        
        Args:
            x: Input tensor, shape (B*T, C) or (B, T, C)
            temporal_dim: If provided, reshapes flat input to (B, temporal_dim, C)
        
        Returns:
            encoded_acts: Sparse encoding with same shape as input
        """
        original_shape = x.shape
        is_3d = len(original_shape) == 3
        
        # Handle 3D input
        if is_3d:
            B, T, C = x.shape
            x_flat = x.reshape(B * T, C)
        else:
            x_flat = x
            if temporal_dim is not None:
                B = x.shape[0] // temporal_dim
                T = temporal_dim
            else:
                # Fallback to per-sample topk
                post_relu_feat_acts = F.relu(self.encoder(x_flat - self.b_dec))
                topk_values, topk_indices = post_relu_feat_acts.topk(self.k, sorted=False, dim=-1)
                buffer = torch.zeros_like(post_relu_feat_acts)
                encoded_acts = buffer.scatter_(dim=-1, index=topk_indices, src=topk_values)
                return encoded_acts
        
        # Encode all positions
        post_relu_feat_acts = F.relu(self.encoder(x_flat - self.b_dec))  # (B*T, dict_size)
        
        if self.window_size == 1:
            # Original per-timestep topk
            topk_values, topk_indices = post_relu_feat_acts.topk(self.k, sorted=False, dim=-1)
            buffer = torch.zeros_like(post_relu_feat_acts)
            encoded_acts = buffer.scatter_(dim=-1, index=topk_indices, src=topk_values)
        else:
            # Window-based topk
            post_relu_3d = post_relu_feat_acts.reshape(B, T, -1)  # (B, T, dict_size)
            encoded_acts = self._window_topk(post_relu_3d, self.k, self.window_size)
            encoded_acts = encoded_acts.reshape(B * T, -1)
        
        # Reshape back to original if needed
        if is_3d:
            encoded_acts = encoded_acts.reshape(B, T, -1)
        
        return encoded_acts
    
    def _window_topk(self, x: torch.Tensor, k: int, window_size: int):
        """Apply topk selection across windows of time steps.
        
        Args:
            x: Input tensor (B, T, dict_size)
            k: Number of top features to keep per window
            window_size: Size of the temporal window
        
        Returns:
            Sparse tensor with same shape as input
        """
        B, T, D = x.shape
        
        # Pad if necessary
        pad_size = (window_size - T % window_size) % window_size
        if pad_size > 0:
            x = F.pad(x, (0, 0, 0, pad_size))  # Pad temporal dimension
            T_padded = T + pad_size
        else:
            T_padded = T
        
        # Reshape into windows: (B, num_windows, window_size, dict_size)
        num_windows = T_padded // window_size
        x_windows = x.reshape(B, num_windows, window_size, D)
        
        # Sum activations across window for each feature
        window_sums = x_windows.sum(dim=2)  # (B, num_windows, dict_size)
        
        # Select top-k features per window
        topk_values, topk_indices = window_sums.topk(k, dim=-1, sorted=False)  # (B, num_windows, k)
        
        # Create mask for selected features
        mask = torch.zeros_like(window_sums)  # (B, num_windows, dict_size)
        mask.scatter_(dim=-1, index=topk_indices, value=1.0)
        
        # Expand mask back to original shape
        mask_expanded = mask.unsqueeze(2).expand(B, num_windows, window_size, D)  # (B, num_windows, window_size, dict_size)
        mask_flat = mask_expanded.reshape(B, T_padded, D)
        
        # Apply mask to original activations
        x_sparse = x * mask_flat
        
        # Remove padding if added
        if pad_size > 0:
            x_sparse = x_sparse[:, :T, :]
        
        return x_sparse

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Decode sparse representation back to original dimension."""
        return self.decoder(x) + self.b_dec

    def forward(self, x: torch.Tensor):
        """Forward pass: encode and decode."""
        encoded = self.encode(x)
        reconstructed = self.decode(encoded)
        return reconstructed, encoded

    @staticmethod
    def from_pretrained(path, k: Optional[int] = None, window_size: Optional[int] = None, device=None):
        """Load a pretrained autoencoder from a file."""
        state_dict = torch.load(path)
        dict_size, activation_dim = state_dict["encoder.weight"].shape

        if k is None:
            k = state_dict["k"].item()
        
        if window_size is None:
            window_size = state_dict.get("window_size", 1)  # Default to 1 for backward compatibility

        autoencoder = AutoEncoderTopK(activation_dim, dict_size, k, window_size)
        autoencoder.load_state_dict(state_dict, strict=False)  # strict=False for backward compatibility
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
    Audio Deepfake Detection Model with TopK SAE and Window-level CPC.
    Combines SSL features with TopK Sparse Autoencoder for improved representation.
    Adds Contrastive Predictive Coding at window level for temporal structure learning.
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
        sae_window_size: int = 8,  # Window size for temporal TopK
        sae_weight: float = 0.1,
        # CPC parameters
        use_cpc: bool = True,  # Whether to use CPC loss
        cpc_hidden_dim: int = 256,  # CPC projection dimension
        cpc_weight: float = 0.5,  # CPC loss weight
        cpc_temperature: float = 0.07,  # Temperature for InfoNCE
        cpc_prediction_steps: list = None,  # Multi-scale Δ, e.g., [1, 2, 4]
    ):
        super(Model, self).__init__()
        self.device = device
        self.use_sae = use_sae
        self.use_sparse_features = use_sparse_features
        self.sae_weight = sae_weight
        self.use_cpc = use_cpc
        self.cpc_weight = cpc_weight
        self.cpc_temperature = cpc_temperature
        self.sae_window_size = sae_window_size
        
        # Default prediction steps if not specified
        if cpc_prediction_steps is None:
            self.cpc_prediction_steps = [1, 2, 4]  # Multi-scale
        else:
            self.cpc_prediction_steps = cpc_prediction_steps
        
        # SSL feature extractor
        self.ssl_model = SSLModel(device=device, cp_path=cp_path)
        
        # TopK SAE with window-based selection
        if self.use_sae:
            self.sae = AutoEncoderTopK(
                activation_dim=1024,  # SSL output dimension
                dict_size=sae_dict_size,
                k=sae_k,
                window_size=sae_window_size
            )
            input_dim = sae_dict_size if use_sparse_features else 1024
        else:
            input_dim = 1024
        
        # CPC modules (only if using CPC)
        if self.use_cpc and self.use_sae:
            # Projection head: maps SAE representation to contrastive space
            self.cpc_proj = nn.Sequential(
                nn.Linear(sae_dict_size, cpc_hidden_dim),
                nn.ReLU(),
                nn.Linear(cpc_hidden_dim, cpc_hidden_dim)
            )
            
            # Predictor: predicts future window representations
            self.cpc_pred = nn.Sequential(
                nn.Linear(cpc_hidden_dim, cpc_hidden_dim),
                nn.ReLU(),
                nn.Linear(cpc_hidden_dim, cpc_hidden_dim)
            )
        
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
        self.last_window_features = None  # For CPC analysis

    def forward(self, input_data: torch.Tensor, return_sae_loss: bool = True, 
                return_cpc_loss: bool = True, return_interpretability: bool = False):
        """
        Forward pass with optional CPC loss.
        
        Args:
            input_data: Audio input tensor
            return_sae_loss: Whether to return SAE reconstruction loss (default: True)
            return_cpc_loss: Whether to return CPC loss (default: True, only if use_cpc=True)
            return_interpretability: Whether to return interpretability information
            
        Returns:
            output: Log softmax classification output
            sae_loss: (optional) SAE reconstruction loss if return_sae_loss=True
            cpc_loss: (optional) CPC loss if return_cpc_loss=True and use_cpc=True
            interp_dict: (optional) interpretability info if return_interpretability=True
        """
        # Extract SSL features (trainable - no torch.no_grad())
        x_ssl = self.ssl_model.extract_feat(input_data)  # [B, T, 1024]
        
        sae_loss = None
        cpc_loss = None
        interp_dict = None
        window_features = None  # For CPC
        
        if self.use_sae:
            # Apply TopK SAE with temporal structure (for window-based topk)
            B, T, C = x_ssl.shape
            
            # Pass 3D tensor to SAE for window-based topk
            encoded = self.sae.encode(x_ssl, temporal_dim=T)  # Returns (B, T, dict_size)
            
            # Flatten for reconstruction
            x_flat = x_ssl.reshape(B * T, C)
            encoded_flat = encoded.reshape(B * T, -1)
            x_recon = self.sae.decode(encoded_flat)
            
            # Calculate reconstruction loss when requested (both train and eval)
            if return_sae_loss:
                sae_loss = F.mse_loss(x_recon, x_flat)
            
            # Choose between sparse features or reconstructed features
            if self.use_sparse_features:
                # Use sparse encoding (better interpretability)
                x = encoded  # Already [B, T, dict_size]
            else:
                # Use reconstructed features
                x = x_recon.reshape(B, T, C)  # [B, T, 1024]
            
            # Convert frame-level features to window-level for CPC
            if self.use_cpc and return_cpc_loss:
                # Aggregate frames into windows for CPC
                window_features = self._aggregate_to_windows(encoded, self.sae_window_size)  # [B, N, dict_size]
                self.last_window_features = window_features
                
                # Compute CPC loss at window level
                cpc_loss = self.compute_cpc_loss(window_features)
            
            # Save for interpretability analysis
            if return_interpretability:
                self.last_sparse_features = encoded  # Already [B, T, dict_size]
                # Find activated feature indices at each time step
                active_mask = (encoded > 0)
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
        
        # Return results (consistent return format)
        if return_interpretability:
            if return_cpc_loss and return_sae_loss:
                return output, sae_loss, cpc_loss, interp_dict
            elif return_sae_loss:
                return output, sae_loss, interp_dict
            elif return_cpc_loss:
                return output, cpc_loss, interp_dict
            return output, interp_dict
        elif return_cpc_loss and return_sae_loss:
            return output, sae_loss, cpc_loss
        elif return_sae_loss:
            return output, sae_loss
        elif return_cpc_loss:
            return output, cpc_loss
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
    
    def _aggregate_to_windows(self, frame_features: torch.Tensor, window_size: int):
        """
        Aggregate frame-level features to window-level representations.
        
        Args:
            frame_features: [B, T, D] frame-level features
            window_size: number of frames per window
            
        Returns:
            window_features: [B, N, D] window-level features (N = num_windows)
        """
        B, T, D = frame_features.shape
        
        # Pad if necessary
        pad_size = (window_size - T % window_size) % window_size
        if pad_size > 0:
            frame_features = F.pad(frame_features, (0, 0, 0, pad_size))
            T = T + pad_size
        
        # Reshape and average within each window
        num_windows = T // window_size
        frame_features = frame_features.reshape(B, num_windows, window_size, D)
        
        # Average pooling over time dimension within each window
        window_features = frame_features.mean(dim=2)  # [B, N, D]
        
        return window_features
    
    def compute_cpc_loss(self, window_features: torch.Tensor):
        """
        Compute Contrastive Predictive Coding (CPC) loss at window level.
        
        Uses InfoNCE with cross-batch negatives to learn temporal structure.
        Multi-scale prediction steps (Δ) for robustness.
        
        Args:
            window_features: [B, N, D] window-level features
            
        Returns:
            cpc_loss: scalar tensor, average over all prediction steps
        """
        B, N, D = window_features.shape
        
        # Project to contrastive space
        s = self.cpc_proj(window_features)  # [B, N, H]
        s = F.normalize(s, dim=-1)  # L2 normalize
        
        total_loss = 0.0
        num_valid_steps = 0
        
        # Multi-scale prediction
        for delta in self.cpc_prediction_steps:
            if N <= delta:
                # Skip if not enough windows
                continue
            
            # Query: predict future from current
            q = self.cpc_pred(s[:, :-delta, :])  # [B, N-Δ, H]
            q = F.normalize(q, dim=-1)
            
            # Key: target future representations
            k = s[:, delta:, :]  # [B, N-Δ, H]
            
            # Flatten batch and time dimensions for InfoNCE
            q = q.reshape(-1, q.size(-1))  # [M, H] where M = B*(N-Δ)
            k = k.reshape(-1, k.size(-1))  # [M, H]
            
            # Compute similarity logits (cross-batch negatives)
            logits = torch.matmul(q, k.T) / self.cpc_temperature  # [M, M]
            
            # Positive samples are on the diagonal
            labels = torch.arange(logits.size(0), device=logits.device)
            
            # InfoNCE loss (cross-entropy)
            loss_nce = F.cross_entropy(logits, labels)
            
            total_loss += loss_nce
            num_valid_steps += 1
        
        if num_valid_steps == 0:
            return torch.tensor(0.0, device=window_features.device)
        
        # Average over all prediction steps
        return total_loss / num_valid_steps
    
    def compute_total_loss(self, classification_loss: torch.Tensor, 
                          sae_loss: torch.Tensor = None, 
                          cpc_loss: torch.Tensor = None):
        """
        Compute total loss including SAE reconstruction and CPC losses.
        
        L = L_cls + λ_sae * L_recon + λ_cpc * L_cpc
        
        Args:
            classification_loss: Classification cross-entropy loss
            sae_loss: SAE reconstruction loss (optional)
            cpc_loss: CPC contrastive loss (optional)
            
        Returns:
            total_loss: weighted sum of all losses
        """
        total = classification_loss
        
        if sae_loss is not None and self.use_sae:
            total = total + (self.sae_weight * sae_loss)
        
        if cpc_loss is not None and self.use_cpc:
            total = total + (self.cpc_weight * cpc_loss)
        
        return total
    
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