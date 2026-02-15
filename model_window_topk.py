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
        """Apply topk selection across windows of time steps with 50% overlap.
        
        Args:
            x: Input tensor (B, T, dict_size)
            k: Number of top features to keep per window
            window_size: Size of the temporal window
        
        Returns:
            Sparse tensor with same shape as input
        """
        B, T, D = x.shape
        
        # Use 50% overlap (stride = window_size // 2)
        stride = max(1, window_size // 2)
        
        # Calculate number of windows with overlap
        if stride >= T:
            # If stride is larger than sequence, fall back to single window
            num_windows = 1
            pad_size = 0
            T_padded = T
        else:
            num_windows = (T - window_size) // stride + 1
            # Pad to ensure last window is complete
            last_window_start = (num_windows - 1) * stride
            required_length = last_window_start + window_size
            pad_size = max(0, required_length - T)
            T_padded = T + pad_size
        
        if pad_size > 0:
            x = F.pad(x, (0, 0, 0, pad_size))
        
        # Create overlapping windows using unfold
        # Result shape: (B, D, num_windows, window_size)
        x_unfolded = x.transpose(1, 2).unfold(2, window_size, stride)
        # Transpose to: (B, num_windows, window_size, D)
        x_windows = x_unfolded.permute(0, 2, 3, 1)
        
        # Sum activations across window for each feature
        window_sums = x_windows.sum(dim=2)  # (B, num_windows, dict_size)
        
        # Select top-k features per window
        topk_values, topk_indices = window_sums.topk(k, dim=-1, sorted=False)
        
        # Create mask for selected features in each window
        mask_windows = torch.zeros_like(window_sums)  # (B, num_windows, dict_size)
        mask_windows.scatter_(dim=-1, index=topk_indices, value=1.0)
        
        # Now we need to combine overlapping selections intelligently
        # Strategy: Weight features by their activation scores across all covering windows
        # Then select top-k from the weighted scores at each timestep
        
        # Initialize vote accumulator
        feature_votes = torch.zeros(B, T_padded, D, device=x.device)
        
        # Each window "votes" for features with their summed activations
        for i in range(num_windows):
            start_idx = i * stride
            end_idx = start_idx + window_size
            
            # Get window activations and selected mask
            window_act = x_windows[:, i, :, :]  # (B, window_size, D)
            window_mask = mask_windows[:, i:i+1, :].expand(B, window_size, D)
            
            # Weighted vote: activation strength if selected, 0 otherwise
            window_vote = window_act * window_mask
            feature_votes[:, start_idx:end_idx, :] += window_vote
        
        # Now at each timestep, select top-k from voted features
        # This ensures smooth transitions at boundaries
        mask_combined = torch.zeros_like(feature_votes)
        for t in range(T_padded):
            votes_t = feature_votes[:, t, :]  # (B, D)
            # Select top-k based on vote strength
            topk_vals, topk_idx = votes_t.topk(k, dim=-1, sorted=False)
            mask_combined[:, t, :].scatter_(dim=-1, index=topk_idx, value=1.0)
        
        # Apply combined mask to original activations
        x_sparse = x * mask_combined
        
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
        sae_window_size: int = 8,  # Window size for temporal TopK
        sae_weight: float = 0.1
    ):
        super(Model, self).__init__()
        self.device = device
        self.use_sae = use_sae
        self.use_sparse_features = use_sparse_features
        self.sae_weight = sae_weight
        
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
    def analyze_temporal_stability(self, dataloader, num_samples=100, window_size=None):
        """
        Analyze temporal stability of SAE features across time.
        This addresses Caren's request to show concrete evidence of temporal instability.
        
        Args:
            dataloader: Data loader
            num_samples: Number of samples to analyze
            window_size: Window size for analysis (defaults to SAE window size)
        
        Returns:
            dict: Temporal stability metrics including:
                - feature_lifetime: Distribution of how long features stay active
                - feature_flipping_rate: How often feature indices change between windows
                - temporal_coherence: Jaccard similarity between consecutive windows
                - persistent_vs_transient: Ratio of long-lived vs short-lived features
        """
        self.eval()
        
        if window_size is None:
            window_size = self.sae.window_size if self.use_sae else 8
        
        all_lifetimes = []
        all_jaccard_scores = []
        all_feature_changes = []
        transient_counts = []
        persistent_counts = []
        
        count = 0
        for batch in dataloader:
            if count >= num_samples:
                break
            
            inputs = batch[0].to(self.device)
            
            # Get sparse features
            _, interp = self(inputs, return_interpretability=True, return_sae_loss=False)
            sparse_features = interp['sparse_features']  # [B, T, D]
            
            for sample_idx in range(sparse_features.shape[0]):
                if count >= num_samples:
                    break
                count += 1
                
                # Print progress every 10 samples
                if count % 10 == 0:
                    print(f"  Processing sample {count}/{num_samples}...", flush=True)
                
                sample_features = sparse_features[sample_idx]  # [T, D]
                T, D = sample_features.shape
                
                # Get active features at each timestep
                active_mask = (sample_features > 0)  # [T, D]
                
                # Analyze feature lifetime
                feature_lifetimes = self._compute_feature_lifetimes(active_mask)
                all_lifetimes.extend(feature_lifetimes)
                
                # Count transient (lifetime < window_size) vs persistent (lifetime >= window_size)
                transient = sum(1 for lt in feature_lifetimes if lt < window_size)
                persistent = sum(1 for lt in feature_lifetimes if lt >= window_size)
                transient_counts.append(transient)
                persistent_counts.append(persistent)
                
                # Analyze window-to-window stability
                num_windows = T // window_size
                if num_windows > 1:
                    for w in range(num_windows - 1):
                        start1 = w * window_size
                        end1 = (w + 1) * window_size
                        start2 = (w + 1) * window_size
                        end2 = (w + 2) * window_size
                        
                        if end2 <= T:
                            # Active features in window w and w+1
                            active_w1 = active_mask[start1:end1].any(dim=0)  # [D]
                            active_w2 = active_mask[start2:end2].any(dim=0)  # [D]
                            
                            # Jaccard similarity
                            intersection = (active_w1 & active_w2).sum().float()
                            union = (active_w1 | active_w2).sum().float()
                            jaccard = intersection / (union + 1e-8)
                            all_jaccard_scores.append(jaccard.item())
                            
                            # Feature change count
                            features_disappeared = (active_w1 & ~active_w2).sum().item()
                            features_appeared = (~active_w1 & active_w2).sum().item()
                            all_feature_changes.append(features_disappeared + features_appeared)
            
            count += inputs.shape[0]
        
        # Compute statistics
        all_lifetimes = torch.tensor(all_lifetimes, dtype=torch.float32)
        
        return {
            'feature_lifetime_mean': all_lifetimes.mean().item(),
            'feature_lifetime_median': all_lifetimes.median().item(),
            'feature_lifetime_std': all_lifetimes.std().item(),
            'feature_lifetime_distribution': all_lifetimes.tolist(),
            'jaccard_similarity_mean': sum(all_jaccard_scores) / (len(all_jaccard_scores) + 1e-8),
            'jaccard_similarity_scores': all_jaccard_scores,
            'feature_flipping_rate_mean': sum(all_feature_changes) / (len(all_feature_changes) + 1e-8),
            'feature_changes_per_window': all_feature_changes,
            'transient_feature_ratio': sum(transient_counts) / (sum(transient_counts) + sum(persistent_counts) + 1e-8),
            'transient_counts': transient_counts,
            'persistent_counts': persistent_counts,
        }
    
    def _compute_feature_lifetimes(self, active_mask):
        """
        Compute lifetime (consecutive activation duration) for each feature activation.
        
        Args:
            active_mask: Boolean tensor [T, D] indicating which features are active
        
        Returns:
            List of lifetimes (in timesteps) for all feature activations
        """
        T, D = active_mask.shape
        lifetimes = []
        
        for feature_idx in range(D):
            feature_active = active_mask[:, feature_idx]  # [T]
            
            # Find runs of consecutive True values
            current_lifetime = 0
            for t in range(T):
                if feature_active[t]:
                    current_lifetime += 1
                else:
                    if current_lifetime > 0:
                        lifetimes.append(current_lifetime)
                        current_lifetime = 0
            
            # Don't forget the last run
            if current_lifetime > 0:
                lifetimes.append(current_lifetime)
        
        return lifetimes
    
    @torch.no_grad()
    def analyze_feature_identity_stability(self, dataloader, num_samples=50):
        """
        Analyze feature identity stability: track which specific features activate
        in consecutive windows and measure index flipping.
        
        This directly addresses: "feature index flipping, transient activations"
        
        Args:
            dataloader: Data loader
            num_samples: Number of samples to analyze
        
        Returns:
            dict: Feature identity metrics including index overlap and turnover rates
        """
        self.eval()
        
        window_size = self.sae.window_size if self.use_sae else 8
        k = self.sae.k.item() if self.use_sae else 128
        
        index_overlap_ratios = []
        index_turnover_rates = []
        feature_set_sizes = []
        
        count = 0
        for batch in dataloader:
            if count >= num_samples:
                break
            
            inputs = batch[0].to(self.device)
            
            # Get sparse features
            _, interp = self(inputs, return_interpretability=True, return_sae_loss=False)
            sparse_features = interp['sparse_features']  # [B, T, D]
            
            for sample_idx in range(sparse_features.shape[0]):
                sample_features = sparse_features[sample_idx]  # [T, D]
                T, D = sample_features.shape
                
                num_windows = T // window_size
                if num_windows < 2:
                    continue
                
                for w in range(num_windows - 1):
                    start1 = w * window_size
                    end1 = (w + 1) * window_size
                    start2 = (w + 1) * window_size
                    end2 = (w + 2) * window_size
                    
                    if end2 <= T:
                        # Get active feature indices in each window
                        active_mask_w1 = (sample_features[start1:end1] > 0).any(dim=0)  # [D]
                        active_mask_w2 = (sample_features[start2:end2] > 0).any(dim=0)  # [D]
                        
                        indices_w1 = set(active_mask_w1.nonzero(as_tuple=True)[0].cpu().tolist())
                        indices_w2 = set(active_mask_w2.nonzero(as_tuple=True)[0].cpu().tolist())
                        
                        # Index overlap ratio
                        if len(indices_w1) > 0 or len(indices_w2) > 0:
                            overlap = len(indices_w1 & indices_w2)
                            union_size = len(indices_w1 | indices_w2)
                            overlap_ratio = overlap / (union_size + 1e-8)
                            index_overlap_ratios.append(overlap_ratio)
                            
                            # Turnover rate: what fraction of features are new
                            turnover = len(indices_w2 - indices_w1) / (len(indices_w2) + 1e-8)
                            index_turnover_rates.append(turnover)
                        
                        feature_set_sizes.append(len(indices_w1))
                        feature_set_sizes.append(len(indices_w2))
            
            count += inputs.shape[0]
        
        return {
            'index_overlap_mean': sum(index_overlap_ratios) / (len(index_overlap_ratios) + 1e-8),
            'index_overlap_std': torch.tensor(index_overlap_ratios, dtype=torch.float32).std().item() if index_overlap_ratios else 0,
            'index_overlap_distribution': index_overlap_ratios,
            'index_turnover_mean': sum(index_turnover_rates) / (len(index_turnover_rates) + 1e-8),
            'index_turnover_distribution': index_turnover_rates,
            'active_features_per_window_mean': sum(feature_set_sizes) / (len(feature_set_sizes) + 1e-8),
            'active_features_per_window_std': torch.tensor(feature_set_sizes, dtype=torch.float32).std().item() if feature_set_sizes else 0,
            'expected_k': k,
            'window_size': window_size,
        }
    
    @torch.no_grad()
    def analyze_temporal_failure_modes(self, dataloader, num_samples=20, visualize_samples=5):
        """
        Identify and quantify specific temporal failure modes.
        Provides concrete examples for the paper.
        
        Failure modes analyzed:
        1. Feature index flipping: Same position, different features
        2. Transient spikes: Features active for only 1-2 timesteps
        3. Unstable representations: High variance in feature sets across windows
        
        Args:
            dataloader: Data loader
            num_samples: Number of samples to analyze
            visualize_samples: Number of samples to return detailed traces for
        
        Returns:
            dict: Detailed failure mode analysis with example traces
        """
        self.eval()
        
        window_size = self.sae.window_size if self.use_sae else 8
        
        flipping_examples = []
        transient_spike_counts = []
        representation_variance = []
        detailed_traces = []
        
        count = 0
        trace_count = 0
        
        for batch in dataloader:
            if count >= num_samples:
                break
            
            inputs = batch[0].to(self.device)
            
            # Get sparse features
            _, interp = self(inputs, return_interpretability=True, return_sae_loss=False)
            sparse_features = interp['sparse_features']  # [B, T, D]
            
            for sample_idx in range(sparse_features.shape[0]):
                if count >= num_samples:
                    break
                    
                sample_features = sparse_features[sample_idx]  # [T, D]
                T, D = sample_features.shape
                
                # 1. Detect feature index flipping
                active_indices_per_timestep = []
                for t in range(T):
                    active_idx = (sample_features[t] > 0).nonzero(as_tuple=True)[0].cpu().tolist()
                    active_indices_per_timestep.append(set(active_idx))
                
                # Count flips between consecutive timesteps
                flips = 0
                for t in range(T - 1):
                    flips += len(active_indices_per_timestep[t] ^ active_indices_per_timestep[t + 1])
                flipping_examples.append(flips / (T - 1) if T > 1 else 0)
                
                # 2. Detect transient spikes (features active <= 2 timesteps)
                active_mask = (sample_features > 0)
                lifetimes = self._compute_feature_lifetimes(active_mask)
                transient_count = sum(1 for lt in lifetimes if lt <= 2)
                transient_spike_counts.append(transient_count / (len(lifetimes) + 1e-8))
                
                # 3. Compute representation variance across windows
                num_windows = T // window_size
                if num_windows > 1:
                    window_active_counts = []
                    for w in range(num_windows):
                        start = w * window_size
                        end = (w + 1) * window_size
                        if end <= T:
                            active_count = active_mask[start:end].any(dim=0).sum().item()
                            window_active_counts.append(active_count)
                    
                    if len(window_active_counts) > 1:
                        variance = torch.tensor(window_active_counts, dtype=torch.float32).var().item()
                        representation_variance.append(variance)
                
                # Save detailed trace for visualization
                if trace_count < visualize_samples:
                    trace = {
                        'sample_idx': count,
                        'timesteps': T,
                        'active_indices_per_timestep': active_indices_per_timestep[:50],  # First 50 timesteps
                        'feature_lifetimes': lifetimes[:100],  # First 100 feature activations
                        'window_feature_counts': window_active_counts if num_windows > 1 else [],
                    }
                    detailed_traces.append(trace)
                    trace_count += 1
                
                count += 1
        
        return {
            'flipping_rate_mean': sum(flipping_examples) / (len(flipping_examples) + 1e-8),
            'flipping_rate_std': torch.tensor(flipping_examples).std().item() if flipping_examples else 0,
            'flipping_rates': flipping_examples,
            'transient_spike_ratio_mean': sum(transient_spike_counts) / (len(transient_spike_counts) + 1e-8),
            'transient_spike_ratios': transient_spike_counts,
            'representation_variance_mean': sum(representation_variance) / (len(representation_variance) + 1e-8),
            'representation_variances': representation_variance,
            'detailed_traces': detailed_traces,
            'num_samples_analyzed': count,
        }
    
    @torch.no_grad()
    def analyze_window_boundary_discontinuity(self, dataloader, num_samples=50):
        """
        LIMITATION 1: Detect boundary discontinuity problem.
        
        Tests whether feature changes at window boundaries are larger than within windows.
        This reveals if the hard window constraint creates artificial discontinuities.
        
        Args:
            dataloader: Data loader
            num_samples: Number of samples to analyze
            
        Returns:
            dict: Boundary vs within-window stability metrics
        """
        self.eval()
        
        window_size = self.sae.window_size if self.use_sae else 8
        
        boundary_jaccard = []  # Jaccard at window boundaries (t % window_size == 0)
        within_jaccard = []    # Jaccard within windows
        boundary_changes = []  # Feature changes at boundaries
        within_changes = []    # Feature changes within windows
        
        count = 0
        for batch in dataloader:
            if count >= num_samples:
                break
            
            inputs = batch[0].to(self.device)
            _, interp = self(inputs, return_interpretability=True, return_sae_loss=False)
            sparse_features = interp['sparse_features']  # [B, T, D]
            
            for sample_idx in range(sparse_features.shape[0]):
                if count >= num_samples:
                    break
                    
                sample_features = sparse_features[sample_idx]  # [T, D]
                T, D = sample_features.shape
                active_mask = (sample_features > 0)  # [T, D]
                
                # Compare consecutive timesteps
                for t in range(T - 1):
                    active_t = active_mask[t]  # [D]
                    active_t1 = active_mask[t + 1]  # [D]
                    
                    # Jaccard similarity
                    intersection = (active_t & active_t1).sum().float()
                    union = (active_t | active_t1).sum().float()
                    jaccard = (intersection / (union + 1e-8)).item()
                    
                    # Feature change count
                    changes = ((active_t & ~active_t1).sum() + (~active_t & active_t1).sum()).item()
                    
                    # Check if this is a boundary transition
                    if (t + 1) % window_size == 0:
                        boundary_jaccard.append(jaccard)
                        boundary_changes.append(changes)
                    else:
                        within_jaccard.append(jaccard)
                        within_changes.append(changes)
                
                count += 1
        
        return {
            'boundary_jaccard_mean': sum(boundary_jaccard) / (len(boundary_jaccard) + 1e-8),
            'within_jaccard_mean': sum(within_jaccard) / (len(within_jaccard) + 1e-8),
            'boundary_jaccard_std': torch.tensor(boundary_jaccard, dtype=torch.float32).std().item() if boundary_jaccard else 0,
            'within_jaccard_std': torch.tensor(within_jaccard, dtype=torch.float32).std().item() if within_jaccard else 0,
            'boundary_changes_mean': sum(boundary_changes) / (len(boundary_changes) + 1e-8),
            'within_changes_mean': sum(within_changes) / (len(within_changes) + 1e-8),
            'discontinuity_score': (sum(within_jaccard) / len(within_jaccard) - sum(boundary_jaccard) / len(boundary_jaccard)) if boundary_jaccard and within_jaccard else 0,
            'interpretation': 'Higher discontinuity_score (>0.05) indicates boundary problem'
        }
    
    @torch.no_grad()
    def analyze_semantic_drift(self, dataloader, num_samples=50, top_k_features=100):
        """
        LIMITATION 3: Detect semantic drift of feature representations.
        
        Tests whether the same feature index represents consistent semantic content
        across different windows/contexts.
        
        Method: For each frequently-activated feature, compute cosine similarity
        of its activation contexts (surrounding features) across occurrences.
        
        Args:
            dataloader: Data loader
            num_samples: Number of samples to analyze
            top_k_features: Analyze top K most frequent features
            
        Returns:
            dict: Semantic consistency metrics for features
        """
        self.eval()
        
        window_size = self.sae.window_size if self.use_sae else 8
        
        # Collect activation contexts for each feature
        feature_contexts = {}  # {feature_idx: [context_vectors]}
        
        count = 0
        for batch in dataloader:
            if count >= num_samples:
                break
            
            inputs = batch[0].to(self.device)
            _, interp = self(inputs, return_interpretability=True, return_sae_loss=False)
            sparse_features = interp['sparse_features']  # [B, T, D]
            
            for sample_idx in range(sparse_features.shape[0]):
                if count >= num_samples:
                    break
                    
                sample_features = sparse_features[sample_idx]  # [T, D]
                T, D = sample_features.shape
                
                # For each window, record which features co-occur
                num_windows = T // window_size
                for w in range(num_windows):
                    start = w * window_size
                    end = (w + 1) * window_size
                    if end <= T:
                        window_features = sample_features[start:end]  # [window_size, D]
                        
                        # Get active features in this window
                        active_in_window = (window_features.sum(dim=0) > 0)  # [D]
                        active_indices = active_in_window.nonzero(as_tuple=True)[0]
                        
                        # For each active feature, record its context (other active features)
                        for feat_idx in active_indices:
                            feat_idx = feat_idx.item()
                            if feat_idx not in feature_contexts:
                                feature_contexts[feat_idx] = []
                            
                            # Context = binary vector of co-occurring features
                            context = active_in_window.float().cpu()
                            context[feat_idx] = 0  # Remove self
                            feature_contexts[feat_idx].append(context)
                
                count += 1
        
        # Analyze semantic consistency for top-K frequent features
        feature_frequencies = {k: len(v) for k, v in feature_contexts.items()}
        top_features = sorted(feature_frequencies.items(), key=lambda x: x[1], reverse=True)[:top_k_features]
        
        semantic_consistency_scores = []
        for feat_idx, freq in top_features:
            if freq < 2:
                continue
            
            contexts = torch.stack(feature_contexts[feat_idx])  # [num_occurrences, D]
            
            # Compute pairwise cosine similarities between contexts
            contexts_norm = F.normalize(contexts, dim=1)
            similarity_matrix = torch.mm(contexts_norm, contexts_norm.T)  # [N, N]
            
            # Average similarity (excluding diagonal)
            mask = ~torch.eye(similarity_matrix.size(0), dtype=torch.bool)
            avg_similarity = similarity_matrix[mask].mean().item()
            
            semantic_consistency_scores.append(avg_similarity)
        
        return {
            'semantic_consistency_mean': sum(semantic_consistency_scores) / (len(semantic_consistency_scores) + 1e-8),
            'semantic_consistency_std': torch.tensor(semantic_consistency_scores, dtype=torch.float32).std().item() if semantic_consistency_scores else 0,
            'consistency_scores': semantic_consistency_scores,
            'num_features_analyzed': len(semantic_consistency_scores),
            'interpretation': 'Low consistency (<0.3) indicates semantic drift problem'
        }
    
    @torch.no_grad()
    def analyze_discriminative_transients(self, dataloader, num_samples=100):
        """
        LIMITATION 4: Identify discriminative transient features (IMPROVED VERSION).
        
        Tests whether short-lived (transient) features are important for classification.
        If so, window constraint may be over-smoothing discriminative signals.
        
        Method: Extract multi-dimensional statistics from transient vs persistent features,
        then use logistic regression to evaluate their discriminative power via AUC.
        
        Args:
            dataloader: Data loader
            num_samples: Number of samples to analyze
            
        Returns:
            dict: Discriminative power metrics including AUC and accuracy
        """
        self.eval()
        
        window_size = self.sae.window_size if self.use_sae else 8
        
        # Collect feature statistics and labels
        transient_feature_stats = []  # Multi-dimensional stats for transient features
        persistent_feature_stats = []  # Multi-dimensional stats for persistent features
        all_labels = []
        
        count = 0
        print(f"  Collecting transient/persistent feature data...", flush=True)
        
        for batch in dataloader:
            if count >= num_samples:
                break
            
            inputs, labels = batch[0].to(self.device), batch[1].to(self.device)
            _, interp = self(inputs, return_interpretability=True, return_sae_loss=False)
            sparse_features = interp['sparse_features']  # [B, T, D]
            
            for sample_idx in range(sparse_features.shape[0]):
                if count >= num_samples:
                    break
                    
                sample_features = sparse_features[sample_idx]  # [T, D]
                T, D = sample_features.shape
                active_mask = (sample_features > 0)  # [T, D]
                
                # Calculate lifetime for each feature
                transient_mask = torch.zeros(D, dtype=torch.bool, device=self.device)
                persistent_mask = torch.zeros(D, dtype=torch.bool, device=self.device)
                
                for feat_idx in range(D):
                    feature_active = active_mask[:, feat_idx]
                    
                    if not feature_active.any():
                        continue
                    
                    # Compute consecutive activation segment lengths
                    lifetimes = []
                    current_lifetime = 0
                    
                    for t in range(T):
                        if feature_active[t]:
                            current_lifetime += 1
                        else:
                            if current_lifetime > 0:
                                lifetimes.append(current_lifetime)
                                current_lifetime = 0
                    
                    if current_lifetime > 0:
                        lifetimes.append(current_lifetime)
                    
                    if lifetimes:
                        median_lifetime = torch.tensor(lifetimes, dtype=torch.float32).median().item()
                        
                        if median_lifetime < window_size:
                            transient_mask[feat_idx] = True
                        else:
                            persistent_mask[feat_idx] = True
                
                # Extract multi-dimensional statistics
                def compute_feature_stats(mask):
                    """Compute 4 statistical features"""
                    if not mask.any():
                        return torch.zeros(4, device=self.device)
                    
                    selected_features = sample_features[:, mask]  # [T, num_selected]
                    
                    mean_activation = selected_features.mean()
                    max_activation = selected_features.max()
                    activation_freq = (selected_features > 0).float().mean()
                    activation_var = selected_features.var()
                    
                    return torch.tensor([
                        mean_activation,
                        max_activation,
                        activation_freq,
                        activation_var
                    ], device=self.device)
                
                transient_stats = compute_feature_stats(transient_mask)
                persistent_stats = compute_feature_stats(persistent_mask)
                
                transient_feature_stats.append(transient_stats.cpu())
                persistent_feature_stats.append(persistent_stats.cpu())
                all_labels.append(labels[sample_idx].cpu())
                
                count += 1
                
                if count % 20 == 0:
                    print(f"    Processed {count}/{num_samples} samples", flush=True)
        
        # Convert to tensors
        transient_features = torch.stack(transient_feature_stats)  # [N, 4]
        persistent_features = torch.stack(persistent_feature_stats)  # [N, 4]
        labels_tensor = torch.stack(all_labels)  # [N]
        
        print(f"  Label distribution: bonafide={((labels_tensor==0).sum()).item()}, spoof={((labels_tensor==1).sum()).item()}", flush=True)
        
        # Check data validity
        if transient_features.abs().sum() == 0:
            print(f"  WARNING: No transient features detected!", flush=True)
            return {
                'transient_discriminative_power': 0.0,
                'persistent_discriminative_power': 0.0,
                'ratio': 0.0,
                'num_samples': count,
                'error': 'No transient features detected',
                'interpretation': 'Ratio >0.5 suggests transients are discriminative, may be over-smoothed'
            }
        
        if persistent_features.abs().sum() == 0:
            print(f"  WARNING: No persistent features detected!", flush=True)
            return {
                'transient_discriminative_power': 0.0,
                'persistent_discriminative_power': 0.0,
                'ratio': 0.0,
                'num_samples': count,
                'error': 'No persistent features detected',
                'interpretation': 'Ratio >0.5 suggests transients are discriminative, may be over-smoothed'
            }
        
        # Method: Logistic Regression with AUC evaluation
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import roc_auc_score, accuracy_score
            
            # Split train/test
            indices = torch.randperm(len(labels_tensor))
            split_idx = int(0.7 * len(indices))
            train_idx = indices[:split_idx]
            test_idx = indices[split_idx:]
            
            def evaluate_features(features, name):
                """Evaluate discriminative power of features"""
                X_train = features[train_idx].numpy()
                y_train = labels_tensor[train_idx].numpy()
                X_test = features[test_idx].numpy()
                y_test = labels_tensor[test_idx].numpy()
                
                # Standardization
                mean = X_train.mean(axis=0, keepdims=True)
                std = X_train.std(axis=0, keepdims=True) + 1e-8
                X_train = (X_train - mean) / std
                X_test = (X_test - mean) / std
                
                # Train classifier
                clf = LogisticRegression(random_state=42, max_iter=1000)
                clf.fit(X_train, y_train)
                
                # Evaluate
                y_pred = clf.predict(X_test)
                y_prob = clf.predict_proba(X_test)[:, 1]
                
                acc = accuracy_score(y_test, y_pred)
                
                # Compute AUC only if both classes present
                if len(set(y_test)) > 1:
                    auc = roc_auc_score(y_test, y_prob)
                else:
                    auc = 0.5
                
                return acc, auc
            
            transient_acc, transient_auc = evaluate_features(transient_features, "transient")
            persistent_acc, persistent_auc = evaluate_features(persistent_features, "persistent")
            
            # Use AUC as main metric
            transient_power = transient_auc
            persistent_power = persistent_auc
            ratio = transient_power / (persistent_power + 1e-8)
            
            return {
                'transient_discriminative_power': float(transient_power),
                'persistent_discriminative_power': float(persistent_power),
                'ratio': float(ratio),
                'transient_accuracy': float(transient_acc),
                'persistent_accuracy': float(persistent_acc),
                'num_samples': count,
                'num_transient_samples': int((transient_features.abs().sum(dim=1) > 0).sum().item()),
                'num_persistent_samples': int((persistent_features.abs().sum(dim=1) > 0).sum().item()),
                'interpretation': 'Ratio >0.5 suggests transients are discriminative, may be over-smoothed'
            }
        
        except ImportError:
            print(f"  WARNING: sklearn not available, falling back to correlation method", flush=True)
            # Fallback to simple correlation
            def compute_correlation_safe(features, labels):
                correlations = []
                for i in range(features.shape[1]):
                    feat_col = features[:, i]
                    if feat_col.std() < 1e-6:
                        continue
                    feat_centered = feat_col - feat_col.mean()
                    label_centered = labels.float() - labels.float().mean()
                    corr = (feat_centered * label_centered).mean() / (feat_col.std() * labels.float().std() + 1e-8)
                    correlations.append(abs(corr.item()))
                return torch.tensor(correlations).mean().item() if correlations else 0.0
            
            transient_corr = compute_correlation_safe(transient_features, labels_tensor)
            persistent_corr = compute_correlation_safe(persistent_features, labels_tensor)
            
            return {
                'transient_discriminative_power': transient_corr,
                'persistent_discriminative_power': persistent_corr,
                'ratio': transient_corr / (persistent_corr + 1e-8),
                'num_samples': count,
                'method': 'correlation_fallback',
                'interpretation': 'Ratio >0.5 suggests transients are discriminative, may be over-smoothed'
            }
    
    @torch.no_grad()
    def analyze_multi_scale_temporal_structure(self, dataloader, num_samples=50):
        """
        LIMITATION 2: Analyze whether fixed window size is optimal.
        
        Tests temporal stability at multiple scales to see if different
        window sizes would be more appropriate for different features/contexts.
        
        Args:
            dataloader: Data loader
            num_samples: Number of samples to analyze
            
        Returns:
            dict: Stability metrics at multiple temporal scales
        """
        self.eval()
        
        # Test different window sizes
        test_window_sizes = [2, 4, 8, 16, 32]
        results_by_scale = {}
        
        count = 0
        for batch in dataloader:
            if count >= num_samples:
                break
            
            inputs = batch[0].to(self.device)
            _, interp = self(inputs, return_interpretability=True, return_sae_loss=False)
            sparse_features = interp['sparse_features']  # [B, T, D]
            
            for sample_idx in range(sparse_features.shape[0]):
                if count >= num_samples:
                    break
                    
                sample_features = sparse_features[sample_idx]  # [T, D]
                T, D = sample_features.shape
                active_mask = (sample_features > 0)  # [T, D]
                
                # Measure stability at each scale
                for ws in test_window_sizes:
                    if ws not in results_by_scale:
                        results_by_scale[ws] = []
                    
                    num_windows = T // ws
                    if num_windows < 2:
                        continue
                    
                    window_similarities = []
                    for w in range(num_windows - 1):
                        start1 = w * ws
                        end1 = (w + 1) * ws
                        start2 = (w + 1) * ws
                        end2 = (w + 2) * ws
                        
                        if end2 <= T:
                            active_w1 = active_mask[start1:end1].any(dim=0)  # [D]
                            active_w2 = active_mask[start2:end2].any(dim=0)  # [D]
                            
                            intersection = (active_w1 & active_w2).sum().float()
                            union = (active_w1 | active_w2).sum().float()
                            jaccard = (intersection / (union + 1e-8)).item()
                            window_similarities.append(jaccard)
                    
                    if window_similarities:
                        avg_similarity = sum(window_similarities) / len(window_similarities)
                        results_by_scale[ws].append(avg_similarity)
                
                count += 1
        
        # Compute average stability for each scale
        scale_analysis = {}
        for ws, similarities in results_by_scale.items():
            if similarities:
                scale_analysis[f'window_{ws}'] = {
                    'mean_jaccard': sum(similarities) / len(similarities),
                    'std_jaccard': torch.tensor(similarities, dtype=torch.float32).std().item()
                }
        
        # Find optimal window size
        optimal_ws = max(scale_analysis.keys(), 
                        key=lambda k: scale_analysis[k]['mean_jaccard'])
        
        return {
            'scale_analysis': scale_analysis,
            'optimal_window_size': optimal_ws,
            'current_window_size': f'window_{self.sae.window_size if self.use_sae else 8}',
            'interpretation': 'If optimal differs from current, fixed window size is suboptimal'
        }
    
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
            output, interp = self(inputs, return_interpretability=True, return_sae_loss=False)
            
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