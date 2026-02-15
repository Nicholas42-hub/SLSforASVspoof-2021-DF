"""
Phase 1: Decision Cue Consistency Analysis
Direction #4 from Caren's feedback: Cue consistency analysis and feedback

This script implements:
1. Feature Attribution Analysis - identify decision-relevant SAE features
2. Decision-Relevant Stability - measure temporal stability for high-influence features
3. Cue Consistency Metrics - measure decision cue reuse within utterances
4. Class-Specific Analysis - compare genuine vs. spoof decision strategies

Usage:
    python analyze_decision_relevance.py \
        --model_path checkpoints/best_model.pth \
        --output_dir decision_analysis \
        --num_samples 200 \
        --top_k_features 50
"""

import argparse
import json
import os
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Subset, Dataset
from tqdm import tqdm
from collections import defaultdict

from data_utils_SSL import genSpoof_list, Dataset_ASVspoof2019_train
from model_window_topk import Model


class FeatureAttributionAnalyzer:
    """Compute gradient-based attribution scores for SAE features."""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()
        
    def compute_gradient_attribution(self, audio, label):
        """
        Compute ∂logits/∂SAE_activations for each feature.
        
        Args:
            audio: Input audio tensor [B, samples]
            label: Ground truth labels [B]
            
        Returns:
            attribution_scores: [dict_size] - importance of each SAE feature
            sae_activations: [B, T, dict_size] - the SAE activations
        """
        audio = audio.to(self.device)
        audio.requires_grad_(True)
        
        # Forward pass through SSL model
        with torch.enable_grad():
            x_ssl = self.model.ssl_model.extract_feat(audio)  # [B, T, 1024]
            B, T, C = x_ssl.shape
            
            # Encode with SAE - keep gradient tracking
            sae_activations = self.model.sae.encode(x_ssl, temporal_dim=T)  # [B, T, dict_size]
            sae_activations.requires_grad_(True)
            sae_activations.retain_grad()
            
            # Decode and classify
            sae_flat = sae_activations.reshape(B * T, -1)
            reconstructed = self.model.sae.decode(sae_flat).reshape(B, T, C)
            
            # Use sparse features if model is configured that way
            if self.model.use_sparse_features:
                features_for_classifier = sae_activations
            else:
                features_for_classifier = reconstructed
            
            # Pool and classify
            pooled = self.model.pool(features_for_classifier.transpose(1, 2)).squeeze(-1)
            logits = self.model.classifier(pooled)  # [B, 2]
            
            # Backward pass - use prediction confidence for the true class
            # This measures: "how much does each feature contribute to correct classification?"
            target_logits = logits.gather(1, label.unsqueeze(1).to(self.device))
            target_logits.sum().backward()
            
            # Extract gradients
            gradients = sae_activations.grad  # [B, T, dict_size]
            
            # Attribution = absolute gradient magnitude, averaged over batch and time
            attribution = gradients.abs().mean(dim=(0, 1))  # [dict_size]
            
        return attribution.detach().cpu(), sae_activations.detach().cpu()
    
    def compute_ablation_attribution(self, audio, label, top_k_indices):
        """
        Validate gradient attribution using ablation.
        
        For top-K features, measure prediction change when feature is zeroed out.
        This is slower but gives causal evidence.
        
        Args:
            audio: Input audio tensor [B, samples]
            label: Ground truth labels [B]
            top_k_indices: Indices of features to ablate
            
        Returns:
            ablation_scores: [top_k] - prediction change per feature
        """
        self.model.eval()
        audio = audio.to(self.device)
        
        with torch.no_grad():
            # Original prediction
            x_ssl = self.model.ssl_model.extract_feat(audio)
            B, T, C = x_ssl.shape
            sae_activations = self.model.sae.encode(x_ssl, temporal_dim=T)
            
            sae_flat = sae_activations.reshape(B * T, -1)
            reconstructed = self.model.sae.decode(sae_flat).reshape(B, T, C)
            
            if self.model.use_sparse_features:
                features = sae_activations
            else:
                features = reconstructed
            
            pooled_orig = self.model.pool(features.transpose(1, 2)).squeeze(-1)
            logits_orig = self.model.classifier(pooled_orig)
            probs_orig = F.softmax(logits_orig, dim=-1)
            
            # Get original prediction for true class
            true_class_prob_orig = probs_orig.gather(1, label.unsqueeze(1).to(self.device))
        
        ablation_scores = []
        
        # Ablate each top-K feature
        for feature_idx in tqdm(top_k_indices, desc="Ablating features"):
            with torch.no_grad():
                sae_activations_ablated = sae_activations.clone()
                sae_activations_ablated[:, :, feature_idx] = 0  # Zero out this feature
                
                sae_flat = sae_activations_ablated.reshape(B * T, -1)
                reconstructed = self.model.sae.decode(sae_flat).reshape(B, T, C)
                
                if self.model.use_sparse_features:
                    features = sae_activations_ablated
                else:
                    features = reconstructed
                
                pooled = self.model.pool(features.transpose(1, 2)).squeeze(-1)
                logits = self.model.classifier(pooled)
                probs = F.softmax(logits, dim=-1)
                
                true_class_prob = probs.gather(1, label.unsqueeze(1).to(self.device))
                
                # Ablation effect = drop in true class probability
                effect = (true_class_prob_orig - true_class_prob).abs().mean().item()
                ablation_scores.append(effect)
        
        return np.array(ablation_scores)


class DecisionCueStabilityAnalyzer:
    """Analyze temporal stability of decision-relevant features."""
    
    def __init__(self, window_size):
        self.window_size = window_size
    
    def compute_jaccard_similarity(self, activations, mask=None):
        """
        Compute Jaccard similarity between consecutive timesteps.
        
        Args:
            activations: [T, dict_size] - SAE activations over time
            mask: [dict_size] - binary mask for which features to consider
            
        Returns:
            jaccard_scores: [T-1] - similarity between consecutive frames
        """
        if mask is not None:
            activations = activations[:, mask]
        
        # Binarize: active (>0) or not
        active = (activations > 0).float()
        
        jaccard_scores = []
        for t in range(active.shape[0] - 1):
            set_t = active[t]
            set_t1 = active[t + 1]
            
            intersection = (set_t * set_t1).sum()
            union = ((set_t + set_t1) > 0).float().sum()
            
            if union > 0:
                jaccard = (intersection / union).item()
            else:
                jaccard = 0.0
            
            jaccard_scores.append(jaccard)
        
        return np.array(jaccard_scores)
    
    def compute_feature_lifetime(self, activations, mask=None):
        """
        Compute how long each feature stays active.
        
        Args:
            activations: [T, dict_size]
            mask: [dict_size] - which features to consider
            
        Returns:
            lifetimes: [num_features] - average consecutive activation length
        """
        if mask is not None:
            activations = activations[:, mask]
        
        active = (activations > 0).float().numpy()
        T, D = active.shape
        
        lifetimes = []
        for feature_idx in range(D):
            feature_active = active[:, feature_idx]
            
            # Find runs of consecutive 1s
            runs = []
            current_run = 0
            for t in range(T):
                if feature_active[t] == 1:
                    current_run += 1
                else:
                    if current_run > 0:
                        runs.append(current_run)
                        current_run = 0
            if current_run > 0:
                runs.append(current_run)
            
            if runs:
                lifetimes.append(np.mean(runs))
            else:
                lifetimes.append(0)
        
        return np.array(lifetimes)
    
    def compute_flipping_rate(self, activations, mask=None):
        """
        Count how many features change state per timestep.
        
        Args:
            activations: [T, dict_size]
            mask: [dict_size]
            
        Returns:
            flipping_rates: [T-1] - number of features that flip per timestep
        """
        if mask is not None:
            activations = activations[:, mask]
        
        active = (activations > 0).float()
        
        flips = []
        for t in range(active.shape[0] - 1):
            # Count features that changed state
            changed = (active[t] != active[t + 1]).float().sum().item()
            flips.append(changed)
        
        return np.array(flips)
    
    def analyze_boundary_effects(self, activations, mask=None):
        """
        Compare stability at window boundaries vs. interior.
        
        Args:
            activations: [T, dict_size]
            mask: [dict_size]
            
        Returns:
            dict with boundary_jaccard, interior_jaccard, cross_boundary_jaccard
        """
        T = activations.shape[0]
        
        # Identify boundary frames (±2 frames from window boundary)
        boundary_margin = 2
        boundary_frames = set()
        for i in range(0, T, self.window_size):
            for offset in range(-boundary_margin, boundary_margin + 1):
                frame = i + offset
                if 0 <= frame < T:
                    boundary_frames.add(frame)
        
        # Interior frames
        interior_frames = set(range(T)) - boundary_frames
        
        # Compute Jaccard for different regions
        all_jaccard = self.compute_jaccard_similarity(activations, mask)
        
        boundary_jaccard = []
        interior_jaccard = []
        cross_boundary_jaccard = []
        
        for t in range(T - 1):
            jaccard = all_jaccard[t]
            
            # Check if this transition crosses a window boundary
            is_cross_boundary = (t + 1) % self.window_size == 0
            
            if is_cross_boundary:
                cross_boundary_jaccard.append(jaccard)
            elif t in boundary_frames or (t + 1) in boundary_frames:
                boundary_jaccard.append(jaccard)
            elif t in interior_frames and (t + 1) in interior_frames:
                interior_jaccard.append(jaccard)
        
        return {
            'boundary': np.mean(boundary_jaccard) if boundary_jaccard else 0,
            'interior': np.mean(interior_jaccard) if interior_jaccard else 0,
            'cross_boundary': np.mean(cross_boundary_jaccard) if cross_boundary_jaccard else 0,
            'boundary_scores': boundary_jaccard,
            'interior_scores': interior_jaccard,
            'cross_boundary_scores': cross_boundary_jaccard
        }


class CueConsistencyAnalyzer:
    """Measure decision cue reuse patterns."""
    
    def compute_cue_overlap(self, activations, decision_mask):
        """
        Measure how consistently the same decision cues are used.
        
        Args:
            activations: [T, dict_size]
            decision_mask: [dict_size] - binary mask of decision-relevant features
            
        Returns:
            cue_overlap_score: scalar - average Jaccard over time
        """
        decision_activations = activations[:, decision_mask]
        active_cues = (decision_activations > 0).float()
        
        overlaps = []
        for t in range(active_cues.shape[0] - 1):
            cue_t = active_cues[t]
            cue_t1 = active_cues[t + 1]
            
            intersection = (cue_t * cue_t1).sum()
            union = ((cue_t + cue_t1) > 0).float().sum()
            
            if union > 0:
                overlap = (intersection / union).item()
            else:
                overlap = 0.0
            
            overlaps.append(overlap)
        
        return np.mean(overlaps) if overlaps else 0.0
    
    def identify_feature_usage_pattern(self, activations, decision_mask):
        """
        Classify decision features by usage pattern.
        
        Returns:
            dict: {
                'persistent': features active >50% of time,
                'transient': features active <20% of time,
                'intermittent': features in between
            }
        """
        decision_activations = activations[:, decision_mask]
        active = (decision_activations > 0).float()
        
        activation_rates = active.mean(dim=0).numpy()
        
        decision_indices = np.where(decision_mask)[0]
        
        persistent = decision_indices[activation_rates > 0.5]
        transient = decision_indices[activation_rates < 0.2]
        intermittent = decision_indices[(activation_rates >= 0.2) & (activation_rates <= 0.5)]
        
        return {
            'persistent': persistent.tolist(),
            'transient': transient.tolist(),
            'intermittent': intermittent.tolist(),
            'activation_rates': activation_rates.tolist()
        }


def main():
    parser = argparse.ArgumentParser(description='Decision Relevance Analysis')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--output_dir', type=str, default='decision_analysis', help='Output directory')
    parser.add_argument('--database_path', type=str, default='/data/projects/punim2637/nnliang/Datasets/ASVspoof2021_LA_eval',
                       help='Path to ASVspoof dataset')
    parser.add_argument('--protocols_path', type=str, 
                       default='/data/projects/punim2637/nnliang/Datasets/keys/LA/CM/trial_metadata.txt',
                       help='Path to protocol files')
    parser.add_argument('--dataset', type=str, default='2021_LA', choices=['2019_LA_train', '2021_LA'],
                       help='Which dataset to use')
    parser.add_argument('--num_samples', type=int, default=200, help='Number of samples to analyze')
    parser.add_argument('--top_k_features', type=int, default=50, help='Number of top features to identify')
    parser.add_argument('--run_ablation', action='store_true', help='Run ablation validation (slow)')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for analysis')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading model from {args.model_path}")
    
    # Load model
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Initialize model with checkpoint config
    model_config = checkpoint.get('model_config', {})
    model = Model(
        args=argparse.Namespace(**model_config) if model_config else argparse.Namespace(),
        device=device,
        use_sae=True,
        use_sparse_features=checkpoint.get('use_sparse_features', True),
        sae_dict_size=checkpoint.get('sae_dict_size', 4096),
        sae_k=checkpoint.get('sae_k', 128),
        sae_window_size=checkpoint.get('sae_window_size', 8),
        sae_weight=checkpoint.get('sae_weight', 0.1)
    )
    
    # Handle DataParallel checkpoints (keys with "module." prefix)
    state_dict = checkpoint['model_state_dict']
    if list(state_dict.keys())[0].startswith('module.'):
        # Remove "module." prefix
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    window_size = checkpoint.get('sae_window_size', 8)
    dict_size = checkpoint.get('sae_dict_size', 4096)
    
    print(f"Model loaded: dict_size={dict_size}, k={checkpoint.get('sae_k', 128)}, window_size={window_size}")
    
    # Load dataset
    print("Loading dataset...")
    
    if args.dataset == '2021_LA':
        # Load 2021 LA eval dataset
        # Parse trial_metadata.txt to get labels
        file_list = []
        labels_dict = {}
        
        # First, get list of actual FLAC files
        flac_dir = os.path.join(args.database_path, 'flac')
        actual_files = set([f.replace('.flac', '') for f in os.listdir(flac_dir) if f.endswith('.flac')])
        print(f"Found {len(actual_files)} actual FLAC files")
        
        # Parse metadata but only include files that actually exist
        with open(args.protocols_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 6:
                    utt_id = parts[1]  # LA_E_xxxxxxx
                    label = parts[5]   # 'bonafide' or 'spoof'
                    if utt_id in actual_files:  # Only include if file exists
                        file_list.append(utt_id)
                        labels_dict[utt_id] = 1 if label == 'bonafide' else 0
        
        print(f"Loaded {len(file_list)} files from 2021 LA eval with labels and existing audio")
        
        # Subsample for analysis - stratified
        genuine_files = [f for f in file_list if labels_dict[f] == 1]
        spoof_files = [f for f in file_list if labels_dict[f] == 0]
        
        # Sample equal numbers from each class
        samples_per_class = min(args.num_samples // 2, len(genuine_files), len(spoof_files))
        np.random.seed(42)
        selected_genuine = list(np.random.choice(genuine_files, samples_per_class * 2, replace=False))  # Sample 2x more
        selected_spoof = list(np.random.choice(spoof_files, samples_per_class * 2, replace=False))
        
        # Pre-filter: test loading FULL file to identify corrupt ones
        print("Pre-filtering corrupt files (this may take a few minutes)...")
        valid_genuine = []
        valid_spoof = []
        
        import librosa
        for f in tqdm(selected_genuine, desc="Testing genuine files"):
            try:
                audio_path = os.path.join(args.database_path, 'flac', f + '.flac')
                X, fs = librosa.load(audio_path, sr=16000)  # Load full file
                if len(X) > 0:  # Ensure we got valid audio
                    valid_genuine.append(f)
                if len(valid_genuine) >= samples_per_class:
                    break
            except Exception as e:
                # Silently skip corrupt files
                continue
        
        for f in tqdm(selected_spoof, desc="Testing spoof files"):
            try:
                audio_path = os.path.join(args.database_path, 'flac', f + '.flac')
                X, fs = librosa.load(audio_path, sr=16000)  # Load full file
                if len(X) > 0:
                    valid_spoof.append(f)
                if len(valid_spoof) >= samples_per_class:
                    break
            except Exception as e:
                # Silently skip corrupt files
                continue
        
        balanced_files = valid_genuine + valid_spoof
        balanced_labels = [labels_dict[f] for f in balanced_files]
        print(f"Filtered to {len(balanced_files)} valid files ({len(valid_genuine)} genuine, {len(valid_spoof)} spoof)")
        
        # Create dataset using Dataset_ASVspoof2021_eval
        from data_utils_SSL import Dataset_ASVspoof2021_eval
        eval_set = Dataset_ASVspoof2021_eval(
            list_IDs=balanced_files,
            base_dir=args.database_path
        )
        
        # For compatibility with later code
        genuine_indices = list(range(samples_per_class))
        spoof_indices = list(range(samples_per_class, samples_per_class * 2))
        balanced_indices = list(range(len(balanced_files)))
        d_label_trn = labels_dict
        file_train = balanced_files
        train_set = eval_set
        
    else:  # 2019_LA_train
        d_label_trn, file_train = genSpoof_list(
            dir_meta=os.path.join(args.protocols_path, 'ASVspoof2019.LA.cm.train.trn.txt'),
            is_train=True
        )
        
        # Create args for dataset with RawBoost parameters
        dataset_args = argparse.Namespace(
            algo=3,
            # RawBoost parameters (using default values)
            SNRmin=0, SNRmax=40,
            nBands=5,
            minF=20, maxF=8000,
            minBW=100, maxBW=1000,
            minCoeff=10, maxCoeff=100,
            minG=0, maxG=0
        )
        
        train_set = Dataset_ASVspoof2019_train(
            args=dataset_args,
            list_IDs=file_train,
            labels=d_label_trn,
            base_dir=os.path.join(args.database_path, 'ASVspoof2019_LA_train/'),
            algo=3
        )
        
        # Subsample for analysis
        num_samples = min(args.num_samples, len(train_set))
        indices = np.random.choice(len(train_set), num_samples, replace=False)
        
        # Stratified sampling - get balanced genuine/spoof
        genuine_indices = [i for i in indices if d_label_trn[file_train[i]] == 1]
        spoof_indices = [i for i in indices if d_label_trn[file_train[i]] == 0]
    
    # Balance
    min_count = min(len(genuine_indices), len(spoof_indices))
    genuine_indices = genuine_indices[:min_count]
    spoof_indices = spoof_indices[:min_count]
    balanced_indices = genuine_indices + spoof_indices
    
    print(f"Analyzing {len(balanced_indices)} samples ({min_count} genuine, {min_count} spoof)")
    
    # Create custom dataset wrapper that returns labels for 2021 eval
    if args.dataset == '2021_LA':
        class LabeledDataset(Dataset):
            def __init__(self, base_dataset, indices, labels_dict, file_list):
                self.base_dataset = base_dataset
                self.indices = indices
                self.labels_dict = labels_dict
                self.file_list = file_list
            
            def __len__(self):
                return len(self.indices)
            
            def __getitem__(self, idx):
                audio, utt_id = self.base_dataset[self.indices[idx]]
                label = self.labels_dict[utt_id]
                return audio, label
        
        dataset_with_labels = LabeledDataset(train_set, balanced_indices, d_label_trn, file_train)
        dataloader = DataLoader(dataset_with_labels, batch_size=args.batch_size, shuffle=False, num_workers=0)
    else:
        subset = Subset(train_set, balanced_indices)
        dataloader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Initialize analyzers
    attribution_analyzer = FeatureAttributionAnalyzer(model, device)
    stability_analyzer = DecisionCueStabilityAnalyzer(window_size)
    cue_analyzer = CueConsistencyAnalyzer()
    
    # ==== PHASE 1: Feature Attribution Analysis ====
    print("\n" + "="*80)
    print("PHASE 1: Computing Feature Attribution Scores")
    print("="*80)
    
    all_attributions = []
    all_activations = []
    all_labels = []
    
    for batch_idx, (audio, labels) in enumerate(tqdm(dataloader, desc="Computing attributions")):
        attribution, activations = attribution_analyzer.compute_gradient_attribution(audio, labels)
        all_attributions.append(attribution)
        all_activations.append(activations)
        all_labels.extend(labels.numpy())
    
    # Aggregate attribution scores
    mean_attribution = torch.stack(all_attributions).mean(dim=0).numpy()
    
    # Identify top-K decision-relevant features
    top_k_indices = np.argsort(mean_attribution)[-args.top_k_features:][::-1]
    top_k_scores = mean_attribution[top_k_indices]
    
    print(f"\nTop-{args.top_k_features} decision-relevant features identified")
    print(f"  Attribution range: {top_k_scores[-1]:.6f} to {top_k_scores[0]:.6f}")
    print(f"  Mean attribution (all features): {mean_attribution.mean():.6f}")
    print(f"  Mean attribution (top-K): {top_k_scores.mean():.6f}")
    print(f"  Ratio: {top_k_scores.mean() / mean_attribution.mean():.2f}x")
    
    # Save attribution results
    attribution_results = {
        'attribution_scores': mean_attribution.tolist(),
        'top_k_indices': top_k_indices.tolist(),
        'top_k_scores': top_k_scores.tolist(),
        'dict_size': dict_size,
        'top_k': args.top_k_features
    }
    
    with open(output_dir / 'feature_attributions.json', 'w') as f:
        json.dump(attribution_results, f, indent=2)
    
    # Optional: Run ablation validation
    if args.run_ablation:
        print("\n" + "="*80)
        print("Running Ablation Validation (this may take a while...)")
        print("="*80)
        
        # Take first batch for ablation
        audio_sample, label_sample = next(iter(dataloader))
        ablation_scores = attribution_analyzer.compute_ablation_attribution(
            audio_sample, label_sample, top_k_indices[:10]  # Only top-10 for speed
        )
        
        print(f"\nAblation validation complete")
        print(f"  Top-10 features ablation impact: {ablation_scores.mean():.4f} ± {ablation_scores.std():.4f}")
        
        attribution_results['ablation_scores_top10'] = ablation_scores.tolist()
        with open(output_dir / 'feature_attributions.json', 'w') as f:
            json.dump(attribution_results, f, indent=2)
    
    print("\n✓ Phase 1 complete - feature_attributions.json saved")
    
    # ==== PHASE 2: Decision-Relevant Stability Analysis ====
    print("\n" + "="*80)
    print("PHASE 2: Decision-Relevant Temporal Stability Analysis")
    print("="*80)
    
    print(f"Analyzing temporal stability for {len(all_activations)} batches...")
    
    decision_mask = np.zeros(dict_size, dtype=bool)
    decision_mask[top_k_indices] = True
    
    all_features_stability = {
        'jaccard': [], 'lifetime': [], 'flipping': [],
        'boundary_effects': []
    }
    
    decision_features_stability = {
        'jaccard': [], 'lifetime': [], 'flipping': [],
        'boundary_effects': []
    }
    
    genuine_activations = []
    spoof_activations = []
    
    # Analyze each sample
    # Create label batches properly
    label_batches = [all_labels[i:i+args.batch_size] for i in range(0, len(all_labels), args.batch_size)]
    
    print(f"Processing {len(label_batches)} batches for stability analysis...")
    
    for activations_batch, labels in tqdm(zip(all_activations, label_batches), total=len(label_batches), desc="Stability analysis"):
        B = activations_batch.shape[0]
        
        for b in range(B):
            activations = activations_batch[b]  # [T, dict_size]
            label = labels[b]
            
            # Store for class-specific analysis
            if label == 1:  # Genuine
                genuine_activations.append(activations)
            else:  # Spoof
                spoof_activations.append(activations)
            
            # All features stability
            jaccard_all = stability_analyzer.compute_jaccard_similarity(activations, mask=None)
            lifetime_all = stability_analyzer.compute_feature_lifetime(activations, mask=None)
            flipping_all = stability_analyzer.compute_flipping_rate(activations, mask=None)
            boundary_all = stability_analyzer.analyze_boundary_effects(activations, mask=None)
            
            all_features_stability['jaccard'].extend(jaccard_all)
            all_features_stability['lifetime'].extend(lifetime_all)
            all_features_stability['flipping'].extend(flipping_all)
            all_features_stability['boundary_effects'].append(boundary_all)
            
            # Decision features stability
            jaccard_decision = stability_analyzer.compute_jaccard_similarity(activations, mask=decision_mask)
            lifetime_decision = stability_analyzer.compute_feature_lifetime(activations, mask=decision_mask)
            flipping_decision = stability_analyzer.compute_flipping_rate(activations, mask=decision_mask)
            boundary_decision = stability_analyzer.analyze_boundary_effects(activations, mask=decision_mask)
            
            decision_features_stability['jaccard'].extend(jaccard_decision)
            decision_features_stability['lifetime'].extend(lifetime_decision)
            decision_features_stability['flipping'].extend(flipping_decision)
            decision_features_stability['boundary_effects'].append(boundary_decision)
    
    # Aggregate boundary effects
    for key in ['all_features', 'decision_features']:
        stability_dict = all_features_stability if key == 'all_features' else decision_features_stability
        boundary_list = stability_dict['boundary_effects']
        
        stability_dict['boundary_summary'] = {
            'mean_boundary': np.mean([b['boundary'] for b in boundary_list]),
            'mean_interior': np.mean([b['interior'] for b in boundary_list]),
            'mean_cross_boundary': np.mean([b['cross_boundary'] for b in boundary_list])
        }
    
    print("\nTemporal Stability Comparison:")
    print(f"  All Features:")
    print(f"    Jaccard: {np.mean(all_features_stability['jaccard']):.3f}")
    print(f"    Lifetime: {np.mean(all_features_stability['lifetime']):.2f}")
    print(f"    Flipping: {np.mean(all_features_stability['flipping']):.2f}")
    print(f"    Interior Jaccard: {all_features_stability['boundary_summary']['mean_interior']:.3f}")
    print(f"    Cross-boundary Jaccard: {all_features_stability['boundary_summary']['mean_cross_boundary']:.3f}")
    
    print(f"\n  Decision-Relevant Features:")
    print(f"    Jaccard: {np.mean(decision_features_stability['jaccard']):.3f}")
    print(f"    Lifetime: {np.mean(decision_features_stability['lifetime']):.2f}")
    print(f"    Flipping: {np.mean(decision_features_stability['flipping']):.2f}")
    print(f"    Interior Jaccard: {decision_features_stability['boundary_summary']['mean_interior']:.3f}")
    print(f"    Cross-boundary Jaccard: {decision_features_stability['boundary_summary']['mean_cross_boundary']:.3f}")
    
    # Decision instability check
    decision_jaccard = np.mean(decision_features_stability['jaccard'])
    all_jaccard = np.mean(all_features_stability['jaccard'])
    
    if decision_jaccard < all_jaccard:
        print(f"\n⚠️  FINDING: Decision features are MORE UNSTABLE than average!")
        print(f"     This indicates decision brittleness - the model's reasoning is inconsistent.")
    else:
        print(f"\n✓  Decision features are relatively stable.")
    
    # Boundary effect check
    interior = decision_features_stability['boundary_summary']['mean_interior']
    cross_boundary = decision_features_stability['boundary_summary']['mean_cross_boundary']
    
    if cross_boundary < interior * 0.7:
        print(f"\n⚠️  FINDING: Window boundaries severely disrupt decision features!")
        print(f"     Interior: {interior:.3f}, Cross-boundary: {cross_boundary:.3f}")
        print(f"     This suggests hard windowing is problematic.")
    
    print("\n✓ Phase 2 complete - stability analysis done")
    
    # ==== PHASE 3: Cue Consistency Analysis ====
    print("\n" + "="*80)
    print("PHASE 3: Decision Cue Consistency Analysis")
    print("="*80)
    
    print(f"Analyzing cue consistency for {len(genuine_activations)} genuine and {len(spoof_activations)} spoof samples...")
    
    genuine_cue_overlaps = []
    spoof_cue_overlaps = []
    
    genuine_usage_patterns = []
    spoof_usage_patterns = []
    
    for activations in tqdm(genuine_activations, desc="Analyzing genuine samples"):
        overlap = cue_analyzer.compute_cue_overlap(activations, decision_mask)
        genuine_cue_overlaps.append(overlap)
        
        pattern = cue_analyzer.identify_feature_usage_pattern(activations, decision_mask)
        genuine_usage_patterns.append(pattern)
    
    for activations in tqdm(spoof_activations, desc="Analyzing spoof samples"):
        overlap = cue_analyzer.compute_cue_overlap(activations, decision_mask)
        spoof_cue_overlaps.append(overlap)
        
        pattern = cue_analyzer.identify_feature_usage_pattern(activations, decision_mask)
        spoof_usage_patterns.append(pattern)
    
    print(f"\nCue Consistency (Decision Cue Overlap):")
    print(f"  Genuine samples: {np.mean(genuine_cue_overlaps):.3f} ± {np.std(genuine_cue_overlaps):.3f}")
    print(f"  Spoof samples: {np.mean(spoof_cue_overlaps):.3f} ± {np.std(spoof_cue_overlaps):.3f}")
    
    if np.mean(spoof_cue_overlaps) < np.mean(genuine_cue_overlaps) * 0.9:
        print(f"\n⚠️  FINDING: Spoof detection shows lower cue consistency!")
        print(f"     The model switches between different decision cues more frequently for spoofs.")
    
    # Feature usage patterns
    persistent_genuine = np.mean([len(p['persistent']) for p in genuine_usage_patterns])
    transient_genuine = np.mean([len(p['transient']) for p in genuine_usage_patterns])
    
    persistent_spoof = np.mean([len(p['persistent']) for p in spoof_usage_patterns])
    transient_spoof = np.mean([len(p['transient']) for p in spoof_usage_patterns])
    
    print(f"\nFeature Usage Patterns (among decision-relevant features):")
    print(f"  Genuine:")
    print(f"    Persistent: {persistent_genuine:.1f}, Transient: {transient_genuine:.1f}")
    print(f"  Spoof:")
    print(f"    Persistent: {persistent_spoof:.1f}, Transient: {transient_spoof:.1f}")
    
    # ==== Save All Results ====
    print("\n" + "="*80)
    print("Saving Results")
    print("="*80)
    
    results = {
        'summary': {
            'num_samples_analyzed': len(all_labels),
            'num_genuine': len(genuine_activations),
            'num_spoof': len(spoof_activations),
            'dict_size': dict_size,
            'window_size': window_size,
            'top_k_features': args.top_k_features
        },
        'attribution': attribution_results,
        'stability': {
            'all_features': {
                'mean_jaccard': float(np.mean(all_features_stability['jaccard'])),
                'mean_lifetime': float(np.mean(all_features_stability['lifetime'])),
                'mean_flipping': float(np.mean(all_features_stability['flipping'])),
                'boundary_effects': all_features_stability['boundary_summary']
            },
            'decision_features': {
                'mean_jaccard': float(np.mean(decision_features_stability['jaccard'])),
                'mean_lifetime': float(np.mean(decision_features_stability['lifetime'])),
                'mean_flipping': float(np.mean(decision_features_stability['flipping'])),
                'boundary_effects': decision_features_stability['boundary_summary']
            }
        },
        'cue_consistency': {
            'genuine': {
                'mean_overlap': float(np.mean(genuine_cue_overlaps)),
                'std_overlap': float(np.std(genuine_cue_overlaps)),
                'mean_persistent_features': float(persistent_genuine),
                'mean_transient_features': float(transient_genuine)
            },
            'spoof': {
                'mean_overlap': float(np.mean(spoof_cue_overlaps)),
                'std_overlap': float(np.std(spoof_cue_overlaps)),
                'mean_persistent_features': float(persistent_spoof),
                'mean_transient_features': float(transient_spoof)
            }
        }
    }
    
    with open(output_dir / 'decision_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")
    print(f"  - feature_attributions.json")
    print(f"  - decision_analysis_results.json")
    
    # ==== Create Visualizations ====
    create_visualizations(
        results, attribution_results, 
        all_features_stability, decision_features_stability,
        genuine_cue_overlaps, spoof_cue_overlaps,
        output_dir
    )
    
    print(f"\nVisualizations saved to {output_dir}")
    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)


def create_visualizations(results, attribution_results, all_stability, decision_stability,
                         genuine_overlaps, spoof_overlaps, output_dir):
    """Create comprehensive visualizations."""
    
    sns.set_style("whitegrid")
    
    # 1. Attribution Distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    attributions = np.array(attribution_results['attribution_scores'])
    top_k_indices = attribution_results['top_k_indices']
    
    ax1.hist(attributions, bins=50, edgecolor='black', alpha=0.7)
    ax1.axvline(attributions[top_k_indices].min(), color='r', linestyle='--', 
                label=f'Top-{len(top_k_indices)} threshold')
    ax1.set_xlabel('Attribution Score')
    ax1.set_ylabel('Count')
    ax1.set_title('Feature Attribution Distribution')
    ax1.legend()
    ax1.set_yscale('log')
    
    # Top features
    top_k_scores = attributions[top_k_indices]
    ax2.bar(range(len(top_k_scores)), sorted(top_k_scores, reverse=True))
    ax2.set_xlabel('Feature Rank')
    ax2.set_ylabel('Attribution Score')
    ax2.set_title(f'Top-{len(top_k_indices)} Decision-Relevant Features')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'attribution_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Stability Comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Jaccard
    axes[0, 0].hist([all_stability['jaccard'], decision_stability['jaccard']], 
                    label=['All Features', 'Decision Features'], bins=30, alpha=0.6)
    axes[0, 0].set_xlabel('Jaccard Similarity')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Temporal Coherence Comparison')
    axes[0, 0].legend()
    axes[0, 0].axvline(np.mean(all_stability['jaccard']), color='C0', linestyle='--', alpha=0.8)
    axes[0, 0].axvline(np.mean(decision_stability['jaccard']), color='C1', linestyle='--', alpha=0.8)
    
    # Lifetime
    axes[0, 1].hist([all_stability['lifetime'], decision_stability['lifetime']], 
                    label=['All Features', 'Decision Features'], bins=30, alpha=0.6)
    axes[0, 1].set_xlabel('Feature Lifetime (frames)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Feature Persistence Comparison')
    axes[0, 1].legend()
    
    # Flipping rate
    axes[1, 0].hist([all_stability['flipping'], decision_stability['flipping']], 
                    label=['All Features', 'Decision Features'], bins=30, alpha=0.6)
    axes[1, 0].set_xlabel('Features Flipping per Timestep')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Feature Instability Comparison')
    axes[1, 0].legend()
    
    # Boundary effects
    boundary_data = {
        'Interior': [
            all_stability['boundary_summary']['mean_interior'],
            decision_stability['boundary_summary']['mean_interior']
        ],
        'Cross-boundary': [
            all_stability['boundary_summary']['mean_cross_boundary'],
            decision_stability['boundary_summary']['mean_cross_boundary']
        ]
    }
    x = np.arange(2)
    width = 0.35
    axes[1, 1].bar(x - width/2, boundary_data['Interior'], width, label='Interior', alpha=0.8)
    axes[1, 1].bar(x + width/2, boundary_data['Cross-boundary'], width, label='Cross-boundary', alpha=0.8)
    axes[1, 1].set_ylabel('Jaccard Similarity')
    axes[1, 1].set_title('Window Boundary Effects')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(['All Features', 'Decision Features'])
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'stability_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Cue Consistency
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Distribution
    axes[0].hist([genuine_overlaps, spoof_overlaps], 
                 label=['Genuine', 'Spoof'], bins=20, alpha=0.6)
    axes[0].set_xlabel('Decision Cue Overlap')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Cue Consistency: Genuine vs. Spoof')
    axes[0].legend()
    axes[0].axvline(np.mean(genuine_overlaps), color='C0', linestyle='--', alpha=0.8)
    axes[0].axvline(np.mean(spoof_overlaps), color='C1', linestyle='--', alpha=0.8)
    
    # Box plot
    axes[1].boxplot([genuine_overlaps, spoof_overlaps], 
                    labels=['Genuine', 'Spoof'])
    axes[1].set_ylabel('Decision Cue Overlap')
    axes[1].set_title('Cue Consistency Distribution')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cue_consistency.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("  ✓ Visualizations created")


if __name__ == '__main__':
    main()
