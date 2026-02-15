"""
Phoneme Probe for SAE Features
Analyzes which SAE features activate for specific phonemes
"""

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple
import json
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

class PhonemeProbe:
    """
    Probe to analyze SAE feature activations for different phonemes
    """
    
    def __init__(self, model, device='cuda'):
        """
        Args:
            model: Your trained model with SAE
            device: Device to run on
        """
        self.model = model
        self.device = device
        self.model.eval()
        
        # Storage for phoneme-feature associations
        self.phoneme_features = defaultdict(lambda: defaultdict(list))
        
    def extract_features_with_timing(self, audio: torch.Tensor, phoneme_timings: List[Tuple[float, float, str]]):
        """
        Extract SAE features aligned with phoneme timings
        
        Args:
            audio: Audio tensor [1, samples]
            phoneme_timings: List of (start_time, end_time, phoneme_label)
        
        Returns:
            Dict mapping phonemes to their feature activations
        """
        with torch.no_grad():
            audio = audio.to(self.device)
            
            # Get sparse features from SAE
            _, _, interp_dict = self.model(audio.unsqueeze(0), 
                                          return_sae_loss=False, 
                                          return_interpretability=True)
            
            # Sparse features: [1, T, D]
            sparse_features = interp_dict['sparse_features'][0]  # [T, D]
            
            # XLS-R has ~50Hz frame rate (20ms per frame)
            frame_rate = 50  # frames per second
            
            phoneme_feature_dict = {}
            
            for start_time, end_time, phoneme in phoneme_timings:
                # Convert time to frame indices
                start_frame = int(start_time * frame_rate)
                end_frame = int(end_time * frame_rate)
                
                # Clip to valid range
                start_frame = max(0, start_frame)
                end_frame = min(sparse_features.shape[0], end_frame)
                
                if start_frame < end_frame:
                    # Get features for this phoneme segment
                    phoneme_feats = sparse_features[start_frame:end_frame]  # [T', D]
                    
                    # Average activations across time for this phoneme
                    avg_activation = phoneme_feats.mean(dim=0)  # [D]
                    
                    if phoneme not in phoneme_feature_dict:
                        phoneme_feature_dict[phoneme] = []
                    phoneme_feature_dict[phoneme].append(avg_activation.cpu().numpy())
            
            return phoneme_feature_dict
    
    def analyze_dataset(self, audio_phoneme_pairs: List[Tuple[torch.Tensor, List]]):
        """
        Analyze entire dataset of audio-phoneme pairs
        
        Args:
            audio_phoneme_pairs: List of (audio_tensor, phoneme_timings)
        """
        print("Analyzing phoneme-feature associations...")
        
        for audio, phoneme_timings in tqdm(audio_phoneme_pairs):
            phoneme_feats = self.extract_features_with_timing(audio, phoneme_timings)
            
            # Accumulate features per phoneme
            for phoneme, feat_list in phoneme_feats.items():
                self.phoneme_features[phoneme]['activations'].extend(feat_list)
        
        # Compute statistics
        self.compute_phoneme_statistics()
    
    def compute_phoneme_statistics(self):
        """Compute mean and std for each phoneme's feature activations"""
        print("Computing phoneme statistics...")
        
        for phoneme in self.phoneme_features:
            activations = np.array(self.phoneme_features[phoneme]['activations'])
            
            # Mean activation per feature
            self.phoneme_features[phoneme]['mean'] = activations.mean(axis=0)
            self.phoneme_features[phoneme]['std'] = activations.std(axis=0)
            
            # Top-k most active features for this phoneme
            top_k = 20
            top_indices = np.argsort(self.phoneme_features[phoneme]['mean'])[-top_k:][::-1]
            self.phoneme_features[phoneme]['top_features'] = top_indices.tolist()
    
    def find_phoneme_specific_features(self, threshold=2.0):
        """
        Find features that are specifically active for certain phonemes
        
        Args:
            threshold: Activation must be this many std devs above mean for other phonemes
        
        Returns:
            Dict mapping features to their associated phonemes
        """
        feature_phoneme_map = defaultdict(list)
        
        all_phonemes = list(self.phoneme_features.keys())
        n_features = len(self.phoneme_features[all_phonemes[0]]['mean'])
        
        for feat_idx in range(n_features):
            # Get activations for this feature across all phonemes
            activations = {p: self.phoneme_features[p]['mean'][feat_idx] 
                          for p in all_phonemes}
            
            # Find phonemes with high activation
            max_phoneme = max(activations, key=activations.get)
            max_activation = activations[max_phoneme]
            
            # Compare to other phonemes
            other_activations = [v for k, v in activations.items() if k != max_phoneme]
            mean_other = np.mean(other_activations)
            std_other = np.std(other_activations)
            
            if std_other > 0 and max_activation > mean_other + threshold * std_other:
                feature_phoneme_map[feat_idx].append((max_phoneme, max_activation))
        
        return feature_phoneme_map
    
    def plot_phoneme_feature_heatmap(self, output_path='phoneme_features.png'):
        """Plot heatmap of phoneme vs feature activations"""
        phonemes = sorted(self.phoneme_features.keys())
        
        # Create matrix: phonemes x features (top 100 features)
        feature_matrix = []
        for phoneme in phonemes:
            feature_matrix.append(self.phoneme_features[phoneme]['mean'])
        
        feature_matrix = np.array(feature_matrix)  # [n_phonemes, n_features]
        
        # Plot top 50 most variable features
        feature_vars = feature_matrix.var(axis=0)
        top_features = np.argsort(feature_vars)[-50:][::-1]
        
        plt.figure(figsize=(15, 10))
        sns.heatmap(feature_matrix[:, top_features], 
                   xticklabels=top_features, 
                   yticklabels=phonemes,
                   cmap='YlOrRd',
                   cbar_kws={'label': 'Mean Activation'})
        plt.xlabel('Feature Index')
        plt.ylabel('Phoneme')
        plt.title('Phoneme-Feature Activation Heatmap (Top 50 Variable Features)')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to {output_path}")
    
    def save_results(self, output_path='phoneme_probe_results.json'):
        """Save probe results to JSON"""
        results = {}
        for phoneme, data in self.phoneme_features.items():
            results[phoneme] = {
                'top_features': data['top_features'],
                'mean_activation': data['mean'].tolist(),
                'std_activation': data['std'].tolist()
            }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_path}")
    
    def print_summary(self):
        """Print summary of phoneme-feature associations"""
        print("\n" + "="*70)
        print("PHONEME-FEATURE ANALYSIS SUMMARY")
        print("="*70)
        
        for phoneme in sorted(self.phoneme_features.keys()):
            top_features = self.phoneme_features[phoneme]['top_features'][:5]
            print(f"\nPhoneme: {phoneme}")
            print(f"  Top 5 features: {top_features}")
            
            # Show activation values
            mean_acts = self.phoneme_features[phoneme]['mean'][top_features]
            print(f"  Activation values: {mean_acts}")


def load_timit_phoneme_data(timit_path: str) -> List[Tuple[torch.Tensor, List]]:
    """
    Load TIMIT dataset with phoneme alignments
    
    Args:
        timit_path: Path to TIMIT dataset
    
    Returns:
        List of (audio_tensor, phoneme_timings)
    """
    import torchaudio
    from pathlib import Path
    
    data_pairs = []
    
    # TIMIT has .wav files and .phn files with phoneme alignments
    wav_files = list(Path(timit_path).rglob("*.wav"))
    
    for wav_file in wav_files[:100]:  # Process first 100 files
        phn_file = wav_file.with_suffix('.phn')
        
        if not phn_file.exists():
            continue
        
        # Load audio
        audio, sr = torchaudio.load(wav_file)
        if sr != 16000:
            audio = torchaudio.transforms.Resample(sr, 16000)(audio)
        
        # Load phoneme timings
        phoneme_timings = []
        with open(phn_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    start_sample, end_sample, phoneme = parts
                    start_time = int(start_sample) / 16000
                    end_time = int(end_sample) / 16000
                    phoneme_timings.append((start_time, end_time, phoneme))
        
        data_pairs.append((audio[0], phoneme_timings))
    
    return data_pairs


def example_usage():
    """Example of how to use the phoneme probe"""
    from model_window_topk import Model as ModelWindowTopK
    
    # Load your trained model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize model (adjust parameters as needed)
    class Args:
        pass
    args = Args()
    
    model = ModelWindowTopK(
        args=args,
        device=device,
        use_sae=True,
        sae_dict_size=4096,
        sae_k=128,
        sae_window_size=8
    )
    
    # Load checkpoint
    checkpoint_path = 'models/topk_sae_window_w8_LA_e40_bs14_lr1e-06_saeW0.1_dict4096_k128_window_topk_w8/best_checkpoint_eer_window_topk_w8.pth'
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Initialize probe
    probe = PhonemeProbe(model, device)
    
    # Load phoneme data (you need to download TIMIT first)
    timit_path = "/path/to/TIMIT"
    audio_phoneme_pairs = load_timit_phoneme_data(timit_path)
    
    # Analyze
    probe.analyze_dataset(audio_phoneme_pairs)
    
    # Get results
    feature_phoneme_map = probe.find_phoneme_specific_features(threshold=1.5)
    
    # Visualize and save
    probe.plot_phoneme_feature_heatmap('analysis_results/phoneme_heatmap.png')
    probe.save_results('analysis_results/phoneme_probe.json')
    probe.print_summary()
    
    return probe


if __name__ == '__main__':
    probe = example_usage()
