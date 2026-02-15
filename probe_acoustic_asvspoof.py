"""
Acoustic Feature Probe using ASVspoof Dataset
Analyzes which SAE features detect specific acoustic artifacts in spoofed audio
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
import librosa
from pathlib import Path

class AcousticProbe:
    """
    Probe to analyze SAE features for acoustic properties using ASVspoof data
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
        
        # Storage for feature-artifact associations
        self.attack_features = defaultdict(lambda: {'activations': [], 'samples': []})
        self.bonafide_features = {'activations': [], 'samples': []}
        
        # Acoustic property storage
        self.feature_correlations = {}
        
    def extract_acoustic_properties(self, audio_path: str, sr=16000) -> Dict:
        """
        Extract acoustic properties from audio file
        
        Returns:
            Dict with acoustic properties: pitch, energy, spectral centroid, etc.
        """
        y, _ = librosa.load(audio_path, sr=sr)
        
        # Fundamental frequency (pitch)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_mean = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
        
        # Energy (RMS)
        rms = librosa.feature.rms(y=y)
        energy_mean = np.mean(rms)
        energy_std = np.std(rms)
        
        # Spectral properties
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zero_crossing = librosa.feature.zero_crossing_rate(y)
        
        # MFCC statistics
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        return {
            'pitch_mean': float(pitch_mean),
            'energy_mean': float(energy_mean),
            'energy_std': float(energy_std),
            'spectral_centroid_mean': float(np.mean(spectral_centroid)),
            'spectral_bandwidth_mean': float(np.mean(spectral_bandwidth)),
            'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
            'zero_crossing_mean': float(np.mean(zero_crossing)),
            'mfcc_mean': mfcc.mean(axis=1).tolist(),
            'mfcc_std': mfcc.std(axis=1).tolist(),
        }
    
    def extract_features(self, audio_path: str) -> Tuple[np.ndarray, Dict]:
        """
        Extract SAE features and acoustic properties from audio
        
        Returns:
            (sparse_features, acoustic_properties)
        """
        # Load audio
        y, sr = librosa.load(audio_path, sr=16000)
        
        # Pad/trim to expected length (same as training data)
        target_len = 64600
        x_len = y.shape[0]
        if x_len >= target_len:
            y = y[:target_len]
        else:
            # Tile/repeat audio to reach target length (same as training)
            num_repeats = int(target_len / x_len) + 1
            y = np.tile(y, num_repeats)[:target_len]
        
        # Create tensor [batch_size, time]
        audio_tensor = torch.FloatTensor(y).unsqueeze(0).to(self.device)  # [1, 64600]
        
        # Get sparse features
        with torch.no_grad():
            result = self.model(audio_tensor,  # Already has batch dimension
                               return_sae_loss=False, 
                               return_interpretability=True)
            # When return_sae_loss=False, return_interpretability=True: returns (output, interp_dict)
            if len(result) == 2:
                _, interp_dict = result
            elif len(result) == 3:
                _, _, interp_dict = result
            else:
                raise ValueError(f"Unexpected model output: {len(result)} values")
            
            # Average features across time
            sparse_features = interp_dict['sparse_features'][0]  # [T, D]
            avg_features = sparse_features.mean(dim=0).cpu().numpy()  # [D]
        
        # Extract acoustic properties
        acoustic_props = self.extract_acoustic_properties(audio_path)
        
        return avg_features, acoustic_props
    
    def analyze_asvspoof_dataset(self, protocol_file: str, audio_dir: str, max_samples=500):
        """
        Analyze ASVspoof dataset to find feature-attack associations
        
        Args:
            protocol_file: Path to protocol file (e.g., ASVspoof2019.LA.cm.dev.trl.txt)
            audio_dir: Directory containing audio files
            max_samples: Maximum samples per attack type
        """
        print(f"Analyzing ASVspoof dataset from {protocol_file}...")
        
        # Parse protocol file
        attack_samples = defaultdict(list)
        
        with open(protocol_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    speaker, utt_id, _, attack_type, label = parts[:5]
                    audio_path = Path(audio_dir) / 'flac' / f'{utt_id}.flac'
                    
                    if audio_path.exists():
                        if label == 'bonafide':
                            if len(self.bonafide_features['samples']) < max_samples:
                                attack_samples['bonafide'].append(str(audio_path))
                        else:  # spoof
                            if len(attack_samples[attack_type]) < max_samples // 10:  # Fewer per attack
                                attack_samples[attack_type].append(str(audio_path))
        
        print(f"Found attacks: {list(attack_samples.keys())}")
        
        # Extract features for each attack type
        for attack_type, audio_paths in attack_samples.items():
            print(f"\nProcessing {attack_type}: {len(audio_paths)} samples")
            
            for audio_path in tqdm(audio_paths, desc=f"  {attack_type}"):
                try:
                    features, acoustic_props = self.extract_features(audio_path)
                    
                    if attack_type == 'bonafide':
                        self.bonafide_features['activations'].append(features)
                        self.bonafide_features['samples'].append(acoustic_props)
                    else:
                        self.attack_features[attack_type]['activations'].append(features)
                        self.attack_features[attack_type]['samples'].append(acoustic_props)
                except Exception as e:
                    print(f"Error processing {audio_path}: {e}")
                    continue
        
        print("\n" + "="*70)
        print("FEATURE EXTRACTION COMPLETE")
        print(f"Bonafide samples: {len(self.bonafide_features['activations'])}")
        print(f"Attack samples: {sum(len(v['activations']) for v in self.attack_features.values())}")
        print("="*70)
        
        # Compute statistics
        try:
            print("\nCalling compute_statistics()...")
            self.compute_statistics()
            print("compute_statistics() completed successfully")
        except Exception as e:
            print(f"ERROR in compute_statistics(): {e}")
            import traceback
            traceback.print_exc()
    
    def compute_statistics(self):
        """Compute statistics for each attack type"""
        print("\nComputing statistics...")
        
        # Bonafide statistics
        if self.bonafide_features['activations']:
            self.bonafide_features['mean'] = np.mean(self.bonafide_features['activations'], axis=0)
            self.bonafide_features['std'] = np.std(self.bonafide_features['activations'], axis=0)
            print(f"  Bonafide: {len(self.bonafide_features['activations'])} samples processed")
        else:
            print("  Warning: No bonafide samples found!")
        
        # Attack statistics
        for attack_type in self.attack_features:
            if self.attack_features[attack_type]['activations']:
                activations = np.array(self.attack_features[attack_type]['activations'])
                self.attack_features[attack_type]['mean'] = activations.mean(axis=0)
                self.attack_features[attack_type]['std'] = activations.std(axis=0)
                print(f"  {attack_type}: {len(self.attack_features[attack_type]['activations'])} samples processed")
            else:
                print(f"  Warning: No samples found for {attack_type}")
    
    def find_discriminative_features(self, top_k=20, threshold=2.0) -> Dict:
        """
        Find features that are discriminative for specific attack types
        
        Returns:
            Dict mapping attack types to their discriminative features
        """
        discriminative_features = {}
        
        if 'mean' not in self.bonafide_features:
            print("Warning: No bonafide statistics available")
            return discriminative_features
        
        bonafide_mean = self.bonafide_features['mean']
        bonafide_std = self.bonafide_features['std']
        
        for attack_type in self.attack_features:
            attack_mean = self.attack_features[attack_type]['mean']
            
            # Compute difference from bonafide
            diff = attack_mean - bonafide_mean
            abs_diff = np.abs(diff)
            
            # Find top-k most different features
            top_indices = np.argsort(abs_diff)[-top_k:][::-1]
            
            # Check if difference is significant
            significant_features = []
            for idx in top_indices:
                if bonafide_std[idx] > 0:
                    z_score = abs_diff[idx] / bonafide_std[idx]
                    if z_score > threshold:
                        significant_features.append({
                            'feature_idx': int(idx),
                            'activation_diff': float(diff[idx]),
                            'z_score': float(z_score),
                            'bonafide_mean': float(bonafide_mean[idx]),
                            'attack_mean': float(attack_mean[idx])
                        })
            
            discriminative_features[attack_type] = significant_features
        
        return discriminative_features
    
    def correlate_features_with_acoustics(self, attack_type='bonafide') -> Dict:
        """
        Correlate SAE features with acoustic properties
        
        Returns:
            Dict mapping acoustic properties to correlated features
        """
        if attack_type == 'bonafide':
            data = self.bonafide_features
        else:
            data = self.attack_features[attack_type]
        
        if not data['activations'] or not data['samples']:
            return {}
        
        activations = np.array(data['activations'])  # [N, D]
        
        # Extract acoustic property arrays
        acoustic_arrays = {}
        for prop in ['pitch_mean', 'energy_mean', 'spectral_centroid_mean']:
            acoustic_arrays[prop] = np.array([s[prop] for s in data['samples']])
        
        # Compute correlations
        correlations = {}
        for prop, values in acoustic_arrays.items():
            # Pearson correlation with each feature
            corr = np.array([np.corrcoef(activations[:, i], values)[0, 1] 
                           for i in range(activations.shape[1])])
            
            # Find top correlated features
            top_positive = np.argsort(corr)[-10:][::-1]
            top_negative = np.argsort(corr)[:10]
            
            correlations[prop] = {
                'top_positive': [(int(i), float(corr[i])) for i in top_positive],
                'top_negative': [(int(i), float(corr[i])) for i in top_negative]
            }
        
        return correlations
    
    def plot_attack_comparison(self, output_path='attack_comparison.png'):
        """Plot comparison of feature activations across attack types"""
        attack_types = list(self.attack_features.keys())
        if not attack_types:
            print("No attack types to plot")
            return
        
        # Get top 50 most discriminative features
        all_diffs = []
        bonafide_mean = self.bonafide_features['mean']
        
        for attack_type in attack_types:
            attack_mean = self.attack_features[attack_type]['mean']
            diff = np.abs(attack_mean - bonafide_mean)
            all_diffs.append(diff)
        
        all_diffs = np.array(all_diffs)
        max_diff_per_feature = all_diffs.max(axis=0)
        top_features = np.argsort(max_diff_per_feature)[-50:][::-1]
        
        # Create matrix for heatmap
        matrix = [bonafide_mean[top_features]]
        labels = ['bonafide']
        
        for attack_type in sorted(attack_types):
            matrix.append(self.attack_features[attack_type]['mean'][top_features])
            labels.append(attack_type)
        
        matrix = np.array(matrix)
        
        # Plot
        plt.figure(figsize=(20, 8))
        sns.heatmap(matrix, 
                   xticklabels=top_features,
                   yticklabels=labels,
                   cmap='RdYlBu_r',
                   center=0,
                   cbar_kws={'label': 'Mean Activation'})
        plt.xlabel('Feature Index')
        plt.ylabel('Audio Type')
        plt.title('SAE Feature Activations: Bonafide vs Spoofing Attacks (Top 50 Discriminative Features)')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {output_path}")
    
    def save_results(self, output_path='acoustic_probe_results.json'):
        """Save probe results"""
        # Ensure statistics are computed
        if 'mean' not in self.bonafide_features:
            print("Warning: Statistics not computed, computing now...")
            self.compute_statistics()
        
        discriminative = self.find_discriminative_features()
        correlations = self.correlate_features_with_acoustics('bonafide')
        
        results = {
            'discriminative_features': discriminative,
            'acoustic_correlations': correlations,
            'attack_types': list(self.attack_features.keys()),
            'num_bonafide_samples': len(self.bonafide_features['activations'])
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_path}")
    
    def print_summary(self):
        """Print summary of acoustic feature analysis"""
        print("\n" + "="*70)
        print("ACOUSTIC FEATURE ANALYSIS SUMMARY")
        print("="*70)
        
        # Discriminative features
        discriminative = self.find_discriminative_features(top_k=10)
        
        for attack_type, features in discriminative.items():
            print(f"\n{attack_type.upper()}")
            print("-" * 50)
            print(f"Top discriminative features:")
            for feat in features[:5]:
                print(f"  Feature {feat['feature_idx']:4d}: "
                      f"diff={feat['activation_diff']:+.4f}, "
                      f"z={feat['z_score']:.2f}")
        
        # Acoustic correlations
        print("\n" + "="*70)
        print("ACOUSTIC PROPERTY CORRELATIONS (Bonafide)")
        print("="*70)
        
        correlations = self.correlate_features_with_acoustics('bonafide')
        for prop, corr_data in correlations.items():
            print(f"\n{prop}:")
            print("  Positive correlations:", 
                  [(f"F{i}", f"{c:.3f}") for i, c in corr_data['top_positive'][:3]])
            print("  Negative correlations:", 
                  [(f"F{i}", f"{c:.3f}") for i, c in corr_data['top_negative'][:3]])


def run_asvspoof_probe(model_path: str, 
                       protocol_file: str,
                       audio_dir: str,
                       output_dir: str = 'analysis_results'):
    """
    Run acoustic probe on ASVspoof dataset
    
    Args:
        model_path: Path to trained model checkpoint
        protocol_file: Path to ASVspoof protocol file
        audio_dir: Directory with audio files
        output_dir: Output directory for results
    """
    import os
    from model_window_topk import Model as ModelWindowTopK
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize model
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
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle DataParallel wrapper
    if 'module.' in list(checkpoint['model_state_dict'].keys())[0]:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint['model_state_dict'].items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model = model.to(device)
    model.eval()
    
    # Initialize probe
    probe = AcousticProbe(model, device)
    
    # Analyze dataset
    probe.analyze_asvspoof_dataset(protocol_file, audio_dir, max_samples=500)
    
    # Generate results
    probe.plot_attack_comparison(f'{output_dir}/attack_comparison.png')
    probe.save_results(f'{output_dir}/acoustic_probe_results.json')
    probe.print_summary()
    
    return probe


if __name__ == '__main__':
    # Example usage
    model_path = 'models/topk_sae_window_w8_LA_e40_bs14_lr1e-06_saeW0.1_dict4096_k128_window_topk_w8/best_checkpoint_eer_window_topk_w8.pth'
    protocol_file = '/data/projects/punim2637/nnliang/Datasets/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'
    audio_dir = '/data/projects/punim2637/nnliang/Datasets/LA/ASVspoof2019_LA_dev'
    
    probe = run_asvspoof_probe(model_path, protocol_file, audio_dir)
