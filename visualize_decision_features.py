#!/usr/bin/env python3
"""
Visualize top decision features and their activation patterns
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm
import librosa
import librosa.display

# Import your model
from model_window_topk import Model as WindowTopKModel
from data_utils_SSL import genSpoof_list

def load_decision_features(result_path):
    """Load top decision features from analysis results"""
    with open(result_path) as f:
        data = json.load(f)
    
    top_features = data['attribution']['top_k_indices'][:10]  # Top 10
    top_scores = data['attribution']['top_k_scores'][:10]
    
    print(f"Top 10 Decision Features:")
    for i, (feat_id, score) in enumerate(zip(top_features, top_scores), 1):
        print(f"  {i}. Feature {feat_id}: attribution={score:.6f}")
    
    return top_features, top_scores

def extract_activations_and_audio(model, audio_path, device='cuda'):
    """
    Extract SAE activations and corresponding audio
    
    Returns:
        activations: [T, dict_size]
        audio: waveform
        sr: sample rate
        spectrogram: mel spectrogram for visualization
    """
    model.eval()
    
    # Load audio
    audio, sr = torchaudio.load(audio_path)
    audio = audio.to(device)
    
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    
    # Get SSL features and SAE activations
    with torch.no_grad():
        ssl_feats = model.ssl_model(audio)
        if isinstance(ssl_feats, tuple):
            ssl_feats = ssl_feats[0]
        
        # Get SAE activations
        activations = model.sae.get_activations(ssl_feats)  # [T, dict_size]
    
    # Compute mel spectrogram for visualization
    audio_np = audio.squeeze().cpu().numpy()
    mel_spec = librosa.feature.melspectrogram(
        y=audio_np, sr=sr, n_mels=128, fmax=8000
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    return activations, audio_np, sr, mel_spec_db

def visualize_feature_activation_pattern(
    activations, mel_spec, feature_ids, audio_path, save_dir
):
    """
    Visualize when and how decision features activate over time
    
    Creates a multi-panel figure:
    - Top: Mel spectrogram
    - Middle: Activation heatmap for top-10 features
    - Bottom: Individual feature activation traces
    """
    activations_np = activations.cpu().numpy()  # [T, dict_size]
    decision_activations = activations_np[:, feature_ids]  # [T, 10]
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), 
                             gridspec_kw={'height_ratios': [2, 1.5, 2]})
    
    # Panel 1: Mel Spectrogram
    ax = axes[0]
    img = librosa.display.specshow(
        mel_spec, x_axis='time', y_axis='mel', 
        fmax=8000, ax=ax, cmap='viridis'
    )
    ax.set_title('Mel Spectrogram', fontsize=14, fontweight='bold')
    ax.set_ylabel('Frequency (Hz)')
    plt.colorbar(img, ax=ax, format='%+2.0f dB')
    
    # Panel 2: Activation Heatmap
    ax = axes[1]
    # Normalize each feature to [0, 1] for better visualization
    decision_activations_norm = decision_activations / (decision_activations.max(axis=0, keepdims=True) + 1e-8)
    
    img = ax.imshow(
        decision_activations_norm.T, 
        aspect='auto', 
        interpolation='nearest',
        cmap='hot',
        origin='lower'
    )
    ax.set_yticks(range(10))
    ax.set_yticklabels([f"F{fid}" for fid in feature_ids])
    ax.set_xlabel('Time (frames)')
    ax.set_ylabel('Feature ID')
    ax.set_title('Top-10 Decision Feature Activations (Normalized)', 
                 fontsize=14, fontweight='bold')
    plt.colorbar(img, ax=ax, label='Normalized Activation')
    
    # Panel 3: Individual traces
    ax = axes[2]
    time_frames = np.arange(decision_activations.shape[0])
    
    for i, feat_id in enumerate(feature_ids):
        activation = decision_activations[:, i]
        ax.plot(time_frames, activation, label=f'F{feat_id}', alpha=0.7, linewidth=1.5)
    
    ax.set_xlabel('Time (frames)', fontsize=12)
    ax.set_ylabel('Activation Strength', fontsize=12)
    ax.set_title('Individual Feature Activation Traces', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', ncol=2, fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    save_path = save_dir / f"{Path(audio_path).stem}_activation_pattern.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualization: {save_path}")

def analyze_activation_statistics(activations, feature_ids, label):
    """
    Analyze why these features are more stable
    """
    activations_np = activations.cpu().numpy()
    decision_activations = activations_np[:, feature_ids]
    
    stats = {
        'mean_activation': decision_activations.mean(axis=0),
        'std_activation': decision_activations.std(axis=0),
        'sparsity': (decision_activations > 0).mean(axis=0),
        'max_activation': decision_activations.max(axis=0),
        'activation_range': decision_activations.max(axis=0) - decision_activations.min(axis=0),
    }
    
    print(f"\n{'='*60}")
    print(f"Activation Statistics - {label}")
    print(f"{'='*60}")
    
    for i, feat_id in enumerate(feature_ids):
        print(f"\nFeature {feat_id}:")
        print(f"  Mean: {stats['mean_activation'][i]:.4f}")
        print(f"  Std:  {stats['std_activation'][i]:.4f}")
        print(f"  Sparsity (% active): {stats['sparsity'][i]*100:.1f}%")
        print(f"  Max value: {stats['max_activation'][i]:.4f}")
        print(f"  Range: {stats['activation_range'][i]:.4f}")
    
    # Check temporal consistency
    active = (decision_activations > 0).astype(float)
    jaccard_scores = []
    for t in range(active.shape[0] - 1):
        intersection = (active[t] * active[t+1]).sum()
        union = ((active[t] + active[t+1]) > 0).sum()
        if union > 0:
            jaccard_scores.append(intersection / union)
    
    print(f"\nTemporal Consistency:")
    print(f"  Mean Jaccard (adjacent frames): {np.mean(jaccard_scores):.4f}")
    print(f"  These features are {'highly stable' if np.mean(jaccard_scores) > 0.9 else 'moderately stable'}")
    
    return stats

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Paths
    result_path = Path("decision_analysis_2021LA_5k/decision_analysis_results.json")
    model_path = Path("models/topk_sae_window_w8_LA_e40_bs14_lr1e-06_saeW0.1_dict4096_k128_window_topk_w8/best_checkpoint_eer_window_topk_w8.pth")
    
    # Load decision features
    top_features, top_scores = load_decision_features(result_path)
    
    # Load model
    print("\nLoading model...")
    
    class Args:
        pass
    args = Args()
    args.architecture = 'XLSR'
    args.num_classes = 2
    
    model = WindowTopKModel(
        args=args,
        device=device,
        cp_path='xlsr2_300m.pt',
        use_sae=True,
        use_sparse_features=True,
        sae_dict_size=4096,
        sae_k=128,
        sae_window_size=8,
        sae_weight=0.1
    )
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Get sample files
    database_path = Path("/data/projects/punim2637/nnliang/Datasets/ASVspoof2021_LA_eval")
    
    # Load genuine and spoof samples
    _, _, eval_loader = genSpoof_list(
        dir_meta=database_path / 'keys/CM',
        is_train=False,
        is_eval=True
    )
    
    # Select a few samples for visualization
    samples_to_viz = [
        ('bonafide', 5),  # 5 genuine samples
        ('spoof', 5),     # 5 spoof samples
    ]
    
    save_dir = Path("decision_feature_visualizations")
    save_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*60)
    print("Visualizing Decision Feature Activation Patterns")
    print("="*60)
    
    for label, count in samples_to_viz:
        print(f"\n{'='*60}")
        print(f"Processing {label.upper()} samples")
        print(f"{'='*60}")
        
        processed = 0
        for utt_id, audio_path, target in eval_loader:
            if label == 'bonafide' and target[0] == 1:
                continue  # Skip spoof
            if label == 'spoof' and target[0] == 0:
                continue  # Skip bonafide
            
            print(f"\nProcessing: {Path(audio_path[0]).name}")
            
            # Extract activations
            activations, audio, sr, mel_spec = extract_activations_and_audio(
                model, audio_path[0], device
            )
            
            # Visualize
            visualize_feature_activation_pattern(
                activations, mel_spec, top_features, 
                audio_path[0], save_dir
            )
            
            # Analyze statistics
            analyze_activation_statistics(activations, top_features, 
                                         f"{label} - {Path(audio_path[0]).stem}")
            
            processed += 1
            if processed >= count:
                break
    
    print(f"\n{'='*60}")
    print("All visualizations saved to:", save_dir)
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
