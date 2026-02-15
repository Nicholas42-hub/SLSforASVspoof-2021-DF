#!/usr/bin/env python3
"""
Compute EER and min t-DCF for ASVspoof 2021 LA evaluation
"""
import numpy as np
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

def compute_eer(bonafide_scores, spoof_scores):
    """
    Compute Equal Error Rate (EER)
    bonafide_scores: scores for bonafide (genuine) samples
    spoof_scores: scores for spoof (fake) samples
    """
    # Combine scores and labels
    # Label: 1 for bonafide, 0 for spoof
    scores = np.concatenate([bonafide_scores, spoof_scores])
    labels = np.concatenate([np.ones(len(bonafide_scores)), np.zeros(len(spoof_scores))])
    
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    
    # Compute EER
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    
    # Find threshold at EER
    thresh = interp1d(fpr, thresholds)(eer)
    
    return eer * 100, thresh

def compute_min_tdcf(bonafide_scores, spoof_scores, p_target=0.05, c_miss=1, c_fa=1):
    """
    Compute minimum tandem Detection Cost Function (t-DCF)
    """
    scores = np.concatenate([bonafide_scores, spoof_scores])
    labels = np.concatenate([np.ones(len(bonafide_scores)), np.zeros(len(spoof_scores))])
    
    # Sort by score
    sorted_indices = np.argsort(scores)
    scores_sorted = scores[sorted_indices]
    labels_sorted = labels[sorted_indices]
    
    # Compute TPR and FPR at all thresholds
    n_bonafide = len(bonafide_scores)
    n_spoof = len(spoof_scores)
    
    # Initialize
    min_dcf = float('inf')
    
    for i in range(len(scores_sorted)):
        threshold = scores_sorted[i]
        
        # Predictions: score >= threshold → bonafide, score < threshold → spoof
        fn = np.sum((labels_sorted >= threshold) & (labels_sorted == 1) == False)  # Miss
        fp = np.sum((labels_sorted >= threshold) & (labels_sorted == 0))  # False alarm
        
        # Rates
        p_miss = fn / n_bonafide if n_bonafide > 0 else 0
        p_fa = fp / n_spoof if n_spoof > 0 else 0
        
        # DCF
        dcf = c_miss * p_miss * p_target + c_fa * p_fa * (1 - p_target)
        
        if dcf < min_dcf:
            min_dcf = dcf
    
    return min_dcf

def load_scores(score_file):
    """Load scores from file"""
    scores_dict = {}
    with open(score_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                file_id, score = parts
                scores_dict[file_id] = float(score)
    return scores_dict

def load_labels(metadata_file):
    """Load labels from trial metadata"""
    labels_dict = {}
    with open(metadata_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 8 and parts[7] == 'eval':
                file_id = parts[1]
                label = parts[5]  # 'bonafide' or 'spoof'
                labels_dict[file_id] = 1 if label == 'bonafide' else 0
    return labels_dict

def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python compute_eer_2021LA.py <score_file>")
        print("Example: python compute_eer_2021LA.py scores/k32_sparse_2021LA_scores.txt")
        sys.exit(1)
    
    score_file = sys.argv[1]
    metadata_file = '/data/projects/punim2637/nnliang/Datasets/keys/LA/CM/trial_metadata.txt'
    
    print(f"\n{'='*60}")
    print(f"Computing metrics for: {score_file}")
    print(f"{'='*60}\n")
    
    # Load data
    print("Loading scores...")
    scores_dict = load_scores(score_file)
    print(f"✓ Loaded {len(scores_dict)} scores")
    
    print("Loading labels...")
    labels_dict = load_labels(metadata_file)
    print(f"✓ Loaded {len(labels_dict)} labels")
    
    # Match scores with labels
    bonafide_scores = []
    spoof_scores = []
    matched = 0
    missing_labels = 0
    missing_scores = 0
    
    for file_id, label in labels_dict.items():
        if file_id in scores_dict:
            score = scores_dict[file_id]
            if label == 1:
                bonafide_scores.append(score)
            else:
                spoof_scores.append(score)
            matched += 1
        else:
            missing_scores += 1
    
    # Check for scores without labels
    for file_id in scores_dict:
        if file_id not in labels_dict:
            missing_labels += 1
    
    print(f"\n{'='*60}")
    print(f"Data matching:")
    print(f"  Matched: {matched}")
    print(f"  Bonafide samples: {len(bonafide_scores)}")
    print(f"  Spoof samples: {len(spoof_scores)}")
    if missing_scores > 0:
        print(f"  ⚠ Labels without scores: {missing_scores}")
    if missing_labels > 0:
        print(f"  ⚠ Scores without labels: {missing_labels}")
    print(f"{'='*60}\n")
    
    # Convert to numpy arrays
    bonafide_scores = np.array(bonafide_scores)
    spoof_scores = np.array(spoof_scores)
    
    # Compute statistics
    print("Score statistics:")
    print(f"  Bonafide: mean={bonafide_scores.mean():.4f}, std={bonafide_scores.std():.4f}, "
          f"min={bonafide_scores.min():.4f}, max={bonafide_scores.max():.4f}")
    print(f"  Spoof:    mean={spoof_scores.mean():.4f}, std={spoof_scores.std():.4f}, "
          f"min={spoof_scores.min():.4f}, max={spoof_scores.max():.4f}")
    
    # Compute metrics
    print(f"\n{'='*60}")
    print("Computing EER...")
    eer, threshold = compute_eer(bonafide_scores, spoof_scores)
    print(f"✓ EER: {eer:.4f}%")
    print(f"  Threshold: {threshold:.6f}")
    
    print("\nComputing min t-DCF...")
    min_tdcf = compute_min_tdcf(bonafide_scores, spoof_scores)
    print(f"✓ min t-DCF: {min_tdcf:.6f}")
    print(f"{'='*60}\n")
    
    # Save results
    result_file = score_file.replace('.txt', '_metrics.txt')
    with open(result_file, 'w') as f:
        f.write(f"Score file: {score_file}\n")
        f.write(f"Bonafide samples: {len(bonafide_scores)}\n")
        f.write(f"Spoof samples: {len(spoof_scores)}\n")
        f.write(f"EER: {eer:.4f}%\n")
        f.write(f"Threshold at EER: {threshold:.6f}\n")
        f.write(f"min t-DCF: {min_tdcf:.6f}\n")
    print(f"Results saved to: {result_file}\n")

if __name__ == '__main__':
    main()
