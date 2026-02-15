#!/usr/bin/env python
import numpy as np
import sys

def compute_det_curve(target_scores, nontarget_scores):
    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate((np.ones(target_scores.size), np.zeros(nontarget_scores.size)))
    
    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]
    
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (np.arange(1, n_scores + 1) - tar_trial_sums)
    
    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size))
    thresholds = np.concatenate((np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))
    
    return frr, far, thresholds

def compute_eer(target_scores, nontarget_scores):
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, thresholds[min_index]

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <score_file> <protocol_file>")
        sys.exit(1)
    
    score_file = sys.argv[1]
    protocol_file = sys.argv[2]
    
    # Load scores and protocol
    scores = []
    with open(score_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                scores.append(float(parts[1]))  # bonafide probability
    
    labels = []
    with open(protocol_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                label = parts[4]  # bonafide or spoof
                labels.append(1 if label == 'bonafide' else 0)
    
    scores = np.array(scores)
    labels = np.array(labels)
    
    if len(scores) != len(labels):
        print(f"Error: Number of scores ({len(scores)}) != number of labels ({len(labels)})")
        sys.exit(1)
    
    # Separate bonafide and spoof scores
    bonafide_scores = scores[labels == 1]
    spoof_scores = scores[labels == 0]
    
    print(f"Total trials: {len(scores)}")
    print(f"Bonafide trials: {len(bonafide_scores)}")
    print(f"Spoof trials: {len(spoof_scores)}")
    
    # Compute EER
    eer, threshold = compute_eer(bonafide_scores, spoof_scores)
    
    print(f"\nResults:")
    print(f"EER: {eer * 100:.4f}%")
    print(f"Threshold: {threshold:.6f}")
