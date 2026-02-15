# Implementation Plan: Decision-Aware SAE Analysis

## Phase 1: Cue Consistency Analysis (Diagnostic)

### 1.1 Feature Attribution Analysis
**Goal**: Identify which SAE features causally influence predictions

**Implementation** (`analyze_decision_relevance.py`):
```python
# Core components:
# 1. Load trained model + SAE
# 2. Forward pass with gradient tracking
# 3. Compute ∂logits/∂SAE_activations for each feature
# 4. Aggregate attribution scores across frames/utterances

Key functions:
- compute_feature_attribution(model, sae, audio) -> attribution_scores
- rank_features_by_influence() -> top_k_indices
- visualize_attribution_distribution()
```

**Input**: Trained model, SAE checkpoint, subset of eval data (100-200 utterances)
**Output**: 
- `feature_attributions.json`: Per-feature causal relevance scores
- `top_k_decision_features.npy`: Indices of most influential features
- Visualization of attribution distribution

**Runtime estimate**: ~1-2 hours for 200 utterances

---

### 1.2 Decision-Relevant Temporal Stability
**Goal**: Measure temporal stability specifically for high-influence features

**Implementation** (`analyze_decision_cue_stability.py`):
```python
# Core components:
# 1. Load feature attributions from 1.1
# 2. Filter activations to only decision-relevant features
# 3. Recompute temporal metrics (Jaccard, lifetime, flipping)

Key metrics:
- Decision Jaccard: Jaccard similarity over top-K influential features only
- Decision Lifetime: How long influential features stay active
- Decision Flipping Rate: Instability in decision-relevant activations

Compare with all-feature metrics to see if instability is concentrated
```

**Input**: SAE activations, feature attributions
**Output**:
- `decision_relevant_stability.json`: Metrics for influential vs. all features
- Comparison plots showing whether instability is concentrated

**Runtime estimate**: ~30 minutes

---

### 1.3 Decision Cue Overlap Metric
**Goal**: Measure whether same decision cues are reused within utterances

**Implementation** (add to `analyze_decision_cue_stability.py`):
```python
# For each utterance:
# 1. At each frame, identify which decision-relevant features are active
# 2. Compute pairwise overlap of active decision features across frames
# 3. Aggregate as "cue consistency score" per utterance

Metric definition:
- Cue Overlap(u) = (1/|T|) Σ_t Jaccard(ActiveDecisionFeatures_t, ActiveDecisionFeatures_{t+1})
- Compare genuine vs. spoofed utterances
```

**Output**:
- `cue_consistency_by_class.json`: Mean cue overlap for genuine/spoof
- Statistical test: whether genuine samples have more consistent cue usage

---

### 1.4 Window Boundary Impact on Decision Cues
**Goal**: Check if window boundaries specifically disrupt important features

**Implementation** (`analyze_boundary_impact_on_decisions.py`):
```python
# For window-based model:
# 1. Identify frames near window boundaries (e.g., ±2 frames)
# 2. Compute decision cue stability near boundaries vs. within windows
# 3. Check if attribution scores drop/spike at boundaries

Analysis:
- Do influential features show larger discontinuities at boundaries?
- Is decision stability lower near boundaries?
```

**Output**:
- `boundary_impact_analysis.json`: Metrics for boundary vs. non-boundary frames
- Visualization: decision relevance and stability across window boundaries

---

## Phase 2: Decision-Aware Regularization (Solution)

### 2.1 Implement Decision-Aware Temporal Loss
**Goal**: Regularize temporal consistency weighted by causal relevance

**Implementation** (modify `model_window_topk.py`):
```python
class DecisionAwareTemporalLoss(nn.Module):
    def __init__(self, feature_weights):
        """
        feature_weights: [num_features] tensor of attribution scores
        """
        self.weights = feature_weights / feature_weights.sum()  # normalize
    
    def forward(self, activations):
        """
        activations: [batch, time, num_features]
        """
        # Compute temporal differences
        diff = activations[:, 1:, :] - activations[:, :-1, :]
        
        # Weight by feature importance
        weighted_diff = diff ** 2 * self.weights[None, None, :]
        
        # Average over time and features
        return weighted_diff.mean()

# In training loop:
temporal_loss = decision_aware_temporal_loss(sae_activations)
total_loss = classification_loss + lambda_temporal * temporal_loss
```

**Key decisions**:
- What threshold for "decision-relevant"? (e.g., top 10%, top 50 features)
- What λ_temporal weight? (start with 0.01, tune via grid search)
- Apply before or after TopK? (probably before to influence selection)

---

### 2.2 Training Script with Decision-Aware Loss
**Goal**: Train window-based SAE with decision-aware regularization

**Implementation** (`train_decision_aware_sae.py`):
```python
# Training procedure:
# 1. Pretrain SAE without temporal loss (or load checkpoint)
# 2. Compute feature attributions on validation set
# 3. Initialize decision-aware temporal loss with attribution weights
# 4. Continue training with combined loss

# Hyperparameters to tune:
# - lambda_temporal: [0.001, 0.01, 0.1]
# - Attribution computation frequency: every N epochs or static
```

**Slurm script**: `train_decision_aware.slurm`

---

### 2.3 Evaluation and Comparison
**Goal**: Validate that decision-aware regularization improves stability

**Implementation** (`evaluate_decision_aware_model.py`):
```python
# Compare three models:
# 1. Baseline (per-frame TopK)
# 2. Window-based (hard boundaries)
# 3. Decision-aware (proposed)

# Metrics:
# - EER on eval set (must not degrade)
# - Decision cue consistency (should improve)
# - Decision-relevant stability (should improve)
# - Boundary discontinuity (should reduce for important features)
```

**Output**: Comparative table for paper/presentation

---

## Timeline Estimate

| Task | Estimated Time | Dependencies |
|------|----------------|--------------|
| 1.1 Feature Attribution | 1 day (code) + 2h (compute) | Trained model |
| 1.2 Decision-Relevant Stability | 0.5 day | 1.1 complete |
| 1.3 Cue Overlap Metric | 0.5 day | 1.1 complete |
| 1.4 Boundary Impact | 0.5 day | 1.1, 1.2 complete |
| **Phase 1 Total** | **2.5 days** | |
| 2.1 Implement Loss | 1 day | 1.1 complete |
| 2.2 Training | 0.5 day (setup) + compute time | 2.1 complete |
| 2.3 Evaluation | 1 day | 2.2 complete |
| **Phase 2 Total** | **2.5 days + training time** | |

**Total for Phase 1 (for next meeting)**: ~3 days of implementation + compute

---

## Next Steps

1. **Immediate**: Start with 1.1 (feature attribution analysis)
2. **Data selection**: Choose ~200 balanced utterances from eval set for analysis
3. **Baseline**: Rerun existing temporal metrics for comparison
4. **Checkpoints**: Use current best window-based model checkpoint

## Key Questions to Resolve

1. **Attribution method**: Gradient-based (∂logits/∂activations) or intervention-based (ablation)?
   - Recommendation: Start with gradients (faster), validate with ablation on subset
   
2. **Top-K threshold**: What percentage of features count as "decision-relevant"?
   - Recommendation: Analyze distribution, likely top 5-20% based on attribution magnitude
   
3. **Aggregation**: Per-frame attribution vs. per-utterance?
   - Recommendation: Compute per-frame, aggregate to get stable feature importance

4. **Class-specific**: Do genuine and spoofed samples use different decision features?
   - Important to analyze! May need separate attribution analysis per class
