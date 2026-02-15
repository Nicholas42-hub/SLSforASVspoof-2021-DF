# Decision Cue Consistency Analysis - Quick Start

This implements **Direction #4** from Caren's feedback: measuring decision cue consistency and using it to guide regularization design.

## Overview

The analysis has three phases:

### Phase 1: Feature Attribution Analysis
**Goal**: Identify which SAE features causally influence predictions

**Method**: Gradient-based attribution (∂logits/∂SAE_activations)

**Output**: 
- Ranking of features by decision relevance
- Top-K most influential features
- Optional ablation validation

### Phase 2: Decision-Relevant Stability Analysis  
**Goal**: Measure temporal stability specifically for high-influence features

**Method**: Compute Jaccard, lifetime, flipping rate for decision features vs. all features

**Output**:
- Stability metrics comparison
- Boundary vs. interior analysis
- Identification of where instability occurs

### Phase 3: Cue Consistency Analysis
**Goal**: Measure whether the model reuses the same decision cues consistently

**Method**: Track which decision features activate together over time

**Output**:
- Cue overlap scores (genuine vs. spoof)
- Feature usage patterns (persistent vs. transient)
- Class-specific decision strategies

## Quick Start

### 1. Basic Run (200 samples, ~30 minutes)

```bash
# Interactive testing
python analyze_decision_relevance.py \
    --model_path checkpoints/window_topk/best_model.pth \
    --output_dir decision_analysis \
    --num_samples 200 \
    --top_k_features 50

# On SLURM cluster
sbatch run_decision_analysis.slurm
```

### 2. Full Analysis with Ablation Validation

```bash
python analyze_decision_relevance.py \
    --model_path checkpoints/window_topk/best_model.pth \
    --output_dir decision_analysis \
    --num_samples 200 \
    --top_k_features 50 \
    --run_ablation  # Validates top-10 features via ablation (slower)
```

### 3. Comprehensive Analysis (500 samples)

```bash
python analyze_decision_relevance.py \
    --model_path checkpoints/window_topk/best_model.pth \
    --output_dir decision_analysis_full \
    --num_samples 500 \
    --top_k_features 100 \
    --batch_size 8 \
    --run_ablation
```

## Interpreting Results

### Key Findings to Look For

#### 1. Decision Feature Instability
```
All features Jaccard: 0.650
Decision features Jaccard: 0.450  ⚠️
```
→ **If decision features are MORE unstable**: Decision brittleness is the problem!
→ The model is switching between different reasoning for the same content

#### 2. Boundary Disruption
```
Interior Jaccard: 0.520
Cross-boundary Jaccard: 0.280  ⚠️
```
→ **If cross-boundary << interior**: Hard windowing disrupts decision features
→ Need softer temporal constraints or overlapping windows

#### 3. Class-Specific Patterns
```
Genuine cue overlap: 0.620
Spoof cue overlap: 0.380  ⚠️
```
→ **If spoof < genuine**: Spoof detection is more inconsistent
→ May need stronger regularization for spoof-specific features

#### 4. Feature Behavior Types
```
Persistent features: 12  (should be stable, instability is bad)
Transient features: 25   (respond to brief artifacts, instability may be ok)
```
→ Helps design **adaptive regularization** - constrain persistent, allow transient

## Output Files

### `decision_analysis/`

1. **`feature_attributions.json`**
   - Attribution score for each SAE feature
   - Top-K decision-relevant feature indices
   - Optional ablation validation scores

2. **`decision_analysis_results.json`**
   - Summary statistics
   - Stability metrics (all vs. decision features)
   - Boundary effect analysis
   - Cue consistency scores (genuine vs. spoof)

3. **Visualizations:**
   - `attribution_distribution.png` - Which features matter
   - `stability_comparison.png` - All vs. decision feature stability
   - `cue_consistency.png` - Genuine vs. spoof cue reuse

## Next Steps After Analysis

### If decision features are unstable:

**Option A: Adaptive Temporal Regularization**
```python
# Weight loss by feature importance
loss = (w_i * ||a_i^t - a_i^{t+1}||^2).sum()
# where w_i = attribution_score_i
```

**Option B: Cue Consistency Regularization**
```python
# Encourage consistent decision cue sets
loss = 1 - Jaccard(active_cues_t, active_cues_{t+1})
```

**Option C: Boundary-Specific Regularization**
```python
# Higher penalty at window boundaries
if is_boundary(t):
    loss *= 5.0
```

### If genuine/spoof differ:

**Class-Specific Regularization**
```python
# Different constraints for genuine vs. spoof features
if label == genuine:
    loss = temporal_loss(genuine_specific_features) * λ_1
else:
    loss = temporal_loss(spoof_specific_features) * λ_2
```

## Integration with Training

After identifying decision-relevant features, you can:

1. **Static approach**: Save feature attributions, use fixed weights during training
2. **Dynamic approach**: Recompute attributions periodically (e.g., every 5 epochs)
3. **Hybrid**: Start with static, refine with dynamic updates

## Troubleshooting

### Memory Issues
- Reduce `--batch_size` to 2 or 1
- Reduce `--num_samples`
- Run without `--run_ablation`

### Slow Execution
- Ablation is slow (~10 min for 10 features)
- Start without `--run_ablation`, add later for validation
- Use fewer samples for initial exploration

### Model Loading Errors
- Check that model checkpoint contains SAE weights
- Verify `sae_dict_size`, `sae_k`, `sae_window_size` match training config
- Try loading checkpoint manually first to debug

## Expected Runtime

| Configuration | Samples | Ablation | Time | GPU Memory |
|--------------|---------|----------|------|------------|
| Quick        | 200     | No       | ~30min | ~8GB |
| Standard     | 200     | Yes (10) | ~45min | ~8GB |
| Full         | 500     | No       | ~1h    | ~12GB |
| Comprehensive| 500     | Yes (10) | ~1.5h  | ~12GB |

## Contact

For questions about implementation or interpretation, refer to:
- `IMPLEMENTATION_PLAN.md` - Detailed technical roadmap
- `response_to_caren_jan2026.txt` - Project rationale
- Caren's original email - Context for the analysis

---

**Remember**: The goal is not just to measure instability, but to understand:
1. WHERE it occurs (boundaries? everywhere?)
2. WHAT causes it (windowing? model inherent?)
3. WHETHER it matters (decision-relevant features? or irrelevant ones?)

This understanding guides the design of principled regularization strategies.
