# Cue Consistency Experiment - Complete Methodology

## Overview

**Script**: `analyze_decision_relevance.py`
**Purpose**: 测量模型对同一utterance是否使用相同的decision cues
**Key Question**: Does the model switch between different decision features, or consistently use the same ones?

---

## Experimental Pipeline

### Phase 1: Feature Attribution (识别决策特征)
### Phase 2: Temporal Stability Analysis (测量稳定性)
### Phase 3: Cue Consistency Analysis (测量一致性) ← **核心**

---

## Phase 3: Cue Consistency Analysis - Detailed Implementation

### Step 1: Data Preparation

```python
# 从Phase 1获得的decision-relevant features
top_k_indices = [117, 2652, 1755, 1632, ...] # Top-50 features
decision_mask = np.zeros(4096, dtype=bool)
decision_mask[top_k_indices] = True  # 创建binary mask

# 从Phase 2收集的所有样本的SAE activations
genuine_activations = []  # List of [T, 4096] tensors (genuine samples)
spoof_activations = []    # List of [T, 4096] tensors (spoof samples)
```

**实际运行参数**（从decision_analysis_2021LA_5k）:
```bash
python analyze_decision_relevance.py \
  --model_path models/.../best_checkpoint_eer_window_topk_w8.pth \
  --output_dir decision_analysis_2021LA_5k \
  --dataset 2021_LA \
  --num_samples 5000 \        # 2500 genuine + 2500 spoof
  --top_k_features 50 \       # Top-50 decision features
  --batch_size 8
```

### Step 2: Compute Cue Overlap (核心算法)

**函数**: `CueConsistencyAnalyzer.compute_cue_overlap()`

```python
def compute_cue_overlap(self, activations, decision_mask):
    """
    测量同一utterance中consecutive frames使用的decision cues重叠度
    
    Args:
        activations: [T, dict_size]  # T个timesteps, 4096个特征
        decision_mask: [dict_size]    # Top-50 decision features的binary mask
    
    Returns:
        cue_overlap_score: scalar     # 平均Jaccard similarity
    """
    # Step 2.1: 提取decision features的激活
    decision_activations = activations[:, decision_mask]  # [T, 50]
    
    # Step 2.2: 二值化 - 判断每个feature是否active
    active_cues = (decision_activations > 0).float()      # [T, 50]
    # 例如: [[1,0,1,1,0,...], [1,1,1,0,0,...], ...]
    
    # Step 2.3: 计算consecutive frames的Jaccard similarity
    overlaps = []
    for t in range(T - 1):  # 对每一对相邻frames
        cue_t = active_cues[t]      # Frame t的active cue set
        cue_t1 = active_cues[t + 1] # Frame t+1的active cue set
        
        # Jaccard similarity = |A ∩ B| / |A ∪ B|
        intersection = (cue_t * cue_t1).sum()          # 同时active的features
        union = ((cue_t + cue_t1) > 0).float().sum()   # 至少一个active的features
        
        if union > 0:
            overlap = intersection / union  # Jaccard score for this frame pair
        else:
            overlap = 0.0
        
        overlaps.append(overlap)
    
    # Step 2.4: 对整个utterance求平均
    return np.mean(overlaps)  # 单个样本的cue consistency score
```

**具体例子**：

假设Top-50 decision features在某个样本的3个连续frames：

```
Frame t=0:  Features {A, B, C, D, E} active  (5个active)
Frame t=1:  Features {B, C, D, E, F} active  (5个active)
Frame t=2:  Features {C, D, E, F, G} active  (5个active)

Jaccard(t=0, t=1):
  Intersection = {B, C, D, E} = 4
  Union = {A, B, C, D, E, F} = 6
  Jaccard = 4/6 = 0.667 = 66.7%

Jaccard(t=1, t=2):
  Intersection = {C, D, E, F} = 4
  Union = {B, C, D, E, F, G} = 6
  Jaccard = 4/6 = 0.667 = 66.7%

Average Cue Overlap = (0.667 + 0.667) / 2 = 0.667 = 66.7%
```

但实际数据是**97-98%** overlap，远高于这个例子！

### Step 3: Identify Feature Usage Patterns

**函数**: `CueConsistencyAnalyzer.identify_feature_usage_pattern()`

```python
def identify_feature_usage_pattern(self, activations, decision_mask):
    """
    分类decision features: Persistent vs Transient vs Intermittent
    
    Args:
        activations: [T, dict_size]
        decision_mask: [dict_size] - Top-50 decision features
    
    Returns:
        dict: {
            'persistent': features active >50% of time,
            'transient': features active <20% of time,
            'intermittent': features in between (20-50%)
        }
    """
    decision_activations = activations[:, decision_mask]  # [T, 50]
    active = (decision_activations > 0).float()           # [T, 50]
    
    # 计算每个decision feature的activation rate
    activation_rates = active.mean(dim=0).numpy()  # [50]
    # 例如: [0.05, 0.82, 0.15, 0.63, ...]
    #       ↑      ↑      ↑      ↑
    #  transient persistent transient persistent
    
    decision_indices = np.where(decision_mask)[0]  # Top-50的真实index
    
    # 分类
    persistent = decision_indices[activation_rates > 0.5]   # Active >50% time
    transient = decision_indices[activation_rates < 0.2]    # Active <20% time
    intermittent = decision_indices[(0.2 <= activation_rates) & (activation_rates <= 0.5)]
    
    return {
        'persistent': persistent.tolist(),
        'transient': transient.tolist(),
        'intermittent': intermittent.tolist(),
        'activation_rates': activation_rates.tolist()
    }
```

**实际结果**：
```
Genuine samples:
  Persistent: 3.20 / 50   (6.4%)
  Transient: 46.39 / 50   (92.8%)

Spoof samples:
  Persistent: 2.87 / 50   (5.7%)
  Transient: 47.00 / 50   (94.0%)
```

### Step 4: Aggregate Across All Samples

```python
# Process all genuine samples
genuine_cue_overlaps = []
genuine_usage_patterns = []

for activations in tqdm(genuine_activations, desc="Analyzing genuine"):
    # 计算每个样本的cue overlap
    overlap = cue_analyzer.compute_cue_overlap(activations, decision_mask)
    genuine_cue_overlaps.append(overlap)
    
    # 识别feature usage pattern
    pattern = cue_analyzer.identify_feature_usage_pattern(activations, decision_mask)
    genuine_usage_patterns.append(pattern)

# Process all spoof samples
spoof_cue_overlaps = []
spoof_usage_patterns = []

for activations in tqdm(spoof_activations, desc="Analyzing spoof"):
    overlap = cue_analyzer.compute_cue_overlap(activations, decision_mask)
    spoof_cue_overlaps.append(overlap)
    
    pattern = cue_analyzer.identify_feature_usage_pattern(activations, decision_mask)
    spoof_usage_patterns.append(pattern)

# Final statistics
print(f"Genuine: {np.mean(genuine_cue_overlaps):.3f} ± {np.std(genuine_cue_overlaps):.3f}")
print(f"Spoof:   {np.mean(spoof_cue_overlaps):.3f} ± {np.std(spoof_cue_overlaps):.3f}")
```

**实际结果**：
```
Genuine: 0.973 ± 0.024  (97.3% ± 2.4%)
Spoof:   0.986 ± 0.030  (98.6% ± 3.0%)
```

### Step 5: Save Results

```python
results = {
    'cue_consistency': {
        'genuine': {
            'mean_overlap': float(np.mean(genuine_cue_overlaps)),      # 0.9726
            'std_overlap': float(np.std(genuine_cue_overlaps)),        # 0.0235
            'mean_persistent_features': float(persistent_genuine),     # 3.20
            'mean_transient_features': float(transient_genuine)        # 46.39
        },
        'spoof': {
            'mean_overlap': float(np.mean(spoof_cue_overlaps)),        # 0.9861
            'std_overlap': float(np.std(spoof_cue_overlaps)),          # 0.0298
            'mean_persistent_features': float(persistent_spoof),       # 2.87
            'mean_transient_features': float(transient_spoof)          # 47.00
        }
    }
}

with open('decision_analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)
```

---

## Key Metrics Explained

### 1. Mean Cue Overlap (核心指标)

**定义**: 同一utterance中consecutive frames的activated decision feature subspace的平均Jaccard similarity

**计算**:
```
For each sample:
  For each frame pair (t, t+1):
    overlap = |active_cues_t ∩ active_cues_t+1| / |active_cues_t ∪ active_cues_t+1|
  sample_overlap = mean(all frame pairs)

final_metric = mean(all samples)
```

**解释**:
- **97.3%**: 相邻frames有97.3%的decision features重叠
- **High value** = 模型consistently使用相同的decision cues
- **Low value** = 模型频繁切换decision strategies

### 2. Persistent vs Transient Features

**定义**:
- **Persistent**: Active >50% of utterance duration
- **Transient**: Active <20% of utterance duration
- **Intermittent**: Active 20-50% of time

**目的**: 理解decision features的temporal dynamics

**发现**:
- 92.8% of decision features are **transient**
- Yet 97.3% cue overlap ← 看似矛盾！
- **解释**: Coordinated activation patterns

---

## What Makes This "Cue Consistency" vs "Temporal Stability"?

### Temporal Stability (Phase 2)
- **测量**: 所有4096个features的激活稳定性
- **方法**: Jaccard similarity on all active features
- **结果**: 96.49% (all features), 97.94% (decision features)

### Cue Consistency (Phase 3)
- **测量**: 只看Top-50 decision features的重复使用
- **方法**: Jaccard similarity on activated decision cue subset
- **结果**: 97.26% (genuine), 98.61% (spoof)

**关键差异**:
```
Temporal Stability:
  "特征是否保持激活?" (Binary: active or not)
  
Cue Consistency:
  "相同的决策线索集合是否被重复使用?" (Set overlap)
```

**Example**:

```
假设4096个特征中，Top-50是decision features

Temporal Stability看全局:
  t=0: Features {1, 5, 17, 99, 117, 200, ...} active  (128个)
  t=1: Features {1, 5, 17, 99, 117, 201, ...} active  (128个)
  Jaccard = 127/129 = 98.4%
  
Cue Consistency只看Top-50:
  t=0: Decision features {17, 99, 117} active      (3个decision features)
  t=1: Decision features {17, 99, 117, 200} active (4个decision features)
  Jaccard = 3/4 = 75%
  
但实际Cue Consistency是97%！说明decision features非常consistent
```

---

## Why This Matters

### 1. Answers Caren's Question

> "Explicitly measure cue reuse for the same utterance"

**✅ Done**: 97-98% of decision cues are reused across consecutive frames

### 2. Diagnostic Value

**Finding 1**: Spoof detection uses MORE consistent cues (98.6% vs 97.3%)
- Implication: Deepfake artifacts are more stereotyped than natural speech

**Finding 2**: 92.8% transient yet 97.3% consistent
- Implication: Consistency comes from coordinated activation, not persistence

**Finding 3**: Window TopK achieves near-optimal cue consistency
- Implication: No need for explicit temporal regularization

### 3. Failed Training Objective Attempts

**CPC Loss**: Tried to improve beyond 97% → EER degraded to 9.04%
**Overlapping Windows**: Tried to smooth boundaries → EER degraded to 7.22%

**Lesson**: 97-98% cue consistency is already optimal balance point

---

## Data Flow Summary

```
Input: Trained Window TopK SAE model + 5000 samples

Phase 1: Attribution Analysis
  ↓ Gradient-based attribution
  → Top-50 decision-relevant features identified
  
Phase 2: Temporal Stability Analysis  
  ↓ All features + Decision features
  → Stability metrics computed
  
Phase 3: Cue Consistency Analysis
  ↓ For each sample:
     1. Extract Top-50 decision feature activations [T, 50]
     2. Binarize: active (>0) or not
     3. Compute Jaccard for each frame pair
     4. Average across utterance
  ↓ Aggregate across all samples
  → Mean overlap: 97.26% (genuine), 98.61% (spoof)
  
  ↓ For each sample:
     1. Compute activation rate for each decision feature
     2. Classify: Persistent (>50%), Transient (<20%), Intermittent
     3. Count features in each category
  ↓ Aggregate
  → Persistent: 3.20, Transient: 46.39 (genuine)
  
Output: decision_analysis_results.json
  - cue_consistency.genuine.mean_overlap: 0.9726
  - cue_consistency.spoof.mean_overlap: 0.9861
  - cue_consistency.genuine.mean_transient_features: 46.39
```

---

## Reproducibility Details

**Hardware**: GPU (CUDA)
**Batch Size**: 8
**Samples**: 5000 (2500 genuine + 2500 spoof)
**Processing Time**: ~1 hour (estimate)
**Output Files**:
- `decision_analysis_results.json` (main results)
- `feature_attributions.json` (attribution scores)

**Key Parameters**:
- `top_k_features`: 50
- `window_size`: 8
- `dict_size`: 4096
- `k` (TopK): 128

---

## Interpretation of Results

### High Cue Overlap (97-98%)

**What it means**:
- Model uses nearly identical decision cue sets across frames
- Stable, consistent decision-making
- Not switching between different reasoning strategies

**Good or bad?**
✅ **Good for reliability**: Predictions based on stable evidence
✅ **Good for interpretability**: Can identify consistent decision drivers
⚠️ **Already optimal**: Further improvement triggers trade-off

### Mostly Transient (92.8%)

**What it means**:
- Most decision features activate briefly (<20% of time)
- Yet maintain 97% subspace overlap
- Coordinated activation patterns

**Good or bad?**
✅ **Good for discrimination**: Transient = responsive to brief artifacts
✅ **Explains CPC failure**: Forcing persistence destroys discrimination
⚠️ **Non-intuitive**: Transience + Consistency seem contradictory

### Class Difference (98.6% vs 97.3%)

**What it means**:
- Spoof detection 1.3% more consistent than genuine verification
- Deepfake artifacts may be more stereotyped
- Simpler decision boundary for spoofs

**Good or bad?**
✅ **Insight for future work**: Could exploit this asymmetry
✅ **Explains training dynamics**: 10:1 spoof:genuine ratio
🤔 **Open question**: Attack-specific patterns?

---

## Comparison with Other Analyses

### vs Temporal Stability Analysis

| Metric | Temporal Stability | Cue Consistency |
|--------|-------------------|-----------------|
| **Scope** | All 4096 features | Top-50 decision features |
| **Measure** | Feature activation stability | Decision cue reuse |
| **Result** | 97.94% (decision features) | 97.26% (genuine), 98.61% (spoof) |
| **Insight** | Decision features more stable | Cue reuse highly consistent |

### vs Feature Lifetime Analysis

| Metric | Lifetime Analysis | Cue Consistency |
|--------|-------------------|-----------------|
| **Granularity** | Individual features | Feature subsets |
| **Measure** | Consecutive activation duration | Set overlap |
| **Result** | 10.22 timesteps (decision) | 97-98% overlap |
| **Insight** | Features are transient | Subspace is stable |

**Complementary Findings**:
- Lifetime: Individual features brief (10.22 steps)
- Consistency: Subspace highly overlapping (97%)
- **Resolution**: Coordinated activation patterns

---

## Limitations and Future Work

### Limitations

1. **Frame-level granularity**: Only consecutive frames
   - Could extend to longer temporal windows
   - Measure consistency across non-adjacent segments

2. **Binary classification**: Active vs inactive
   - Could use activation magnitudes
   - Weight overlap by feature importance

3. **Top-50 threshold**: Arbitrary cutoff
   - Could analyze sensitivity to k
   - Try different top-K values

### Future Directions

1. **Attack-specific analysis**:
   - Do different attacks use different cue sets?
   - Consistency varies by spoofing method?

2. **Temporal localization**:
   - Where in utterance do decision cues activate?
   - Artifact-specific temporal patterns?

3. **Exploit asymmetry**:
   - Use higher spoof consistency for training?
   - Class-specific temporal constraints?

---

## Bottom Line

**Cue Consistency Analysis measures whether the model uses the same decision features repeatedly for the same utterance.**

**Method**: Jaccard similarity of activated Top-50 decision feature subspaces across consecutive frames

**Result**: 97-98% cue reuse (exceptionally high)

**Implication**: Window TopK achieves near-optimal cue consistency through structural constraints, explaining why explicit temporal losses fail.

**Value**: Diagnostic insight into model decision-making, not a training objective.
