# Dataset Balance Issue & Solution

## Problem

When running classification-based visualizations, you encountered:

```
✅ Collected attention weights from 250 samples
   📊 Bonafide: 0, Spoof: 250
```

**Root Cause**: The ASVspoof2021 LA evaluation dataset is heavily imbalanced, with spoof samples appearing first in the dataset. The balanced sampler processed 11,348 batches but only found spoof samples before reaching the 250-sample quota.

## Quick Diagnosis

Before running visualizations, check your dataset balance:

```bash
python check_dataset_balance.py \
  --protocols_path /path/to/ASVspoof2021.LA.cm.eval.trl.txt
```

**Expected Output:**

```
📊 Label Distribution:
   Total samples: 181566
   Bonafide (label=1): XXXX (X.XX%)
   Spoof (label=0): XXXX (XX.XX%)

🎯 Balanced Sampling Feasibility:
   ✅ Can collect up to XXXX balanced samples
      (XXXX from each class)
```

## Solutions

### Option 1: Disable Balanced Sampling (Quick Fix)

Collect samples without balancing - visualizations will adapt to available classes:

```python
# In evaluate_with_attention_viz.py, line 166
visualizer.collect_attention_weights(
    eval_loader,
    num_samples=args.num_viz_samples,
    has_labels=args.has_labels,
    label_dict=label_dict,
    balanced_sampling=False  # Changed from True
)
```

**Pros**: Works immediately with any dataset
**Cons**: May get mostly one class; some visualizations skipped

### Option 2: Use Shuffled DataLoader (Recommended)

Shuffle the dataset so bonafide and spoof samples are interleaved:

```python
# In evaluate_with_attention_viz.py, around line 140
eval_loader = DataLoader(
    eval_set,
    batch_size=args.batch_size,
    shuffle=True,  # Add this!
    num_workers=4
)
```

**Pros**: Balanced sampling works correctly
**Cons**: Slightly longer startup time

### Option 3: Increase Number of Samples

If the dataset is very imbalanced, collect more samples to ensure both classes are found:

```bash
# Instead of 500, try 2000 or more
--num_viz_samples 2000
```

**Pros**: Higher chance of finding minority class
**Cons**: Longer collection time, more memory usage

### Option 4: Manual Class-Specific Collection (Advanced)

Collect bonafide and spoof samples separately:

1. First pass: collect only bonafide (`label == 1`)
2. Second pass: collect only spoof (`label == 0`)
3. Combine results

This guarantees balanced collection but requires code modifications.

## Updated Code Features

The visualization code now includes:

### 1. **Graceful Handling of Missing Classes**

- Checks if each class has samples before creating visualizations
- Skips visualizations for missing classes instead of crashing
- Shows warnings: `⚠️ No Bonafide samples available - skipping visualization`

### 2. **Class Balance Warning**

```
⚠️  WARNING: Imbalanced dataset detected!
   Bonafide samples: 0
   Spoof samples: 250
   Some visualizations will be skipped.
```

### 3. **Safe Division in Statistics**

```python
if bonafide_total > 0:
    print(f"✅ Correct: {bonafide_correct}/{bonafide_total} ...")
else:
    print(f"⚠️  No bonafide samples collected")
```

## Recommended Workflow

1. **Check dataset balance first:**

   ```bash
   python check_dataset_balance.py --protocols_path /path/to/protocol.txt
   ```

2. **Use shuffled dataloader** (best for balanced datasets):

   ```python
   eval_loader = DataLoader(..., shuffle=True)
   ```

3. **Run visualization with appropriate sample count:**

   ```bash
   # If balanced: 500 samples = 250 each class
   --num_viz_samples 500

   # If imbalanced: increase to ensure minority class appears
   --num_viz_samples 2000
   ```

4. **Check collection output:**
   ```
   ✅ Collected attention weights from 500 samples
      📊 Bonafide: 250, Spoof: 250  ← Should be balanced!
   ```

## Expected Behavior

### With Balanced Collection (Working)

```
📊 Bonafide: 250, Spoof: 250
✅ All 14 visualizations generated
```

### With Imbalanced Collection (Handled Gracefully)

```
📊 Bonafide: 0, Spoof: 250
⚠️  Skipping bonafide-specific visualizations
✅ Generated 5 spoof-only visualizations
```

## Files Updated

1. `visualize_attention_evaluation.py` - Added graceful handling for missing classes
2. `check_dataset_balance.py` - New utility to check dataset balance
3. `evaluate_with_attention_viz.py` - Can enable shuffle in DataLoader

## Next Steps

1. **Upload updated files to server:**

   ```bash
   scp visualize_attention_evaluation.py root@server:/root/autodl-tmp/SLSforASVspoof-2021-DF/
   scp check_dataset_balance.py root@server:/root/autodl-tmp/SLSforASVspoof-2021-DF/
   ```

2. **Check dataset balance on server:**

   ```bash
   python check_dataset_balance.py \
     --protocols_path /root/autodl-tmp/CLAD/Datasets/ASVspoof2021_LA_eval/ASVspoof2021.LA.cm.eval.trl.txt
   ```

3. **Choose solution based on results** (see Options 1-4 above)

4. **Re-run visualization** with chosen fix
