# Classification-Based Attention Visualization - Quick Start

## ✅ What's New

The visualization system now supports analyzing attention patterns based on **classification correctness**:

1. **Correctly Classified Bonafide** ✅ (True label=1, Prediction=1)
2. **Incorrectly Classified Bonafide** ❌ (True label=1, Prediction=0) - False Rejection
3. **Correctly Classified Spoof** ✅ (True label=0, Prediction=0)
4. **Incorrectly Classified Spoof** ❌ (True label=0, Prediction=1) - False Acceptance

## 🚀 Quick Start

### For datasets WITH true labels (e.g., dev set):

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_with_attention_viz.py \
  --checkpoint /path/to/checkpoint.pth \
  --database_path /path/to/dev_set/ \
  --protocols_path /path/to/dev_protocol.txt \
  --track LA \
  --viz_dir classification_viz \
  --num_viz_samples 1000 \
  --batch_size 16 \
  --group_size 3 \
  --has_labels \
  --classification_viz
```

### For datasets WITHOUT true labels (e.g., unlabeled eval):

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_with_attention_viz.py \
  --checkpoint /path/to/checkpoint.pth \
  --database_path /path/to/eval_set/ \
  --protocols_path /path/to/eval_protocol.txt \
  --track LA \
  --viz_dir standard_viz \
  --num_viz_samples 500 \
  --batch_size 16 \
  --group_size 3
```

## 📊 Output Files

### Standard Visualizations (Always Generated):

- `layer_weight_heatmap_spoof.png` - Layer attention for predicted spoof
- `layer_weight_heatmap_bonafide.png` - Layer attention for predicted bonafide
- `layer_weight_comparison.png` - Side-by-side comparison
- `temporal_attention_heatmap_spoof.png` - Time×layer attention (spoof)
- `temporal_attention_heatmap_bonafide.png` - Time×layer attention (bonafide)
- `intra_group_attention_spoof.png` - Within-group attention (spoof)
- `intra_group_attention_bonafide.png` - Within-group attention (bonafide)
- `inter_group_attention_spoof.png` - Between-group attention (spoof)
- `inter_group_attention_bonafide.png` - Between-group attention (bonafide)

### Classification Visualizations (Only with --has_labels --classification_viz):

- `layer_weight_correct_bonafide.png` - Correctly classified bonafide samples
- `layer_weight_incorrect_bonafide.png` - False rejections
- `layer_weight_correct_spoof.png` - Correctly classified spoof samples
- `layer_weight_incorrect_spoof.png` - False acceptances
- `classification_comparison_4way.png` - **2×2 grid comparing all 4 categories**

## 🎯 Use Cases

### Research Questions You Can Answer:

1. **Error Analysis**:

   - Why does the model misclassify certain samples?
   - Do incorrectly classified samples have different attention patterns?

2. **Attack Vulnerability**:

   - What attention patterns do successful spoofs (false acceptances) exhibit?
   - Do they mimic bonafide attention or show unique patterns?

3. **Layer Importance**:

   - Which layers are most important for correct classification?
   - Do errors come from attending to wrong layers?

4. **Model Interpretability**:
   - How does the model's attention differ between correct and incorrect predictions?
   - Can we identify attention signatures of errors?

## 📝 Example Output (Statistics)

When using `--has_labels`, you'll see classification statistics:

```
📊 Classification Statistics:
   Overall Accuracy: 98.50% (985/1000)

   Bonafide (label=1):
      ✅ Correct: 490/500 (98.00%)
      ❌ Incorrect (False Rejection): 10/500 (2.00%)

   Spoof (label=0):
      ✅ Correct: 495/500 (99.00%)
      ❌ Incorrect (False Acceptance): 5/500 (1.00%)
```

## 🔧 New Command-Line Arguments

| Argument               | Description                                  | Required            |
| ---------------------- | -------------------------------------------- | ------------------- |
| `--has_labels`         | Dataset provides true labels (not just IDs)  | No (default: False) |
| `--classification_viz` | Generate classification-based visualizations | No (default: False) |

**Note**: `--classification_viz` requires `--has_labels` to be set.

## 💡 Tips

1. **Use development/validation sets** for classification analysis (they have labels)
2. **Collect enough samples** (recommend 500-1000) to ensure all categories have data
3. **Compare across categories** to identify what makes errors different
4. **Look for patterns** in false acceptances - these are the most critical errors

## 📚 Files Created

- `CLASSIFICATION_VISUALIZATION_GUIDE.md` - Detailed documentation
- `run_classification_viz.sh` - Example shell script
- Updated `visualize_attention_evaluation.py` - Core visualization library
- Updated `evaluate_with_attention_viz.py` - Command-line interface

## 🐛 Troubleshooting

**"Labels and predictions are identical"**

- You're using predictions as labels (no true labels provided)
- Solution: Use `--has_labels` with a labeled dataset

**"No samples found for category X"**

- Perfect classification or insufficient samples
- Solution: Increase `--num_viz_samples`

**Memory error**

- Too many samples or large batch size
- Solution: Reduce `--batch_size` or `--num_viz_samples`

## 🎨 Visualization Features

All heatmaps include:

- ✅ **White grid lines** for clarity
- ✅ **Viridis colormap** (publication-quality)
- ✅ **300 DPI resolution** (high quality)
- ✅ **Layer on Y-axis** (vertical)
- ✅ **Samples on X-axis** (horizontal)

## 📧 Questions?

See `CLASSIFICATION_VISUALIZATION_GUIDE.md` for detailed documentation.
