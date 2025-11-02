# Classification-Based Attention Visualization Guide

This guide explains how to generate attention visualizations based on classification correctness (correct vs incorrect predictions for bonafide and spoof samples).

## Overview

The enhanced visualization system now supports analyzing attention patterns based on four classification categories:

1. **Correctly Classified Bonafide** (label=1, prediction=1) ✅
2. **Incorrectly Classified Bonafide** (label=1, prediction=0) - False Rejection (FR) ❌
3. **Correctly Classified Spoof** (label=0, prediction=0) ✅
4. **Incorrectly Classified Spoof** (label=0, prediction=1) - False Acceptance (FA) ❌

## Requirements

Your dataset **must provide true labels** to use classification-based visualizations. This typically means using:

- Training set
- Development/validation set
- Evaluation set with ground truth labels

## Usage

### Basic Command (Standard Visualizations Only)

For datasets **without** true labels (e.g., unlabeled evaluation sets):

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_with_attention_viz.py \
  --checkpoint /path/to/checkpoint.pth \
  --database_path /path/to/data/ \
  --protocols_path /path/to/protocol.txt \
  --track LA \
  --viz_dir attention_viz_output \
  --num_viz_samples 500 \
  --batch_size 16 \
  --group_size 3
```

This generates:

- `layer_weight_heatmap_spoof.png`
- `layer_weight_heatmap_bonafide.png`
- `layer_weight_comparison.png`
- `temporal_attention_heatmap_*.png`
- `intra_group_attention_*.png`
- `inter_group_attention_*.png`

### Classification-Based Visualizations

For datasets **with** true labels:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_with_attention_viz.py \
  --checkpoint /path/to/checkpoint.pth \
  --database_path /path/to/labeled_data/ \
  --protocols_path /path/to/protocol_with_labels.txt \
  --track LA \
  --viz_dir classification_viz_output \
  --num_viz_samples 500 \
  --batch_size 16 \
  --group_size 3 \
  --has_labels \
  --classification_viz
```

**New Flags:**

- `--has_labels`: Indicates that the dataset provides true labels
- `--classification_viz`: Generate classification-based visualizations

This generates **all standard visualizations PLUS**:

- `layer_weight_correct_bonafide.png` - Attention for correctly classified bonafide
- `layer_weight_incorrect_bonafide.png` - Attention for false rejections
- `layer_weight_correct_spoof.png` - Attention for correctly classified spoof
- `layer_weight_incorrect_spoof.png` - Attention for false acceptances
- `classification_comparison_4way.png` - 2×2 grid comparing all four categories

## Example: LA Development Set

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_with_attention_viz.py \
  --checkpoint /root/autodl-tmp/SLSforASVspoof-2021-DF/models/.../best_checkpoint_eer.pth \
  --database_path /root/autodl-tmp/CLAD/Datasets/ASVspoof2021_LA_dev/ \
  --protocols_path /root/autodl-tmp/CLAD/Datasets/ASVspoof2021_LA_dev/ASVspoof2021.LA.cm.dev.trl.txt \
  --track LA \
  --viz_dir classification_analysis_LA_dev \
  --num_viz_samples 1000 \
  --batch_size 16 \
  --group_size 3 \
  --has_labels \
  --classification_viz
```

## Output Statistics

When using `--has_labels`, the system prints classification statistics:

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

## Visualization Details

### Individual Category Heatmaps

Each category gets its own heatmap showing attention patterns:

- **X-axis**: Sample number (showing 5 samples by default)
- **Y-axis**: Layer (0-23 for 24-layer model)
- **Color**: Attention weight (viridis colormap: purple→blue→green→yellow)
- **Grid**: White lines for clarity

### 4-Way Comparison Grid

A 2×2 grid layout:

```
┌─────────────────────────┬─────────────────────────┐
│ (a) Correct Bonafide    │ (b) Incorrect Bonafide  │
│     (True Accept)       │     (False Rejection)   │
├─────────────────────────┼─────────────────────────┤
│ (c) Correct Spoof       │ (d) Incorrect Spoof     │
│     (True Reject)       │     (False Acceptance)  │
└─────────────────────────┴─────────────────────────┘
```

## Interpretation Guide

### What to Look For

1. **Correct vs Incorrect Patterns**:

   - Do correctly classified samples show different attention patterns than incorrect ones?
   - Which layers are most important for correct classification?

2. **Bonafide vs Spoof Differences**:

   - Do bonafide and spoof samples attend to different layers?
   - Are there consistent patterns across categories?

3. **False Rejection Analysis**:

   - What makes bonafide samples get misclassified as spoof?
   - Do they show spoof-like attention patterns?

4. **False Acceptance Analysis**:
   - What makes spoof samples fool the model?
   - Do they mimic bonafide attention patterns?

### Research Questions

- **Layer Importance**: Which layers contribute most to correct classification?
- **Error Analysis**: What attention patterns lead to misclassification?
- **Model Behavior**: Does the model focus on different features for errors?
- **Attack Vulnerability**: Do false acceptances show specific patterns?

## Programmatic Usage

You can also use the visualization API directly in Python:

```python
from visualize_attention_evaluation import AttentionVisualizer
from torch.utils.data import DataLoader
from model_g3_heatmap import Model

# Load model and data (with labels)
model = Model(args, device)
model.load_state_dict(checkpoint)
model.eval()

# Create visualizer
visualizer = AttentionVisualizer(model, device, 'output_dir')

# Collect with labels
visualizer.collect_attention_weights(
    labeled_dataloader,
    num_samples=500,
    has_labels=True  # Critical for classification analysis
)

# Generate standard visualizations
visualizer.generate_all_visualizations()

# Generate classification visualizations
visualizer.generate_classification_visualizations(num_samples_per_category=5)
```

## Notes

- The system automatically detects if labels and predictions are different
- If using a dataset without true labels but setting `--has_labels`, predictions will be used as "labels"
- Requires sufficient samples in each category (min 1, recommends 5+)
- Categories with 0 samples will show "No samples" message in the grid

## Troubleshooting

**"Labels and predictions are identical"**:

- You're using an unlabeled dataset
- Use a training/dev set or evaluation set with ground truth

**"No samples found for category X"**:

- Perfect classification (no errors) or no samples of that class
- Increase `--num_viz_samples` to collect more data

**Memory issues**:

- Reduce `--batch_size`
- Reduce `--num_viz_samples`
- Use gradient checkpointing (model modification needed)

## Citation

If you use these visualizations in your research, please cite the original SLS paper and mention the enhanced visualization framework.
