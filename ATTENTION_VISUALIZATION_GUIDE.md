# Attention Visualization Guide

This guide explains how to generate attention heatmaps similar to research paper figures during model evaluation.

## 📊 What Visualizations Are Generated?

### 1. **Layer Weight Heatmap** (Similar to your Figure 2)

- Shows attention distribution across different SSL layers
- Separate heatmaps for spoof and bonafide samples
- Each row = different sample/model run
- Each column = SSL layer
- **File**: `layer_weight_heatmap_*.png`, `layer_weight_comparison.png`

### 2. **Temporal Attention Visualization** (intra_attn)

- Shows how attention is distributed across time frames
- Displays layer-wise temporal patterns
- Helps understand which parts of audio are most important
- **File**: `temporal_attention_heatmap_*.png`

### 3. **Intra-Group Attention**

- Visualizes attention within layer groups
- Shows how layers within a group interact
- **File**: `intra_group_attention_*.png`

### 4. **Inter-Group Attention**

- Visualizes attention between layer groups
- Shows which groups are most important for classification
- **File**: `inter_group_attention_*.png`

---

## 🚀 Quick Start

### Method 1: Standalone Evaluation Script (Recommended)

```bash
python evaluate_with_attention_viz.py \
  --checkpoint models/best_model_eer_g3_viz_only.pth \
  --database_path /root/autodl-tmp/CLAD/Datasets/ASVspoof2021_LA_eval/ \
  --protocols_path /root/autodl-tmp/CLAD/Datasets/ASVspoof2021_LA_eval/ASVspoof2021.LA.cm.eval.trl.txt \
  --track LA \
  --viz_dir attention_analysis_LA \
  --num_viz_samples 100 \
  --batch_size 16 \
  --group_size 3
```

### Method 2: Using Main Script with Visualization

```bash
python main.py \
  --track LA \
  --is_eval \
  --eval \
  --model_path models/best_model_eer_g3_viz_only.pth \
  --protocols_path /path/to/protocol.txt \
  --database_path /path/to/eval/data/ \
  --model_type g3_heatmap \
  --batch_size 16 \
  --group_size 3 \
  --visualize_attention \
  --viz_samples 100
```

### Method 3: Python API

```python
from visualize_attention_evaluation import AttentionVisualizer
from model_g3_heatmap import Model
import torch

# Load your model
model = Model(args, device)
checkpoint = torch.load('best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Create visualizer
visualizer = AttentionVisualizer(model, device, save_dir='my_viz')

# Collect attention weights
visualizer.collect_attention_weights(eval_loader, num_samples=100)

# Generate all visualizations
visualizer.generate_all_visualizations()

# Or generate specific visualizations:
visualizer.plot_layer_weight_heatmap(num_runs=5, class_label=0)  # Spoof only
visualizer.plot_temporal_attention_heatmap(num_samples=20, class_label=1)  # Bonafide only
visualizer.plot_intra_group_attention(class_label=None)  # All samples
visualizer.plot_inter_group_attention()
```

---

## 📁 Output Files

After running, you'll find these files in your visualization directory:

```
attention_analysis_LA/
├── layer_weight_heatmap_spoof.png          # Layer attention for spoof samples
├── layer_weight_heatmap_bonafide.png       # Layer attention for bonafide samples
├── layer_weight_comparison.png             # Side-by-side comparison (like Figure 2)
├── temporal_attention_heatmap_spoof.png    # Temporal patterns for spoof
├── temporal_attention_heatmap_bonafide.png # Temporal patterns for bonafide
├── intra_group_attention_spoof.png         # Intra-group weights for spoof
├── intra_group_attention_bonafide.png      # Intra-group weights for bonafide
├── inter_group_attention_spoof.png         # Inter-group weights for spoof
└── inter_group_attention_bonafide.png      # Inter-group weights for bonafide
```

---

## 🔍 Understanding the Visualizations

### Layer Weight Heatmap

- **Darker colors** = higher attention weight
- **Lighter colors** = lower attention weight
- Different rows represent different samples
- Look for patterns: Do certain layers consistently get high attention?

### Temporal Attention (intra_attn)

- **X-axis**: Time frames in the audio
- **Y-axis**: SSL layers
- Shows which temporal regions each layer focuses on
- Useful for understanding: "Does the model focus on beginning/middle/end of audio?"

### Intra-Group Attention

- **X-axis**: Layers within a group
- **Y-axis**: Group number
- Shows layer importance within each group
- Helps understand hierarchical attention structure

### Inter-Group Attention

- **Bar height**: Importance of each group
- Shows which layer groups contribute most to final decision
- Higher bars = more important groups

---

## ⚙️ Advanced Usage

### Analyze Specific Samples

```python
# Collect specific number of samples
visualizer.collect_attention_weights(eval_loader, num_samples=50)

# Generate visualizations for spoof only
visualizer.plot_layer_weight_heatmap(num_runs=10, class_label=0)
visualizer.plot_temporal_attention_heatmap(num_samples=30, class_label=0)
```

### Custom Styling

Modify `visualize_attention_evaluation.py` to change:

- Color maps (e.g., `cmap='viridis'` → `cmap='plasma'`)
- Figure sizes
- Font sizes
- DPI for higher/lower resolution

### Extract Attention Weights

```python
# After collecting attention weights
temporal_attns = visualizer.temporal_attns  # List of (L, T) arrays
intra_attns = visualizer.intra_attns        # List of (num_groups, group_size) arrays
inter_attns = visualizer.inter_attns        # List of (num_groups,) arrays
labels = visualizer.labels                   # True labels
predictions = visualizer.predictions         # Predicted labels

# Analyze specific samples
sample_idx = 0
sample_temporal = temporal_attns[sample_idx]  # Shape: (25, 201) for 25 layers, 201 time frames
sample_label = labels[sample_idx]
```

---

## 🎯 Example Analysis Workflow

### 1. Generate visualizations for your best model

```bash
python evaluate_with_attention_viz.py \
  --checkpoint models/best_model_eer_g3_viz_only.pth \
  --database_path /path/to/LA_eval/ \
  --protocols_path /path/to/protocol.txt \
  --track LA \
  --viz_dir results/attention_LA \
  --num_viz_samples 200
```

### 2. Compare different tracks

```bash
# LA track
python evaluate_with_attention_viz.py --checkpoint best_model.pth --track LA --viz_dir viz_LA

# DF track
python evaluate_with_attention_viz.py --checkpoint best_model.pth --track DF --viz_dir viz_DF

# In-the-Wild
python evaluate_with_attention_viz.py --checkpoint best_model.pth --track In-the-Wild --viz_dir viz_wild
```

### 3. Analyze differences between spoof and bonafide

- Compare `layer_weight_heatmap_spoof.png` vs `layer_weight_heatmap_bonafide.png`
- Look for systematic differences in layer attention patterns
- Check if temporal attention differs (beginning vs end focus)

---

## 💡 Tips

1. **Sample Size**: Use 100-200 samples for stable statistics while keeping computation reasonable

2. **Class Balance**: If your eval set is imbalanced, visualizations might be dominated by majority class

3. **Memory**: Large visualizations use GPU memory. Reduce `num_viz_samples` if you encounter OOM errors

4. **Interpretation**:

   - High attention on early layers → model relies on low-level features
   - High attention on late layers → model uses high-level semantic features
   - Temporal patterns can reveal artifacts (e.g., always attending to audio start/end)

5. **Paper Figures**: The layer weight comparison plot (`layer_weight_comparison.png`) is publication-ready

---

## 🐛 Troubleshooting

**Problem**: "No attention weights collected"

- **Solution**: Make sure model's `return_attention=True` is set in forward pass

**Problem**: Visualizations look weird/uniform

- **Solution**: Check if model is properly loaded and in eval mode

**Problem**: Out of memory

- **Solution**: Reduce `--num_viz_samples` or `--batch_size`

**Problem**: Want different color schemes

- **Solution**: Edit `visualize_attention_evaluation.py` and change `cmap` parameter

---

## 📚 Files Reference

- `visualize_attention_evaluation.py` - Main visualization code
- `evaluate_with_attention_viz.py` - Standalone evaluation script
- `model_g3_heatmap.py` - Model with attention weight storage
- This guide - `ATTENTION_VISUALIZATION_GUIDE.md`

---

## 🎨 Customization Examples

### Change Heatmap Colors

```python
# In visualize_attention_evaluation.py, find the sns.heatmap() calls and change:
cmap='viridis'  # → Blue to yellow
cmap='YlOrRd'   # → Yellow to red
cmap='coolwarm' # → Blue to red
cmap='plasma'   # → Purple to yellow
```

### Adjust Figure Sizes

```python
# Change figsize parameter:
fig, ax = plt.subplots(figsize=(14, 6))  # Width, Height in inches
```

### Save Different Format

```python
# Change file extension:
plt.savefig(save_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
```

---

Happy visualizing! 🎨📊
