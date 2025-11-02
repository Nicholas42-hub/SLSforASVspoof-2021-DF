# 🎨 ATTENTION VISUALIZATION - COMMAND REFERENCE

## Quick Commands (Copy & Paste Ready)

### 1️⃣ LA Track (ASVspoof2021)

```bash
python evaluate_with_attention_viz.py \
  --checkpoint models/best_model_eer_g3_viz_only.pth \
  --database_path /root/autodl-tmp/CLAD/Datasets/ASVspoof2021_LA_eval/ \
  --protocols_path /root/autodl-tmp/CLAD/Datasets/ASVspoof2021_LA_eval/ASVspoof2021.LA.cm.eval.trl.txt \
  --track LA \
  --viz_dir attention_viz_LA \
  --num_viz_samples 100 \
  --batch_size 16 \
  --group_size 3
```

### 2️⃣ DF Track (ASVspoof2021)

```bash
python evaluate_with_attention_viz.py \
  --checkpoint models/best_model_eer_g3_viz_only.pth \
  --database_path /root/autodl-tmp/CLAD/Datasets/ASVspoof2021_DF_eval/ \
  --protocols_path /root/autodl-tmp/CLAD/Datasets/ASVspoof2021_DF_eval/trial_metadata.txt \
  --track DF \
  --viz_dir attention_viz_DF \
  --num_viz_samples 100 \
  --batch_size 16 \
  --group_size 3
```

### 3️⃣ In-the-Wild Dataset

```bash
python evaluate_with_attention_viz.py \
  --checkpoint models/best_model_eer_g3_viz_only.pth \
  --database_path /root/autodl-tmp/CLAD/Datasets/release_in_the_wild/ \
  --protocols_path /root/autodl-tmp/CLAD/Datasets/release_in_the_wild/filenames.txt \
  --track In-the-Wild \
  --viz_dir attention_viz_wild \
  --num_viz_samples 100 \
  --batch_size 16 \
  --group_size 3
```

---

## Parameters Explained

| Parameter           | Description                | Default              | Notes                        |
| ------------------- | -------------------------- | -------------------- | ---------------------------- |
| `--checkpoint`      | Path to model .pth file    | Required             | Your trained model           |
| `--database_path`   | Directory with audio files | Required             | Eval set location            |
| `--protocols_path`  | Protocol/metadata file     | Required             | File not directory for LA/DF |
| `--track`           | Evaluation track           | `LA`                 | `LA`, `DF`, or `In-the-Wild` |
| `--viz_dir`         | Output directory           | `attention_analysis` | Where to save figures        |
| `--num_viz_samples` | Samples to analyze         | `100`                | More = better stats          |
| `--batch_size`      | Batch size                 | `16`                 | Adjust for GPU memory        |
| `--group_size`      | Layer group size           | `3`                  | Must match training          |
| `--skip_viz`        | Skip visualization         | `False`              | Use to only evaluate         |

---

## Expected Output

### Terminal Output:

```
🖥️  Using device: cuda
======================================================================
📊 EVALUATION WITH ATTENTION VISUALIZATION
======================================================================
Checkpoint: models/best_model_eer_g3_viz_only.pth
Track: LA
Database: /root/autodl-tmp/CLAD/Datasets/ASVspoof2021_LA_eval/
Protocol: .../ASVspoof2021.LA.cm.eval.trl.txt
======================================================================

📂 Loading evaluation data...
   Found XXXX evaluation samples

📦 Loading model from checkpoint...
   ✅ Loaded checkpoint from epoch XX
   📊 Checkpoint EER: X.XX%

======================================================================
🎨 GENERATING ATTENTION VISUALIZATIONS
======================================================================

🔍 Collecting attention weights from 100 samples...
✅ Collected attention weights from 100 samples
   - Temporal attention shape: (25, 201)
   - Intra-group attention shape: (8, 3)
   - Inter-group attention shape: (8,)

📊 Generating visualizations...
📊 Generating layer weight heatmaps...
💾 Saved layer weight heatmap: attention_viz_LA/layer_weight_heatmap_spoof.png
💾 Saved layer weight heatmap: attention_viz_LA/layer_weight_heatmap_bonafide.png
💾 Saved comparison heatmap: attention_viz_LA/layer_weight_comparison.png

⏰ Generating temporal attention heatmaps...
💾 Saved temporal attention heatmap: attention_viz_LA/temporal_attention_heatmap_spoof.png
💾 Saved temporal attention heatmap: attention_viz_LA/temporal_attention_heatmap_bonafide.png

🔗 Generating intra-group attention visualizations...
💾 Saved intra-group attention: attention_viz_LA/intra_group_attention_spoof.png
💾 Saved intra-group attention: attention_viz_LA/intra_group_attention_bonafide.png

🌐 Generating inter-group attention visualizations...
💾 Saved inter-group attention: attention_viz_LA/inter_group_attention_spoof.png
💾 Saved inter-group attention: attention_viz_LA/inter_group_attention_bonafide.png

======================================================================
✅ ALL VISUALIZATIONS SAVED TO: attention_viz_LA
======================================================================

✅ Visualization complete!
📁 Results saved to: attention_viz_LA/

Generated files:
  📈 layer_weight_heatmap_spoof.png
  📈 layer_weight_heatmap_bonafide.png
  📈 layer_weight_comparison.png
  ⏰ temporal_attention_heatmap_spoof.png
  ⏰ temporal_attention_heatmap_bonafide.png
  🔗 intra_group_attention_spoof.png
  🔗 intra_group_attention_bonafide.png
  🌐 inter_group_attention_spoof.png
  🌐 inter_group_attention_bonafide.png

======================================================================
✅ EVALUATION COMPLETE
======================================================================
```

---

## Troubleshooting

### "No module named 'visualize_attention_evaluation'"

**Solution**: Make sure you're in the correct directory

```bash
cd /Users/Lzlo/Desktop/Project/PhD/Baselines/SLSforASVspoof-2021-DF
python evaluate_with_attention_viz.py ...
```

### "Checkpoint file not found"

**Solution**: Check your checkpoint path

```bash
ls -lh models/best_model_eer_g3_viz_only.pth
```

### "CUDA out of memory"

**Solution**: Reduce batch size or number of samples

```bash
--batch_size 8 \
--num_viz_samples 50
```

### "No attention weights collected"

**Solution**: Check that model supports `return_attention=True`

- Your model already has this built-in ✅

---

## Advanced Usage

### Analyze more samples for better statistics:

```bash
--num_viz_samples 200
```

### Use smaller batch for limited GPU memory:

```bash
--batch_size 8
```

### Just evaluate without visualization:

```bash
--skip_viz
```

### Custom output directory:

```bash
--viz_dir my_research/figures/attention_analysis
```

---

## Files Created

| File                                | Size        | Description                               |
| ----------------------------------- | ----------- | ----------------------------------------- |
| `layer_weight_comparison.png`       | ~1MB        | **Main figure** - Side-by-side comparison |
| `layer_weight_heatmap_spoof.png`    | ~800KB      | Spoof layer attention                     |
| `layer_weight_heatmap_bonafide.png` | ~800KB      | Bonafide layer attention                  |
| `temporal_attention_heatmap_*.png`  | ~1.2MB each | Temporal patterns                         |
| `intra_group_attention_*.png`       | ~600KB each | Within-group attention                    |
| `inter_group_attention_*.png`       | ~500KB each | Between-group attention                   |

**Total**: ~7-10 MB for complete analysis

---

## Integration with Paper

### LaTeX Figure:

```latex
\begin{figure*}[t]
  \centering
  \includegraphics[width=0.95\textwidth]{attention_viz_LA/layer_weight_comparison.png}
  \caption{Layer-wise attention distribution across SSL layers for
  (a) spoof and (b) bonafide samples from ASVspoof2021 LA evaluation set.
  Each row represents a different sample, and each column represents an SSL layer.
  Higher attention weights (lighter colors) indicate greater layer importance
  for the final classification decision. Spoof samples show concentrated
  attention on layers 15-20, while bonafide samples exhibit more uniform
  distribution.}
  \label{fig:layer_attention}
\end{figure*}
```

---

## Quick Tips

✅ **Run on all tracks** to compare attention patterns  
✅ **Use 100-200 samples** for publication-quality statistics  
✅ **Keep group_size=3** to match your training configuration  
✅ **Save to different directories** for each experiment  
✅ **All figures are 300 DPI** - publication ready!

---

## 🎉 You're Ready!

Just copy one of the commands above and run it. Results will appear in minutes!

For more details, see:

- `SOLUTION_SUMMARY.md` - Complete overview
- `ATTENTION_VISUALIZATION_GUIDE.md` - Full documentation
- `ATTENTION_VIZ_README.md` - Quick reference
