#!/bin/bash
# Example script for generating classification-based attention visualizations
# This assumes you have a dataset with true labels (e.g., dev set or labeled eval set)

# Configuration
CHECKPOINT="/root/autodl-tmp/SLSforASVspoof-2021-DF/models/g3_heatmap_LA_CCE_100_16_1e-06_group3_contrastiveFalse_g3_viz_only/best_checkpoint_eer_g3_viz_only.pth"
DATABASE_PATH="/root/autodl-tmp/CLAD/Datasets/ASVspoof2021_LA_dev/"  # Use dev set (has labels)
PROTOCOLS_PATH="/root/autodl-tmp/CLAD/Datasets/ASVspoof2021_LA_dev/ASVspoof2021.LA.cm.dev.trl.txt"
OUTPUT_DIR="classification_viz_LA_$(date +%Y%m%d)"

echo "🎯 Generating Classification-Based Attention Visualizations"
echo "============================================================"
echo "Checkpoint: $CHECKPOINT"
echo "Data: $DATABASE_PATH"
echo "Output: $OUTPUT_DIR"
echo "============================================================"

CUDA_VISIBLE_DEVICES=0 python evaluate_with_attention_viz.py \
  --checkpoint "$CHECKPOINT" \
  --database_path "$DATABASE_PATH" \
  --protocols_path "$PROTOCOLS_PATH" \
  --track LA \
  --viz_dir "$OUTPUT_DIR" \
  --num_viz_samples 1000 \
  --batch_size 16 \
  --group_size 3 \
  --has_labels \
  --classification_viz

echo ""
echo "✅ Done! Check the output directory: $OUTPUT_DIR"
echo ""
echo "Generated visualizations:"
echo "  Standard:"
echo "    - layer_weight_heatmap_spoof.png"
echo "    - layer_weight_heatmap_bonafide.png"
echo "    - layer_weight_comparison.png"
echo "    - temporal_attention_heatmap_*.png"
echo "    - intra_group_attention_*.png"
echo "    - inter_group_attention_*.png"
echo ""
echo "  Classification-based:"
echo "    - layer_weight_correct_bonafide.png"
echo "    - layer_weight_incorrect_bonafide.png (False Rejections)"
echo "    - layer_weight_correct_spoof.png"
echo "    - layer_weight_incorrect_spoof.png (False Acceptances)"
echo "    - classification_comparison_4way.png"
