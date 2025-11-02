#!/bin/bash
#
# Example script to generate attention visualizations for your best model
# Modify paths according to your setup
#

echo "🎨 Generating Attention Visualizations for G3 Model"
echo "=================================================="

# Configuration
CHECKPOINT="/root/autodl-tmp/SLSforASVspoof-2021-DF/models/best_model_eer_g3_viz_only.pth"
VIZ_BASE_DIR="attention_visualizations"

# ============================================================================
# Example 1: LA Track Evaluation
# ============================================================================
echo ""
echo "📊 Example 1: ASVspoof2021 LA Track"
echo "-----------------------------------"

python evaluate_with_attention_viz.py \
  --checkpoint "$CHECKPOINT" \
  --database_path /root/autodl-tmp/CLAD/Datasets/ASVspoof2021_LA_eval/ \
  --protocols_path /root/autodl-tmp/CLAD/Datasets/ASVspoof2021_LA_eval/ASVspoof2021.LA.cm.eval.trl.txt \
  --track LA \
  --viz_dir "${VIZ_BASE_DIR}/LA_track" \
  --num_viz_samples 100 \
  --batch_size 16 \
  --group_size 3

# ============================================================================
# Example 2: DF Track Evaluation
# ============================================================================
echo ""
echo "📊 Example 2: ASVspoof2021 DF Track"
echo "-----------------------------------"

python evaluate_with_attention_viz.py \
  --checkpoint "$CHECKPOINT" \
  --database_path /root/autodl-tmp/CLAD/Datasets/ASVspoof2021_DF_eval/ \
  --protocols_path /root/autodl-tmp/CLAD/Datasets/ASVspoof2021_DF_eval/trial_metadata.txt \
  --track DF \
  --viz_dir "${VIZ_BASE_DIR}/DF_track" \
  --num_viz_samples 100 \
  --batch_size 16 \
  --group_size 3

# ============================================================================
# Example 3: In-the-Wild Evaluation
# ============================================================================
echo ""
echo "📊 Example 3: In-the-Wild Dataset"
echo "-----------------------------------"

python evaluate_with_attention_viz.py \
  --checkpoint "$CHECKPOINT" \
  --database_path /root/autodl-tmp/CLAD/Datasets/release_in_the_wild/ \
  --protocols_path /root/autodl-tmp/CLAD/Datasets/release_in_the_wild/filenames.txt \
  --track In-the-Wild \
  --viz_dir "${VIZ_BASE_DIR}/in_the_wild" \
  --num_viz_samples 100 \
  --batch_size 16 \
  --group_size 3

echo ""
echo "=================================================="
echo "✅ All visualizations completed!"
echo "📁 Results saved in: ${VIZ_BASE_DIR}/"
echo ""
echo "Generated files for each track:"
echo "  - layer_weight_heatmap_spoof.png"
echo "  - layer_weight_heatmap_bonafide.png"
echo "  - layer_weight_comparison.png (Publication-ready!)"
echo "  - temporal_attention_heatmap_*.png"
echo "  - intra_group_attention_*.png"
echo "  - inter_group_attention_*.png"
echo "=================================================="
