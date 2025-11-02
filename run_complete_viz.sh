#!/bin/bash
# 运行完整的分类可视化（包括正确和错误分类）
# Run complete classification visualization (both correct and incorrect)

CUDA_VISIBLE_DEVICES=0 python evaluate_with_attention_viz.py \
  --checkpoint /root/autodl-tmp/SLSforASVspoof-2021-DF/models/g3_heatmap_LA_CCE_100_16_1e-06_group3_contrastiveFalse_g3_viz_only/best_model_eer_g3_viz_only.pth \
  --database_path /root/autodl-tmp/CLAD/Datasets/ASVspoof2021_LA_eval/ \
  --protocols_path /root/autodl-tmp/CLAD/Datasets/ASVspoof2021_LA_eval/trial_metadata.txt \
  --track LA \
  --viz_dir attention_viz_LA_complete \
  --num_viz_samples 500 \
  --batch_size 16 \
  --group_size 3 \
  --has_labels \
  --classification_viz \
  --incorrect_only_viz
