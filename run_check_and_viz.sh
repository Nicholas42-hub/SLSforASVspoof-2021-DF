#!/bin/bash
# Quick check for dataset balance before running visualization

echo "🔍 Checking dataset balance..."
python check_dataset_balance.py \
  --protocols_path /root/autodl-tmp/CLAD/Datasets/ASVspoof2021_LA_eval/ASVspoof2021.LA.cm.eval.trl.txt

echo ""
echo "📊 Now run the visualization with shuffled dataloader..."
echo ""
echo "CUDA_VISIBLE_DEVICES=0 python evaluate_with_attention_viz.py \\"
echo "  --checkpoint /root/autodl-tmp/SLSforASVspoof-2021-DF/models/.../best_model_eer_g3_viz_only.pth \\"
echo "  --database_path /root/autodl-tmp/CLAD/Datasets/ASVspoof2021_LA_eval/ \\"
echo "  --protocols_path /root/autodl-tmp/CLAD/Datasets/ASVspoof2021_LA_eval/ASVspoof2021.LA.cm.eval.trl.txt \\"
echo "  --track LA \\"
echo "  --viz_dir attention_viz_LA_classification \\"
echo "  --num_viz_samples 500 \\"
echo "  --batch_size 16 \\"
echo "  --group_size 3 \\"
echo "  --has_labels \\"
echo "  --classification_viz"
