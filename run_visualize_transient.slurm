#!/bin/bash
#SBATCH --job-name=viz_transient
#SBATCH --time=0:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-a100-short
#SBATCH --output=logs/visualize_transient_%j.out
#SBATCH --error=logs/visualize_transient_%j.err

# 可视化瞬态特征

echo "开始时间: $(date)"
echo "="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "="

# 激活环境
source ~/.bashrc
conda activate ssl_anti_spoofing

# 设置路径
MODEL_PATH="/data/projects/punim2637/nnliang/SLSforASVspoof-2021-DF/models/topk_sae_window8_k128/best_model.pth"
OUTPUT_DIR="transient_analysis_visualization"

# 运行可视化
echo "运行瞬态特征可视化..."
python visualize_transient_features.py \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_samples 20

echo ""
echo "完成时间: $(date)"
