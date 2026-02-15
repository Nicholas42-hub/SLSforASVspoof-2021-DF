#!/bin/bash
#SBATCH --job-name=rerun_limitations
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-a100-short
#SBATCH --output=logs/rerun_limitations_%j.out
#SBATCH --error=logs/rerun_limitations_%j.err

# 使用改进的瞬态分析重新运行 window limitations analysis

echo "开始时间: $(date)"
echo "="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "="

# 加载模块
module purge
module load GCCcore/11.3.0
module load Python/3.10.4
module load CUDA/11.8.0

# 激活虚拟环境
source SLS_venv/bin/activate

# 设置路径
MODEL_PATH="/data/projects/punim2637/nnliang/SLSforASVspoof-2021-DF/models/topk_sae_window_w8_LA_e40_bs14_lr1e-06_saeW0.1_dict4096_k128_window_topk_w8/best_checkpoint_eer_window_topk_w8.pth"
OUTPUT_DIR="window_limitations_analysis_v2"

echo "改进内容:"
echo "  ✓ 使用 Logistic Regression + AUC 评估判别能力"
echo "  ✓ 多维特征统计 (mean, max, freq, var)"
echo "  ✓ 更鲁棒的数据验证"
echo "  ✓ 详细的诊断信息"
echo ""

# 运行分析
echo "运行改进的 window limitations 分析..."
python analyze_window_limitations.py \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_samples 100

echo ""
echo "结果保存到: $OUTPUT_DIR/limitations_analysis.json"
echo ""
echo "完成时间: $(date)"
