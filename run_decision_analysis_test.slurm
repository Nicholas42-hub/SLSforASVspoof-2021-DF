#!/bin/bash
#SBATCH --job-name=decision_test
#SBATCH --partition=gpu-a100
#SBATCH --time=1:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/decision_test_%j.out
#SBATCH --error=logs/decision_test_%j.err

echo "Job started at: $(date)"

# Activate virtual environment
source /data/projects/punim2637/nnliang/SLSforASVspoof-2021-DF/SLS_venv/bin/activate

# Add fairseq to Python path
export PYTHONPATH="/data/projects/punim2637/nnliang/SLSforASVspoof-2021-DF/fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1:$PYTHONPATH"

# Run analysis with 500 samples (quick test)
python analyze_decision_relevance.py \
    --model_path models/topk_sae_window_w8_LA_e40_bs14_lr1e-06_saeW0.1_dict4096_k128_window_topk_w8/best_checkpoint_eer_window_topk_w8.pth \
    --dataset 2021_LA \
    --num_samples 500 \
    --output_dir decision_analysis_test_500 \
    --batch_size 8

echo "Job completed at: $(date)"
