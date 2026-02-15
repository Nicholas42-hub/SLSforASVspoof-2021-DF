#!/bin/bash
#SBATCH --job-name=eval_overlap
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/data/projects/punim2637/nnliang/SLSforASVspoof-2021-DF/slurm-%j.out

# Print job info
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"
echo "Running on node: $(hostname)"
echo "========================================="

# Navigate to working directory
cd /data/projects/punim2637/nnliang/SLSforASVspoof-2021-DF

# Activate virtual environment
source /data/projects/punim2637/nnliang/SLSforASVspoof-2021-DF/SLS_venv/bin/activate

# Set fairseq path
export PYTHONPATH="/data/projects/punim2637/nnliang/SLSforASVspoof-2021-DF/fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1:$PYTHONPATH"

# Print environment info
echo ""
echo "Python: $(which python)"
echo "CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

# Run evaluation
echo "========================================="
echo "EVALUATING OVERLAPPING WINDOWS MODEL"
echo "========================================="
echo ""

python eval_overlap_clean.py \
    --model_path models/topk_sae_window_w8_LA_e40_bs14_lr1e-06_saeW0.1_dict4096_k128_window_topk_w8/best_checkpoint_eer_window_topk_w8.pth \
    --database_path /data/projects/punim2637/nnliang/Datasets/ASVspoof2021_LA_eval \
    --batch_size 16 \
    --num_samples 5000 \
    --output overlap_eval_5k_results.json

echo ""
echo "========================================="
echo "Job completed at: $(date)"
echo "========================================="
