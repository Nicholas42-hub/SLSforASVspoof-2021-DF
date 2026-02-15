#!/bin/bash
#SBATCH --job-name=test_lim
#SBATCH --partition=gpu-a100-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=0:10:00
#SBATCH --output=logs/test_limitations_%j.out
#SBATCH --error=logs/test_limitations_%j.err

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

source SLS_venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/data/projects/punim2637/nnliang/SLSforASVspoof-2021-DF/fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1:$PYTHONPATH

python analyze_window_limitations.py \
    --model_path models/topk_sae_window_w8_LA_e40_bs14_lr1e-06_saeW0.1_dict4096_k128_window_topk_w8/best_checkpoint_eer_window_topk_w8.pth \
    --database_path ../Datasets/LA \
    --protocols_path ../Datasets/LA/ASVspoof2019_LA_cm_protocols \
    --output_dir window_limitations_test \
    --num_samples 1 \
    --device cuda

echo "End time: $(date)"
