#!/bin/bash
#SBATCH --job-name=sae_analysis
#SBATCH --partition=gpu-a100-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=logs/sae_analysis_%j.out
#SBATCH --error=logs/sae_analysis_%j.err

# Load modules
module purge
module load fosscuda/2020b
module load pytorch/1.13.1-py39-cuda11.7.0

# Activate virtual environment
source SLS_venv/bin/activate

# Run analysis (500 samples = 250 bonafide + 250 spoof for balanced analysis)
python analyze_sae_neurons.py \
    --model_path models/topk_sae_LA_e40_bs14_lr1e-06_saeW0.1_dict4096_k128_topk_sae_sparse/best_checkpoint_eer_topk_sae_sparse.pth \
    --database_path /data/projects/punim2637/nnliang/Datasets/LA \
    --protocols_path keys/ASVspoof2019.LA.cm.eval.trl.txt \
    --num_samples 500 \
    --output_dir analysis_results

echo "Analysis complete!"
