#!/bin/bash
#SBATCH --job-name=temporal_compare
#SBATCH --partition=gpu-a100-short
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/temporal_comparison_%j.out
#SBATCH --error=logs/temporal_comparison_%j.err

# Load modules
module purge
module load GCCcore/11.3.0
module load Python/3.10.4
module load CUDA/11.8.0

# Activate environment
source SLS_venv/bin/activate

# Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/data/projects/punim2637/nnliang/SLSforASVspoof-2021-DF/fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1"

# Print environment info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "Python: $(which python)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo ""

# Model paths
MODEL1="/data/projects/punim2637/nnliang/SLSforASVspoof-2021-DF/models/topk_sae_LA_e40_bs14_lr1e-06_saeW0.1_dict4096_k128_k128_sparse_4096dim/best_checkpoint_eer_k128_sparse_4096dim.pth"
MODEL2="/data/projects/punim2637/nnliang/SLSforASVspoof-2021-DF/models/topk_sae_window_w8_LA_e40_bs14_lr1e-06_saeW0.1_dict4096_k128_window_topk_w8/best_checkpoint_eer_window_topk_w8.pth"

# Dataset paths
DATABASE_PATH="/data/projects/punim2637/nnliang/Datasets/LA"
PROTOCOLS_PATH="/data/projects/punim2637/nnliang/Datasets/LA/ASVspoof2019_LA_cm_protocols"

# Run comparison
python compare_temporal_models.py \
    --model1_path "$MODEL1" \
    --model1_name "Per-timestep TopK (w=1)" \
    --model1_window 1 \
    --model2_path "$MODEL2" \
    --model2_name "Window TopK (w=8)" \
    --model2_window 8 \
    --database_path "$DATABASE_PATH" \
    --protocols_path "$PROTOCOLS_PATH" \
    --output_dir temporal_comparison \
    --num_samples 100 \
    --batch_size 8 \
    --device cuda

echo ""
echo "End time: $(date)"
echo "Comparison complete!"
