#!/bin/bash
#SBATCH --job-name=cpc_limits
#SBATCH --partition=gpu-a100-short
#SBATCH --time=1:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/cpc_limitations_%j.out
#SBATCH --error=logs/cpc_limitations_%j.err

# Load modules
module purge
module load GCCcore/11.3.0
module load Python/3.10.4
module load CUDA/11.8.0

# Activate environment
source SLS_venv/bin/activate

# Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/data/projects/punim2637/nnliang/SLSforASVspoof-2021-DF/fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1"

echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo ""

# Model path
MODEL_PATH="/data/projects/punim2637/nnliang/SLSforASVspoof-2021-DF/models/cpc_window_w8_LA_e40_bs14_lr1e-06_saeW0.1_cpcW0.5_dict4096_k128_cpc/best_checkpoint_eer_cpc.pth"

# Run limitation analysis
python analyze_cpc_limitations.py \
    --model_path "$MODEL_PATH" \
    --database_path "/data/projects/punim2637/nnliang/Datasets/LA" \
    --protocols_path "/data/projects/punim2637/nnliang/Datasets/LA/ASVspoof2019_LA_cm_protocols" \
    --output_dir cpc_limitations_analysis \
    --num_samples 100 \
    --batch_size 8 \
    --device cuda

echo ""
echo "End time: $(date)"
echo "Analysis complete!"
