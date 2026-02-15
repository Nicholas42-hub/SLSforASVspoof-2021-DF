#!/bin/bash
#SBATCH --job-name=global_cue
#SBATCH --partition=gpu-a100-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/global_cue_consistency_%j.out
#SBATCH --error=logs/global_cue_consistency_%j.err

# Load modules
module purge
module load GCCcore/11.3.0
module load Python/3.10.4
module load CUDA/11.8.0

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Setup
cd /data/projects/punim2637/nnliang/SLSforASVspoof-2021-DF
source SLS_venv/bin/activate

# Add fairseq to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/data/gpfs/projects/punim2637/nnliang/SLSforASVspoof-2021-DF/fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1"
export PYTHONUNBUFFERED=1

# Create output directory
mkdir -p logs
mkdir -p global_cue_consistency_500

python analyze_global_cue_consistency.py \
    --model_path models/topk_sae_window_w8_LA_e40_bs14_lr1e-06_saeW0.1_dict4096_k128_window_topk_w8/best_checkpoint_eer_window_topk_w8.pth \
    --num_samples 500 \
    --dataset 2021_LA \
    --output_dir global_cue_consistency_500 \
    --batch_size 1

echo "Global Cue Consistency Analysis Complete!"
echo "Results saved to global_cue_consistency_500/"
