#!/bin/bash
#SBATCH --job-name=eval_overlap_eer
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=slurm-%j.out

# Print job info
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Start Time: $(date)"
echo "Running on node: $(hostname)"
echo "========================================"

# Activate environment
source /data/gpfs/projects/punim2637/nnliang/SLSforASVspoof-2021-DF/SLS_venv/bin/activate

# Print Python info
echo "Python: $(which python)"
echo "CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())')"

echo ""
echo "========================================"
echo "EVALUATING OVERLAPPING WINDOWS EER"
echo "========================================"

# Run evaluation
cd /data/projects/punim2637/nnliang/SLSforASVspoof-2021-DF
python eval_overlap_eer.py

echo ""
echo "========================================"
echo "Job completed at: $(date)"
echo "========================================"
