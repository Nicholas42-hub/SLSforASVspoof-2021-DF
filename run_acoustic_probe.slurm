#!/bin/bash
#SBATCH --job-name=acoustic_probe
#SBATCH --output=/data/projects/punim2637/nnliang/SLSforASVspoof-2021-DF/logs/acoustic_probe_%j.out
#SBATCH --error=/data/projects/punim2637/nnliang/SLSforASVspoof-2021-DF/logs/acoustic_probe_%j.err
#SBATCH --partition=gpu-a100-short
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-02:00:00

# Load required modules
module purge
module load GCCcore/11.3.0
module load Python/3.10.4
module load CUDA/11.8.0

# Set working directory
cd /data/projects/punim2637/nnliang/SLSforASVspoof-2021-DF

# Activate virtual environment
source SLS_venv/bin/activate

# Print environment information
echo "=========================================="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="

# Add fairseq to Python path
export PYTHONPATH="/data/projects/punim2637/nnliang/SLSforASVspoof-2021-DF/fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1:$PYTHONPATH"

# Run acoustic probe analysis
echo "Starting acoustic probe analysis..."
python probe_acoustic_asvspoof.py

EXIT_CODE=$?

echo "=========================================="
echo "Analysis finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s
