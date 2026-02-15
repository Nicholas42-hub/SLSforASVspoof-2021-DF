#!/bin/bash
#SBATCH --job-name=quick_test
#SBATCH --partition=gpu-a100-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=logs/quick_test_%j.out
#SBATCH --error=logs/quick_test_%j.err

# Load modules
module purge
module load GCCcore/11.3.0
module load Python/3.10.4
module load CUDA/11.8.0

# Activate environment
cd /data/projects/punim2637/nnliang/SLSforASVspoof-2021-DF
source SLS_venv/bin/activate

# Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/data/projects/punim2637/nnliang/SLSforASVspoof-2021-DF/fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1"

echo "Running quick 10-sample test..."
python3 test_feature_temporal_quick.py 2>&1
EXIT_CODE=$?
echo "Exit code: $EXIT_CODE"
echo "Test complete!"
