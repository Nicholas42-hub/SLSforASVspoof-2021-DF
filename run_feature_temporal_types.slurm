#!/bin/bash
#SBATCH --job-name=feature_temporal_types
#SBATCH --partition=gpu-a100-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/feature_temporal_types_%j.out
#SBATCH --error=logs/feature_temporal_types_%j.err

# Load modules
module purge
echo "Modules purged"
module load GCCcore/11.3.0
echo "GCCcore loaded"
module load Python/3.10.4
echo "Python loaded"
module load CUDA/11.8.0
echo "CUDA loaded"

# Activate environment
cd /data/projects/punim2637/nnliang/SLSforASVspoof-2021-DF
source SLS_venv/bin/activate
echo "Environment activated"

# Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/data/projects/punim2637/nnliang/SLSforASVspoof-2021-DF/fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1"
export PYTHONUNBUFFERED=1

echo "=========================================="
echo "Testing Hypothesis: Discriminative features are TRANSIENT"
echo "If TRUE: Explains why overlapping windows fail (smooth discriminative features)"
echo "If FALSE: Need other explanation"
echo "=========================================="

echo "Running Python script..."
python3 test_feature_temporal_quick.py
echo "Python script finished"

echo "=========================================="
echo "Analysis complete! Check feature_temporal_types_analysis.json"
echo "=========================================="
