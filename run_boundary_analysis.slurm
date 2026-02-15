#!/bin/bash
#SBATCH --job-name=boundary_sem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-a100
#SBATCH --time=3:00:00
#SBATCH --mem=32G

source /data/projects/punim2637/nnliang/SLSforASVspoof-2021-DF/SLS_venv/bin/activate

# Add fairseq to PYTHONPATH
export PYTHONPATH="/data/projects/punim2637/nnliang/SLSforASVspoof-2021-DF/fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1:$PYTHONPATH"

cd /data/projects/punim2637/nnliang/SLSforASVspoof-2021-DF

echo "=========================================="
echo "Analyzing Boundary Semantics"
echo "=========================================="
echo "Start time: $(date)"

python analyze_boundary_semantics.py

echo "End time: $(date)"
echo "=========================================="
