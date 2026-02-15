#!/bin/bash
#SBATCH --job-name=topk_sae_df
#SBATCH --output=/data/projects/punim2637/nnliang/SLSforASVspoof-2021-DF/logs/train_df_%j.out
#SBATCH --error=/data/projects/punim2637/nnliang/SLSforASVspoof-2021-DF/logs/train_df_%j.err
#SBATCH --partition=gpu-a100-short
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-04:00:00

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
echo "GPU devices: $CUDA_VISIBLE_DEVICES"
echo "Python version: $(python --version)"
echo "Python path: $(which python)"
echo "=========================================="

# Test basic imports
echo "Testing imports..."
python -c "import torch; print(f'PyTorch {torch.__version__}')" || { echo "PyTorch import failed"; exit 1; }
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" || { echo "CUDA check failed"; exit 1; }

# Add fairseq to Python path instead of importing it
export PYTHONPATH="/data/projects/punim2637/nnliang/SLSforASVspoof-2021-DF/fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1:$PYTHONPATH"
echo "Fairseq path added to PYTHONPATH"

echo "=========================================="
echo "Starting TopK SAE training..."
echo "=========================================="

# Run the training command
python main.py \
    --database_path=/data/projects/punim2637/nnliang/Datasets/LA/ \
    --protocols_path=/data/projects/punim2637/nnliang/Datasets/LA/ \
    --track=LA \
    --batch_size=14 \
    --num_epochs=40 \
    --lr=1e-6 \
    --weight_decay=1e-4 \
    --cp_path=xlsr2_300m.pt \
    --sae_weight=0.1 \
    --sae_dict_size=4096 \
    --sae_k=128 \
    --use_sparse_features \
    --seed=1234 \
    --comment=topk_sae_sparse

echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s