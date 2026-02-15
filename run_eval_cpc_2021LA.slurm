#!/bin/bash
#SBATCH --job-name=eval_cpc
#SBATCH --partition=gpu-a100-short
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/eval_cpc_2021LA_%j.out
#SBATCH --error=logs/eval_cpc_2021LA_%j.err

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
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# CPC Model path
MODEL_PATH="/data/projects/punim2637/nnliang/SLSforASVspoof-2021-DF/models/cpc_window_w8_LA_e40_bs14_lr1e-06_saeW0.1_cpcW0.5_dict4096_k128_cpc/best_checkpoint_eer_cpc.pth"

# Database path
DATABASE_PATH="/data/projects/punim2637/nnliang/Datasets/ASVspoof2021_LA_eval"

# Output file
OUTPUT_FILE="scores/scores_cpc_2021_LA.txt"

echo "Model: $MODEL_PATH"
echo "Database: $DATABASE_PATH"
echo "Output: $OUTPUT_FILE"
echo ""

# Run evaluation
python eval_cpc_2021_LA.py \
    --model_path "$MODEL_PATH" \
    --database_path "$DATABASE_PATH" \
    --batch_size 32 \
    --output_file "$OUTPUT_FILE" \
    --sae_k 128 \
    --window_size 8 \
    --cpc_hidden_dim 256 \
    --cpc_weight 0.5

echo ""
echo "End time: $(date)"
echo "Evaluation complete!"
