#!/bin/bash
#SBATCH --job-name=eval_LA_2021
#SBATCH --account=punim2637
#SBATCH --partition=gpu-a100-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=0-01:00:00
#SBATCH --output=logs/eval_LA_2021_%j.out
#SBATCH --error=logs/eval_LA_2021_%j.err

echo "=========================================="
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Started at: $(date)"
echo "=========================================="

# Load modules
module load python/3.10.4

# Activate virtual environment
source /data/projects/punim2637/nnliang/SLSforASVspoof-2021-DF/SLS_venv/bin/activate

# Add fairseq to Python path
export PYTHONPATH="/data/projects/punim2637/nnliang/SLSforASVspoof-2021-DF/fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1:$PYTHONPATH"

# Set working directory
cd /data/projects/punim2637/nnliang/SLSforASVspoof-2021-DF

# Define paths
MODEL_PATH="models/topk_sae_LA_e40_bs14_lr1e-06_saeW0.1_dict4096_k128_topk_sae_test/best_checkpoint_eer_topk_sae_test.pth"
PROTOCOLS_PATH="database/ASVspoof_DF_cm_protocols/ASVspoof2021.LA.cm.eval.available.txt"
DATABASE_PATH="/data/projects/punim2637/nnliang/Datasets/ASVspoof2021_LA_eval"
SCORES_OUTPUT="scores/scores_LA_2021_topk_sae.txt"

echo ""
echo "=========================================="
echo "Evaluation Configuration"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Protocol: $PROTOCOLS_PATH"
echo "Database: $DATABASE_PATH"
echo "Output: $SCORES_OUTPUT"
echo ""

# Create scores directory if not exists
mkdir -p scores

# Run evaluation
echo "Running evaluation on ASVspoof 2021 LA eval set..."
python main.py --track LA --is_eval \
    --cp_path xlsr2_300m.pt \
    --model_path $MODEL_PATH \
    --protocols_path $PROTOCOLS_PATH \
    --database_path $DATABASE_PATH \
    --eval_output $SCORES_OUTPUT

echo ""
echo "=========================================="
echo "Computing EER on ASVspoof 2021 LA"
echo "=========================================="

# Run evaluation metric
python evaluate_2021_LA.py "$SCORES_OUTPUT" /data/projects/punim2637/nnliang/Datasets/keys eval

echo ""
echo "=========================================="
echo "Evaluation completed at: $(date)"
echo "=========================================="
