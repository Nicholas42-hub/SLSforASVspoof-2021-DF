#!/bin/bash
#SBATCH --job-name=eval_2021LA
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/eval_2021LA_%j.out
#SBATCH --error=logs/eval_2021LA_%j.err

echo "=========================================="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "=========================================="

# Activate virtual environment
source SLS_venv/bin/activate

# Set paths
MODEL_PATH="models/topk_sae_window_w8_LA_e40_bs14_lr1e-06_saeW0.1_dict4096_k128_window_topk_w8/best_checkpoint_eer_window_topk_w8.pth"
DATABASE_PATH="/data/projects/punim2637/nnliang/Datasets/ASVspoof2021_LA_eval"
OUTPUT_FILE="scores/window_topk_w8_2021LA_scores.txt"

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    exit 1
fi

# Check if database exists
if [ ! -d "$DATABASE_PATH" ]; then
    echo "ERROR: Database not found at $DATABASE_PATH"
    exit 1
fi

# Run evaluation
echo ""
echo "Starting evaluation..."
echo "Model: $MODEL_PATH"
echo "Database: $DATABASE_PATH"
echo "Output: $OUTPUT_FILE"
echo ""

python eval_2021_LA_window_torchaudio.py \
    --model_path "$MODEL_PATH" \
    --database_path "$DATABASE_PATH" \
    --batch_size 32 \
    --output_file "$OUTPUT_FILE" \
    --sae_k 128 \
    --window_size 8

# Check if evaluation succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Evaluation completed successfully!"
    echo "=========================================="
    
    # Compute EER
    echo ""
    echo "Computing EER..."
    python compute_eer_2021LA.py "$OUTPUT_FILE"
    
    echo ""
    echo "=========================================="
    echo "Analysis finished at: $(date)"
    echo "Exit code: $?"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "ERROR: Evaluation failed!"
    echo "Exit code: $?"
    echo "=========================================="
fi
