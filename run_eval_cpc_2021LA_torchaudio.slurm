#!/bin/bash
#SBATCH --job-name=eval_cpc_2021LA
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --partition=gpu-a100-short
#SBATCH --gres=gpu:1
#SBATCH --output=logs/eval_cpc_2021LA_torchaudio_%j.out
#SBATCH --error=logs/eval_cpc_2021LA_torchaudio_%j.err

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Activate environment
source SLS_venv/bin/activate

# Model and data paths
MODEL_PATH="/data/projects/punim2637/nnliang/SLSforASVspoof-2021-DF/models/cpc_window_w8_LA_e40_bs14_lr1e-06_saeW0.1_cpcW0.5_dict4096_k128_cpc/best_checkpoint_eer_cpc.pth"
DATABASE_PATH="/data/projects/punim2637/nnliang/Datasets/ASVspoof2021_LA_eval"
OUTPUT_FILE="scores/scores_cpc_2021_LA.txt"

echo "Model: $MODEL_PATH"
echo "Database: $DATABASE_PATH"
echo "Output: $OUTPUT_FILE"
echo ""

# Run evaluation
python eval_cpc_2021_LA_torchaudio.py \
    --model_path "$MODEL_PATH" \
    --database_path "$DATABASE_PATH" \
    --output_file "$OUTPUT_FILE" \
    --batch_size 32 \
    --sae_k 128 \
    --window_size 8 \
    --cpc_hidden_dim 256 \
    --cpc_weight 0.5

echo ""
echo "End time: $(date)"
echo "Evaluation complete!"
