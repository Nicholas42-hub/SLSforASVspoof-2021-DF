#!/bin/bash
#SBATCH --job-name=temp_test
#SBATCH --partition=gpu-a100-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=0:30:00
#SBATCH --output=logs/temporal_test_%j.out
#SBATCH --error=logs/temporal_test_%j.err

echo "=========================================="
echo "Quick Test - Temporal Stability Analysis"
echo "Started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "=========================================="
echo ""

# Load modules
module purge
module load GCCcore/11.3.0
module load Python/3.10.4
module load CUDA/11.8.0

# Set working directory
cd /data/projects/punim2637/nnliang/SLSforASVspoof-2021-DF

# Activate virtual environment
source SLS_venv/bin/activate

# Set PYTHONPATH
export PYTHONPATH="/data/projects/punim2637/nnliang/SLSforASVspoof-2021-DF/fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1:$PYTHONPATH"

# Set dataset paths
DATABASE_PATH="/data/projects/punim2637/nnliang/Datasets/LA"
PROTOCOLS_PATH="${DATABASE_PATH}/ASVspoof2019_LA_cm_protocols"
MODEL_PATH="models/topk_sae_window_w8_LA_e40_bs14_lr1e-06_saeW0.1_dict4096_k128_window_topk_w8/best_checkpoint_eer_window_topk_w8.pth"
OUTPUT_DIR="temporal_stability_analysis"

echo "Configuration:"
echo "  Model: $MODEL_PATH"
echo "  Database: $DATABASE_PATH"
echo "  Output: $OUTPUT_DIR"
echo "  Samples: 5 (TEST RUN)"
echo ""

# Run analysis with small sample size
python analyze_temporal_stability.py \
    --model_path "$MODEL_PATH" \
    --database_path "$DATABASE_PATH" \
    --protocols_path "$PROTOCOLS_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_samples 5 \
    --batch_size 2 \
    --device cuda

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Test completed successfully!"
else
    echo "Test failed with exit code: $EXIT_CODE"
fi
echo "Finished at: $(date)"
echo "=========================================="

exit $EXIT_CODE
