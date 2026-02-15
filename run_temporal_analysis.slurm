#!/bin/bash
#SBATCH --job-name=temporal_analysis
#SBATCH --output=/data/projects/punim2637/nnliang/SLSforASVspoof-2021-DF/logs/temporal_analysis_%j.out
#SBATCH --error=/data/projects/punim2637/nnliang/SLSforASVspoof-2021-DF/logs/temporal_analysis_%j.err
#SBATCH --partition=gpu-a100-short
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
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

# Add fairseq to Python path
export PYTHONPATH="/data/projects/punim2637/nnliang/SLSforASVspoof-2021-DF/fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1:$PYTHONPATH"

echo "=========================================="
echo "Temporal Stability Analysis for Window-based TopK SAE"
echo "Started at: $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Running on: $(hostname)"
echo "=========================================="

# Model path
MODEL_PATH="models/topk_sae_window_w8_LA_e40_bs14_lr1e-06_saeW0.1_dict4096_k128_window_topk_w8/best_checkpoint_eer_window_topk_w8.pth"

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    echo "Available models:"
    ls -lh models/
    exit 1
fi

# Check if dataset exists
DATABASE_PATH="/data/projects/punim2637/nnliang/Datasets/LA"
if [ ! -d "$DATABASE_PATH" ]; then
    echo "ERROR: Dataset not found at $DATABASE_PATH"
    exit 1
fi

echo ""
echo "Configuration:"
echo "  Model: $MODEL_PATH"
echo "  Database: $DATABASE_PATH"
echo "  Output: temporal_stability_analysis/"
echo "  Samples: 100"
echo ""

# Run temporal stability analysis
python analyze_temporal_stability.py \
    --model_path "$MODEL_PATH" \
    --database_path "$DATABASE_PATH" \
    --protocols_path "${DATABASE_PATH}/ASVspoof2019_LA_cm_protocols" \
    --output_dir temporal_stability_analysis \
    --num_samples 100 \
    --batch_size 8 \
    --device cuda

exit_code=$?

echo ""
echo "=========================================="
if [ $exit_code -eq 0 ]; then
    echo "Analysis completed successfully!"
    echo ""
    echo "Results saved to: temporal_stability_analysis/"
    echo ""
    echo "Generated files:"
    ls -lh temporal_stability_analysis/
else
    echo "Analysis failed with exit code: $exit_code"
fi
echo "Finished at: $(date)"
echo "=========================================="

exit $exit_code
