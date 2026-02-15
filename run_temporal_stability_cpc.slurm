#!/bin/bash
#SBATCH --job-name=cpc_temporal_stability
#SBATCH --account=punim2637
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/temporal_stability_cpc_%j.out
#SBATCH --error=logs/temporal_stability_cpc_%j.err

# Load modules
module purge
module load GCCcore/11.3.0
module load Python/3.10.4
module load CUDA/11.8.0

# Activate virtual environment
source SLS_venv/bin/activate

# Set paths
export PYTHONPATH="${PYTHONPATH}:/data/projects/punim2637/nnliang/SLSforASVspoof-2021-DF"
export PYTHONPATH="/data/projects/punim2637/nnliang/SLSforASVspoof-2021-DF/fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1:$PYTHONPATH"
cd /data/projects/punim2637/nnliang/SLSforASVspoof-2021-DF

# Model path
MODEL_PATH="/data/projects/punim2637/nnliang/SLSforASVspoof-2021-DF/models/cpc_window_w8_LA_e40_bs14_lr1e-06_saeW0.1_cpcW0.5_dict4096_k128_cpc/best_checkpoint_eer_cpc.pth"

# Output directory
OUTPUT_DIR="temporal_stability_analysis/cpc"

# Create output directory
mkdir -p $OUTPUT_DIR
mkdir -p logs

echo "Starting CPC Temporal Stability Analysis"
echo "========================================"
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"
echo "Time: $(date)"
echo ""

# Run analysis
python analyze_temporal_stability_cpc.py \
    --model_path "$MODEL_PATH" \
    --database_path "/data/projects/punim2637/nnliang/Datasets/LA" \
    --protocols_path "/data/projects/punim2637/nnliang/Datasets/LA/ASVspoof2019_LA_cm_protocols" \
    --output_dir "$OUTPUT_DIR" \
    --num_samples 100 \
    --batch_size 8 \
    --device cuda

echo ""
echo "Analysis completed at $(date)"
echo "Results saved to: $OUTPUT_DIR"
