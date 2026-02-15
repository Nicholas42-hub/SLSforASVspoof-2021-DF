#!/bin/bash
#SBATCH --job-name=analyze_neurons
#SBATCH --output=/data/projects/punim2637/nnliang/SLSforASVspoof-2021-DF/logs/analyze_%j.out
#SBATCH --error=/data/projects/punim2637/nnliang/SLSforASVspoof-2021-DF/logs/analyze_%j.err
#SBATCH --partition=gpu-a100-short
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-01:00:00

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
echo "SAE Neuron Interpretability Analysis"
echo "Started at: $(date)"
echo "=========================================="

# Analyze K=32 model
echo ""
echo "Analyzing K=32 model..."
python analyze_sae_neurons.py \
    --model_path models/topk_sae_LA_e40_bs14_lr1e-06_saeW0.1_dict4096_k32_k32_sparse_4096dim/best_checkpoint_eer_k32_sparse_4096dim.pth \
    --sae_k 32 \
    --num_samples 500 \
    --output_dir analysis_results_k32

echo ""
echo "=========================================="

# Analyze K=128 model
echo ""
echo "Analyzing K=128 model..."
python analyze_sae_neurons.py \
    --model_path models/topk_sae_LA_e40_bs14_lr1e-06_saeW0.1_dict4096_k128_k128_sparse_4096dim/best_checkpoint_eer_k128_sparse_4096dim.pth \
    --sae_k 128 \
    --num_samples 500 \
    --output_dir analysis_results_k128

echo ""
echo "=========================================="
echo "Analysis completed at: $(date)"
echo "=========================================="
