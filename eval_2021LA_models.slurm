#!/bin/bash
#SBATCH --job-name=eval_2021LA
#SBATCH --output=/data/projects/punim2637/nnliang/SLSforASVspoof-2021-DF/logs/eval_2021LA_%j.out
#SBATCH --error=/data/projects/punim2637/nnliang/SLSforASVspoof-2021-DF/logs/eval_2021LA_%j.err
#SBATCH --partition=gpu-a100-short
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-02:00:00

# Load required modules
module purge
module load GCCcore/11.3.0
module load Python/3.10.4
module load CUDA/11.8.0
module load FFmpeg/4.4.2  # For audio codecs including FLAC

# Set working directory
cd /data/projects/punim2637/nnliang/SLSforASVspoof-2021-DF

# Activate virtual environment
source SLS_venv/bin/activate

# Add fairseq to Python path
export PYTHONPATH="/data/projects/punim2637/nnliang/SLSforASVspoof-2021-DF/fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1:$PYTHONPATH"

echo "=========================================="
echo "ASVspoof 2021 LA Evaluation"
echo "Started at: $(date)"
echo "=========================================="

# Evaluate K=32 model
echo ""
echo "Evaluating K=32 model..."
python eval_2021_LA_torchaudio.py \
    --model_path models/topk_sae_LA_e40_bs14_lr1e-06_saeW0.1_dict4096_k32_k32_sparse_4096dim/best_checkpoint_eer_k32_sparse_4096dim.pth \
    --database_path /data/projects/punim2637/nnliang/Datasets/ASVspoof2021_LA_eval \
    --batch_size 64 \
    --sae_k 32 \
    --output_file scores/k32_sparse_2021LA_scores.txt

echo ""
echo "=========================================="

# Evaluate K=128 model
echo ""
echo "Evaluating K=128 model..."
python eval_2021_LA_torchaudio.py \
    --model_path models/topk_sae_LA_e40_bs14_lr1e-06_saeW0.1_dict4096_k128_k128_sparse_4096dim/best_checkpoint_eer_k128_sparse_4096dim.pth \
    --database_path /data/projects/punim2637/nnliang/Datasets/ASVspoof2021_LA_eval \
    --batch_size 64 \
    --sae_k 128 \
    --output_file scores/k128_sparse_2021LA_scores.txt

echo ""
echo "=========================================="
echo "Evaluation completed at: $(date)"
echo "=========================================="
