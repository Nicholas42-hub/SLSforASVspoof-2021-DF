#!/bin/bash
#SBATCH --job-name=eval_2021_LA
#SBATCH --output=/data/projects/punim2637/nnliang/SLSforASVspoof-2021-DF/logs/eval_2021_LA_%j.out
#SBATCH --error=/data/projects/punim2637/nnliang/SLSforASVspoof-2021-DF/logs/eval_2021_LA_%j.err
#SBATCH --partition=gpu-a100-short
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
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

# Print environment information
echo "=========================================="
echo "Evaluating on ASVspoof 2021 LA"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="

# Add fairseq to Python path
export PYTHONPATH="/data/projects/punim2637/nnliang/SLSforASVspoof-2021-DF/fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1:$PYTHONPATH"

# Run evaluation
python eval_2021_LA_simple.py \
    --model_path=models/topk_sae_LA_e40_bs14_lr1e-06_saeW0.1_dict4096_k128_topk_sae_sparse/best_checkpoint_eer_topk_sae_sparse.pth \
    --database_path=/data/projects/punim2637/nnliang/Datasets/ASVspoof2021_LA_eval/ \
    --batch_size=32 \
    --output_file=scores/scores_2021_LA_best.txt

echo "=========================================="
echo "Evaluation finished at: $(date)"
echo "=========================================="

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s
