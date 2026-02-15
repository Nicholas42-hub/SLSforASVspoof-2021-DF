#!/bin/bash
#SBATCH --job-name=cpc_train
#SBATCH --output=/data/projects/punim2637/nnliang/SLSforASVspoof-2021-DF/logs/train_cpc_%j.out
#SBATCH --error=/data/projects/punim2637/nnliang/SLSforASVspoof-2021-DF/logs/train_cpc_%j.err
#SBATCH --partition=gpu-a100-short
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-04:00:00

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
echo "Training Window TopK SAE with CPC"
echo "=========================================="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="

# Add fairseq to Python path
export PYTHONPATH="/data/projects/punim2637/nnliang/SLSforASVspoof-2021-DF/fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1:$PYTHONPATH"

# Check if checkpoint exists to determine resume flag
MODEL_DIR="models/cpc_window_w8_LA_e40_bs14_lr1e-06_saeW0.1_cpcW0.5_dict4096_k128_cpc"
CHECKPOINT="$MODEL_DIR/last_checkpoint.pth"

if [ -f "$CHECKPOINT" ]; then
    echo "Found checkpoint, resuming training..."
    RESUME_FLAG="--resume"
else
    echo "No checkpoint found, starting fresh..."
    RESUME_FLAG=""
fi

# Run CPC training
python train_cpc.py \
    --database_path=/data/projects/punim2637/nnliang/Datasets/LA/ \
    --protocols_path=/data/projects/punim2637/nnliang/Datasets/LA/ \
    --track=LA \
    --batch_size=14 \
    --num_epochs=40 \
    --lr=1e-6 \
    --weight_decay=1e-4 \
    --cp_path=xlsr2_300m.pt \
    --sae_weight=0.1 \
    --sae_dict_size=4096 \
    --sae_k=128 \
    --sae_window_size=8 \
    --cpc_hidden_dim=256 \
    --cpc_weight=0.5 \
    --cpc_temperature=0.07 \
    --cpc_prediction_steps 1 2 4 \
    --seed=1234 \
    --comment=cpc \
    $RESUME_FLAG

EXIT_CODE=$?

echo "=========================================="
echo "Training finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

# Check if training is complete by examining the checkpoint
if [ -f "$CHECKPOINT" ]; then
    # Extract current epoch from training log
    LOG_FILE="$MODEL_DIR/training_log.csv"
    if [ -f "$LOG_FILE" ]; then
        LAST_EPOCH=$(tail -1 "$LOG_FILE" | cut -d',' -f1)
        echo "Last completed epoch: $LAST_EPOCH"
        
        # If not finished (epoch < 39 since epoch 0-39 = 40 epochs), resubmit
        if [ "$LAST_EPOCH" -lt 39 ]; then
            echo "Training not complete (epoch $LAST_EPOCH/39), auto-resubmitting..."
            sleep 5
            sbatch $0  # Resubmit this same script
            echo "New job submitted!"
        else
            echo "Training complete! All 40 epochs finished."
        fi
    fi
fi

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s
