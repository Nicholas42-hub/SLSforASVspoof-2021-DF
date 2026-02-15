#!/bin/bash
# Helper script to monitor your SLURM job

JOB_ID=$1

if [ -z "$JOB_ID" ]; then
    echo "Usage: ./monitor_job.sh <job_id>"
    echo ""
    echo "Current jobs:"
    squeue -u $USER
    exit 1
fi

echo "=========================================="
echo "Monitoring Job: $JOB_ID"
echo "=========================================="

# Check job status
echo -e "\n[Job Status]"
squeue -j $JOB_ID

# Show recent output
echo -e "\n[Recent Output] (last 30 lines)"
LOG_FILE="logs/train_df_${JOB_ID}.out"
if [ -f "$LOG_FILE" ]; then
    tail -30 "$LOG_FILE"
else
    echo "Log file not yet created: $LOG_FILE"
fi

# Show recent errors
echo -e "\n[Recent Errors] (last 20 lines)"
ERR_FILE="logs/train_df_${JOB_ID}.err"
if [ -f "$ERR_FILE" ]; then
    tail -20 "$ERR_FILE"
else
    echo "Error file not yet created: $ERR_FILE"
fi

echo -e "\n=========================================="
echo "To continuously monitor output: tail -f logs/train_df_${JOB_ID}.out"
echo "To cancel job: scancel $JOB_ID"
echo "=========================================="
