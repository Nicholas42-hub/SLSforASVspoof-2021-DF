#!/bin/bash

# Monitor CPC temporal stability analysis job

echo "Monitoring CPC Temporal Stability Analysis"
echo "=========================================="
echo ""

JOB_ID="20550898"
LOG_FILE="logs/temporal_stability_cpc_${JOB_ID}.out"

echo "Job ID: $JOB_ID"
echo "Log file: $LOG_FILE"
echo ""

# Check job status
echo "Job Status:"
squeue -j $JOB_ID 2>/dev/null || echo "Job not in queue (may be completed or failed)"
echo ""

# Show log if exists
if [ -f "$LOG_FILE" ]; then
    echo "Current Log Output:"
    echo "-------------------"
    tail -30 "$LOG_FILE"
    echo ""
    echo "For full log: cat $LOG_FILE"
else
    echo "Log file not created yet. Job may be pending..."
fi

echo ""
echo "To check again, run: ./monitor_cpc_temporal.sh"
