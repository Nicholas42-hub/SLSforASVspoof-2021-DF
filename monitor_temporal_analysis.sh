#!/bin/bash
# Monitor temporal stability analysis job

JOB_ID=${1:-20502912}
LOG_DIR="/data/projects/punim2637/nnliang/SLSforASVspoof-2021-DF/logs"

echo "Monitoring job: $JOB_ID"
echo "========================================"

# Check job status
echo ""
echo "Job Status:"
squeue -j $JOB_ID 2>/dev/null || echo "Job completed or not found"

# Show last 30 lines of output
echo ""
echo "Last 30 lines of output:"
echo "----------------------------------------"
tail -30 "$LOG_DIR/temporal_analysis_${JOB_ID}.out" 2>/dev/null || echo "Output file not yet created"

# Show errors if any
echo ""
echo "Errors (if any):"
echo "----------------------------------------"
tail -20 "$LOG_DIR/temporal_analysis_${JOB_ID}.err" 2>/dev/null || echo "No errors"

# Check results directory
echo ""
echo "Results directory:"
echo "----------------------------------------"
ls -lh temporal_stability_analysis/ 2>/dev/null || echo "Results not yet available"

echo ""
echo "========================================"
echo "To monitor continuously, run:"
echo "  watch -n 5 ./monitor_temporal_analysis.sh $JOB_ID"
