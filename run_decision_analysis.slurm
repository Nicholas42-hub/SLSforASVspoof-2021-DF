#!/bin/bash
#SBATCH --job-name=decision_analysis
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/decision_analysis_%j.out
#SBATCH --error=logs/decision_analysis_%j.err

# Decision Relevance Analysis - Phase 1 of Caren's Direction #4
# Implements: Feature Attribution, Decision-Relevant Stability, Cue Consistency

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Setup
cd /data/projects/punim2637/nnliang/SLSforASVspoof-2021-DF
source SLS_venv/bin/activate

# Add fairseq to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/data/gpfs/projects/punim2637/nnliang/SLSforASVspoof-2021-DF/fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1"

# Create logs directory if it doesn't exist
mkdir -p logs
mkdir -p decision_analysis_2021LA_5k

# Run analysis
echo "Starting Decision Relevance Analysis..."

python analyze_decision_relevance.py \
    --model_path models/topk_sae_window_w8_LA_e40_bs14_lr1e-06_saeW0.1_dict4096_k128_window_topk_w8/best_checkpoint_eer_window_topk_w8.pth \
    --output_dir decision_analysis_2021LA_5k \
    --dataset 2021_LA \
    --num_samples 5000 \
    --top_k_features 50 \
    --batch_size 8 \
    --run_ablation

# Optional: Run with more samples for comprehensive analysis
# python analyze_decision_relevance.py \
#     --model_path checkpoints/window_topk/best_model.pth \
#     --output_dir decision_analysis_full \
#     --num_samples 500 \
#     --top_k_features 100 \
#     --batch_size 8

echo "Analysis complete at: $(date)"
echo "Results saved to: decision_analysis/"

# Print summary
if [ -f decision_analysis_2021LA_5k/decision_analysis_results.json ]; then
    echo ""
    echo "=== SUMMARY ==="
    python -c "
import json
with open('decision_analysis_2021LA_5k/decision_analysis_results.json') as f:
    results = json.load(f)
    
print(f\"Samples analyzed: {results['summary']['num_samples_analyzed']}\")
print(f\"  Genuine: {results['summary']['num_genuine']}\")
print(f\"  Spoof: {results['summary']['num_spoof']}\")
print()
print('Temporal Stability:')
print(f\"  All features Jaccard: {results['stability']['all_features']['mean_jaccard']:.3f}\")
print(f\"  Decision features Jaccard: {results['stability']['decision_features']['mean_jaccard']:.3f}\")
print()
print('Boundary Effects (Decision Features):')
print(f\"  Interior: {results['stability']['decision_features']['boundary_effects']['mean_interior']:.3f}\")
print(f\"  Cross-boundary: {results['stability']['decision_features']['boundary_effects']['mean_cross_boundary']:.3f}\")
print()
print('Cue Consistency:')
print(f\"  Genuine: {results['cue_consistency']['genuine']['mean_overlap']:.3f}\")
print(f\"  Spoof: {results['cue_consistency']['spoof']['mean_overlap']:.3f}\")
"
fi
