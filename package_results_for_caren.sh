#!/bin/bash

# Package all analysis results for Caren's review

OUTPUT_DIR="results_for_caren_$(date +%Y%m%d)"
mkdir -p "$OUTPUT_DIR"

echo "Packaging analysis results for Caren..."
echo "=========================================="

# 1. Copy comparison report
echo "1. Per-timestep vs Window TopK comparison..."
cp temporal_comparison/comparison_report.txt "$OUTPUT_DIR/1_comparison_per_timestep_vs_window.txt"
cp temporal_comparison/comparison_lifetime.png "$OUTPUT_DIR/1_lifetime_comparison.png" 2>/dev/null
cp temporal_comparison/comparison_jaccard.png "$OUTPUT_DIR/1_jaccard_comparison.png" 2>/dev/null

# 2. Copy Window TopK failure analysis
echo "2. Window TopK failure mode analysis..."
cp temporal_stability_analysis/top_k_window/summary_report.txt "$OUTPUT_DIR/2_window_topk_failure_modes.txt"
cp temporal_stability_analysis/top_k_window/*.png "$OUTPUT_DIR/" 2>/dev/null

# 3. Copy CPC analysis
echo "3. CPC degradation analysis..."
cp cpc_analysis_summary.txt "$OUTPUT_DIR/3_cpc_degradation_analysis.txt" 2>/dev/null

# 4. Copy all EER metrics
echo "4. Performance metrics (EER/t-DCF)..."
cat > "$OUTPUT_DIR/4_all_model_performance.txt" << 'EOF'
PERFORMANCE COMPARISON - ASVspoof2021 LA
========================================

Per-timestep TopK (k=128, w=1):
  EER: 2.8012%
  min t-DCF: 0.452261
  
Window TopK (k=128, w=8):
  EER: 2.9405%
  min t-DCF: 0.452261
  Degradation: +0.14% (acceptable)

CPC (k=128, w=8, cpc_weight=0.5):
  EER: 9.0409%
  min t-DCF: 0.452261
  Degradation: +6.24% (catastrophic)
  
EOF

cat scores/k128_sparse_2021LA_scores_metrics.txt >> "$OUTPUT_DIR/4_all_model_performance.txt"
echo "" >> "$OUTPUT_DIR/4_all_model_performance.txt"
cat scores/window_topk_w8_2021LA_scores_metrics.txt >> "$OUTPUT_DIR/4_all_model_performance.txt"
echo "" >> "$OUTPUT_DIR/4_all_model_performance.txt"
cat scores/scores_cpc_2021_LA_metrics.txt >> "$OUTPUT_DIR/4_all_model_performance.txt"

# 5. Copy email draft
echo "5. Email draft..."
cp email_draft_to_caren.txt "$OUTPUT_DIR/0_EMAIL_DRAFT.txt"

# 6. Create executive summary
echo "6. Creating executive summary..."
cat > "$OUTPUT_DIR/EXECUTIVE_SUMMARY.txt" << 'EOF'
EXECUTIVE SUMMARY - TEMPORAL FEATURE ANALYSIS
==============================================

COMPLETED WORK (as requested by Caren):

1. ✓ Defined "good" temporal SAE features
   - 6 quantitative metrics established
   - Covers persistence, coherence, and failure modes

2. ✓ Analyzed Window TopK failure modes
   - 17.6% transient spikes (noise activations)
   - 4.66 features flip per frame (residual instability)
   - Despite structural constraints, temporal noise persists

3. ✓ Controlled study with temporal regularization
   - Per-timestep: EER 2.80%, highly volatile (59.6% transient)
   - Window TopK: EER 2.94%, balanced (17.6% transient)
   - CPC: EER 9.04%, over-regularized (performance collapse)

KEY INSIGHT:
Window-based TopK achieves optimal temporal-discriminative balance through
architectural constraints, not explicit loss functions.

PAPER POSITIONING:
"Understanding Temporal Dynamics in Sparse Audio Representations"
- Diagnostic framework (not performance improvement)
- Failure mode analysis (not engineering solution)
- Temporal-discriminative trade-off characterization

NEXT STEPS:
- CPC temporal stability analysis (in progress, Job ID: 20550248)
- Qualitative feature semantic analysis (optional)
- Paper draft outline

FILES IN THIS PACKAGE:
- 0_EMAIL_DRAFT.txt: Complete email to Caren
- 1_comparison_per_timestep_vs_window.txt: Quantitative comparison
- 2_window_topk_failure_modes.txt: Detailed failure analysis
- 3_cpc_degradation_analysis.txt: Root cause of CPC failure
- 4_all_model_performance.txt: EER/t-DCF metrics
- *.png: Visualization plots

READY FOR FRIDAY MEETING: YES ✓
EOF

# Create README
cat > "$OUTPUT_DIR/README.txt" << 'EOF'
RESULTS PACKAGE FOR CAREN
==========================

This directory contains all analysis results responding to your three requests:
1. Define "good" temporal SAE features → See files 1 and 2
2. Analyze Window TopK failure modes → See file 2
3. Controlled temporal regularization study → See files 1, 3, 4

Start with:
- EXECUTIVE_SUMMARY.txt (2-minute overview)
- 0_EMAIL_DRAFT.txt (full detailed email)

Then review supporting data files 1-4 for specifics.

Visualizations (*.png) show lifetime distributions, Jaccard similarity, etc.
EOF

echo ""
echo "Package complete!"
echo "================="
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Contents:"
ls -lh "$OUTPUT_DIR"
echo ""
echo "To review:"
echo "  cat $OUTPUT_DIR/EXECUTIVE_SUMMARY.txt"
echo "  cat $OUTPUT_DIR/0_EMAIL_DRAFT.txt"
