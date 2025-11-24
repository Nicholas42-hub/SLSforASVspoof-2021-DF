# Meeting Feedback Implementation Summary

**Date**: November 7, 2025  
**Meeting Date**: [From transcript]

## ✅ Completed Changes

### 1. **Table Formatting (Chris's Feedback)**

**Issue**: Tables lacked clear visual separation between baseline methods and proposed method.

**Action Taken**:

- Added `\midrule` separator in Table 1 (Main Results) between baseline methods and "Our Method"
- Added `\midrule` separator in Table 2 (Ablation Studies) between baseline and proposed components
- Changed "XLS-R + SLS" to "XLS-R + SLS (Baseline)" for clarity

**Location**: Lines ~227, ~257 in paper_draft.tex

---

### 2. **Strengthen Problem-Solution Narrative in Section 2.1 Overview (Chris & Karen's Feedback)**

**Issue**: Section only described WHAT was done, not WHY. Missing motivation for hierarchical attention and theoretical significance.

**Action Taken**:

- Added **"Problem Motivation"** subsection explaining two fundamental limitations:
  - Temporal uniformity (all frames weighted equally)
  - Inter-layer independence (missing complementary relationships)
- Added **"Our Solution"** subsection with explicit challenge-solution mapping for each stage:
  - Stage 1: Addresses "varying frame informativeness within layers"
  - Stage 2: Addresses "complementary features within abstraction levels"
  - Stage 3: Addresses "multi-level artifact manifestation across hierarchy"
- Added **"Why Hierarchical Attention Solves the Problem"** explaining compositional structure modeling
- Emphasized contrast with flat fusion and independent weighting approaches

**Location**: Section 2.1 (~lines 107-125)

---

### 3. **Enhance Hierarchical Attention Sections with Purpose/Limitations (Karen's Feedback)**

**Issue**: Each attention stage lacked clear explanation of what limitation it solves and why it's necessary.

**Action Taken** for each stage:

**Stage 1: Temporal Attention**

- Added "Challenge Addressed" subsection explaining varying frame informativeness
- Added "Solution" subsection explaining how attention weights identify stable regions
- Provided concrete examples (phase discontinuities, segmentation noise)

**Stage 2: Intra-Group Attention**

- Added "Challenge Addressed" explaining within-level complementarities
- Added "Solution" explaining how grouping prevents redundant feature selection
- Provided examples (spectral envelopes vs phase relationships)

**Stage 3: Inter-Group Attention**

- Added "Challenge Addressed" explaining multi-level artifact manifestation
- Added "Solution" explaining adaptive multi-level integration
- Provided examples (vocoder distortions, prosody anomalies, semantic inconsistencies)
- Added "Hierarchical Design Rationale" summarizing the three-stage pipeline

**Location**: Section 2.3 (~lines 151-200)

---

### 4. **Add Research Questions and Experiment Aims (Chris's Feedback)**

**Issue**: Experiments only described HOW they were done, not WHAT questions they answer or WHAT issues they investigate.

**Action Taken**:

- Added opening paragraph to Section 3 with **three research questions**:

  - **RQ1**: Does hierarchical attention achieve state-of-the-art performance?
  - **RQ2**: What are individual contributions of each component?
  - **RQ3**: Do learned attention patterns validate our design?

- Added aim statement to **Section 3.2 (Main Results)**:

  - Evaluating whether explicit inter-layer dependency modeling improves generalization

- Added aim statement to **Section 3.3 (Ablation Studies)**:

  - Understanding complementary effects on in-domain vs cross-domain performance

- Added aim statement to **Section 3.4 (Attention Analysis)**:
  - Investigating whether learned patterns align with architectural motivations

**Location**: Section 3 opening (~line 206), subsection openings (~lines 216, 252, 274)

---

## ⚠️ Still Needs Verification

### 5. **Score Consistency Check (Chris's Critical Feedback)**

**Issue**: Chris questioned if 8.87% for SLS on In-the-Wild is average or best score. Original paper reported 7.46%. Using inconsistent metrics (average for one method, best for another) creates bad impression.

**Action Required**:

- **URGENT**: Check the original SLS paper (zhang2024audio) to verify:
  - Is 8.87% the average score or best score?
  - What did they report for other datasets (21 LA: 3.88%, 21 DF: 2.09%)?
  - Are these all averages or all best scores?
- If 8.87% is the best score but other methods report averages, need to either:
  - Use average scores consistently across all methods, OR
  - Use best scores consistently across all methods
- **Add footnote** to table clarifying: "Results are averaged over 3 runs" (if using averages)

**Status**: NOT YET VERIFIED - needs checking against original paper

---

### 6. **Bold All Top Scores (Chris's Suggestion)**

**Issue**: Some reviewers prefer bolding ALL highest scores in each column, not just proposed method.

**Current State**: Only proposed method scores are bolded (2.46, 1.93, 6.87)

**Action Required**:

- Check each column in Table 1:
  - 21 LA: 2.46 is lowest ✓ (already bold)
  - 21 DF: 1.93 is lowest ✓ (already bold)
  - ITW: 6.87 is lowest ✓ (already bold)
- **Conclusion**: Current bolding is correct since proposed method has best scores in all columns

**Status**: VERIFIED - no changes needed (proposed method wins all columns)

---

## 📋 Summary

| Feedback Item                            | Status                        | Priority     |
| ---------------------------------------- | ----------------------------- | ------------ |
| Table visual separation (midrules)       | ✅ DONE                       | Medium       |
| Problem-solution narrative (Section 2.1) | ✅ DONE                       | **HIGH**     |
| Stage-by-stage purpose/limitations       | ✅ DONE                       | **HIGH**     |
| Research questions for experiments       | ✅ DONE                       | **HIGH**     |
| Score consistency verification           | ⚠️ NEEDS VERIFICATION         | **CRITICAL** |
| Bold all top scores                      | ✅ VERIFIED (already correct) | Low          |

---

## 🎯 Next Steps

1. **CRITICAL**: Verify SLS scores (8.87% for ITW) against original paper - check if comparing average vs average or best vs best consistently
2. Compile paper in Overleaf to check page count (must be ≤ 4 pages)
3. Review expanded Overview and Hierarchical Attention sections for length - may need condensing if over page limit
4. Update author names and affiliations before submission

---

## 📄 Files Modified

- `paper_draft.tex` - Main manuscript (multiple sections enhanced)

## 📊 Word Count Impact

- Section 2.1 (Overview): **~200 words added**
- Section 2.3 (Hierarchical Attention): **~400 words added**
- Section 3 (Experiments): **~100 words added**
- **Total added**: ~700 words

**Warning**: Added content may push paper over 4-page limit. Monitor page count in Overleaf compilation.
