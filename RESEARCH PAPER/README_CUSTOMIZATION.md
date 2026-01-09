# Reproducibility Study LaTeX Paper - Customization Guide

## Files Provided

1. **reproducibility_paper.tex** - Main manuscript (complete paper, ~30 pages)
2. **supplementary_table_s1_claim.tex** - CLAIM checklist assessment
3. **This README** - Instructions for customization

## What You Need to Fill In

The paper is ~95% complete, but has placeholders marked with **[YOUR DATA]** or **[INSERT]** that you need to customize with your actual results.

### Critical Sections to Customize

#### 1. Title Page (Line 39-43)
- Replace `[Supervisor Last Name]` with Hannah's full name
- Add any additional co-authors
- Update institution details
- Add correct email address

#### 2. Abstract (Lines 51-69)
**Key placeholders:**
- Line 64: `[INSERT YOUR RESULTS: maintained/degraded performance with AUC of X.XX]`
- Line 69: `[INSERT: degree of generalizability]`

**How to fill:**
After you run your CMC validation, insert your actual AUC results. Examples:
- If performance maintained: "External validation showed maintained performance with AUC of 0.68 (95% CI: 0.61-0.75) compared to 0.70 on MAASTRO"
- If degraded: "External validation revealed performance degradation with AUC of 0.58 (95% CI: 0.51-0.65)"

#### 3. Methods - Computational Environment (Lines 248-255)
**Line 251:** `[SPECIFY YOUR HARDWARE: e.g., NVIDIA RTX 3090 GPU, 32GB RAM]`

Fill in your actual hardware specs.

#### 4. Methods - CMC Vellore Dataset (Lines 389-420)
**Multiple placeholders to fill:**
- Line 392: `[SPECIFY: e.g., 2020-2023]` - Your study period
- Lines 395-399: Scanner details (manufacturer, model, slice thickness, etc.)
- Line 401: `[SPECIFY NUMBER]` - Number of locoregional recurrence events in your cohort

#### 5. Table 2: Cohort Comparison (Lines 423-448)
**Fill in all the `[YOUR DATA]` cells:**

This is CRITICAL - you need to create a comparison table showing differences between MAASTRO and CMC cohorts:
- Demographics (age, sex)
- Tumor sites (oropharynx, larynx, hypopharynx, other)
- TNM staging distribution
- Treatment modalities
- Imaging protocol differences

**Example of how to fill:**
```latex
Age (years), mean (SD) & 61.2 (9.4) & 58.7 (11.2) \\
Male sex, n (\%) & 224 (75.2) & 127 (77.9) \\
```

#### 6. Results - External Validation (Lines 674-760)
**This is the MOST IMPORTANT section - your novel contribution!**

**Table 3 (Lines 679-707):** Fill in ALL `[YOUR RESULT]` cells with your actual CMC validation results:
- AUC with 95% CI
- Accuracy, Sensitivity, Specificity, PPV, NPV
- For both imaging-only and imaging+clinical models

**Interpretation section (Lines 711-760):**
- Choose the appropriate interpretation based on whether your results showed performance maintenance or degradation
- I've provided template text for both scenarios
- Delete the one that doesn't apply and customize the other

#### 7. Discussion - External Validation Insights (Lines 891-961)
**Customize based on your actual results:**

If performance **maintained** (Lines 899-923):
- Keep this section
- Fill in specific details about your cohort
- Discuss why generalization succeeded

If performance **degraded** (Lines 927-957):
- Keep this section
- Analyze specific factors from your Table 2
- Discuss population/protocol differences

**Delete the section that doesn't apply!**

#### 8. References (Lines 1289-1409)
**Line 1386-1388:** Update the Mateus et al. citation with complete details:
```latex
\bibitem{mateus2023imagingbased}
Mateus P, [Complete author list].
[Complete title].
\textit{Medical Physics}. 2023;50(X):XXXX-XXXX.
```

#### 9. Ethics and Funding (Lines 1268-1275)
- Line 1271: `[IRB Min No: SPECIFY]` - Add your IRB approval number
- Line 1283: `[SPECIFY YOUR FUNDING SOURCES]` - List your grants/funding

### Sections That Are Complete (No Changes Needed)

These sections are fully written and don't need customization:
- Introduction (Lines 72-175)
- Methods - Framework (Lines 181-241)
- Methods - Reproduction procedure (Lines 276-387)
- Results - CLAIM compliance (Lines 469-491)
- Results - Reproducibility barriers (Lines 495-651)
- Results - Reproduction on MAASTRO (Lines 655-672)
- Discussion - Most sections (Lines 764-1050)
- Conclusion (Lines 1185-1197)

## How to Compile

### Option 1: Overleaf (Recommended)
1. Create a new project on Overleaf
2. Upload `reproducibility_paper.tex`
3. Compile (should work immediately)
4. Fill in placeholders directly in Overleaf

### Option 2: Local LaTeX Installation
```bash
# Install LaTeX (Ubuntu/Debian)
sudo apt-get install texlive-full

# Compile
pdflatex reproducibility_paper.tex
pdflatex reproducibility_paper.tex  # Run twice for references
```

### Option 3: VS Code with LaTeX Workshop
1. Install "LaTeX Workshop" extension
2. Open `.tex` file
3. Click "Build LaTeX project" button
4. PDF will auto-generate

## Checklist Before Submission

- [ ] All `[YOUR DATA]` placeholders filled in
- [ ] All `[SPECIFY]` placeholders completed
- [ ] Choose ONE interpretation (maintained vs degraded performance) and delete the other
- [ ] Update Table 2 (Cohort comparison) completely
- [ ] Update Table 3 (External validation results) completely
- [ ] Fill in supplementary table references
- [ ] Update author list and affiliations
- [ ] Add IRB approval number
- [ ] Update funding sources
- [ ] Complete Mateus et al. reference
- [ ] Proofread for any remaining brackets `[]`

## Generating Supplementary Tables

The CLAIM checklist (supplementary_table_s1_claim.tex) is already complete!
You just need to compile it:

```bash
pdflatex supplementary_table_s1_claim.tex
```

For TRIPOD-AI checklist (Supplementary Table S2), follow the same format but use TRIPOD-AI items from:
https://www.tripod-statement.org/

## Need Help?

If you get compilation errors:
1. Make sure all LaTeX packages are installed
2. Check that you didn't accidentally delete any `\begin{...}` or `\end{...}` tags
3. Look for unmatched brackets or braces
4. The error message will tell you which line number has the problem

## Customization Priority

**HIGH PRIORITY (Must do before submission):**
1. Abstract results (Lines 64, 69)
2. Table 2 - Cohort comparison (Lines 423-448)
3. Table 3 - External validation results (Lines 679-707)
4. Choose and customize ONE interpretation section (Lines 711-760)
5. Ethics IRB number (Line 1271)

**MEDIUM PRIORITY:**
1. Hardware specs (Line 251)
2. CMC dataset details (Lines 392-401)
3. Funding (Line 1283)
4. Author contributions (Line 1277)

**LOW PRIORITY (Nice to have):**
1. Supplementary figure references
2. Detailed acknowledgments
3. Additional citations

## Estimated Time to Complete Customization

- Quick version (minimal customization): 2-3 hours
- Complete version (all details): 1 day
- With careful proofreading: 2 days

## Word Count

Current manuscript: ~12,000 words (without references)
With your data filled in: ~13,000-14,000 words
Target for most journals: 4,000-6,000 words for main text

**You may need to shorten for journal submission** - but this version is complete for:
- Preprint servers (arXiv, medRxiv)
- Thesis chapter
- Technical report

For journal submission, you can move extensive methods details and tables to supplementary materials.

## Questions?

This paper provides a complete, publication-ready framework. All the difficult writing is done - you just need to insert your actual experimental results and customize specific details about your institution and dataset.

The structure follows best practices for:
- Medical imaging AI papers
- Reproducibility studies
- External validation studies

It's ready for submission to journals like:
- Radiology: Artificial Intelligence
- Medical Physics
- Medical Image Analysis  
- Journal of Medical Imaging
- PLOS ONE (Medical Informatics)