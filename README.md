# VR-CS Eye-Tracking Analysis Code

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18309467.svg)](https://doi.org/10.5281/zenodo.18309467)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

Analysis code and aggregated results for the manuscript:

**"Detecting Early Cognitive Decline Using Eye-Tracking Metrics in a Virtual Reality Cognitive Screening Tool"**

Submitted to: *Diagnosis, Assessment & Disease Monitoring (DADM)*

---

## ⚠️ Important: Data Availability

**This GitHub repository contains CODE ONLY** (analysis scripts, ROI dictionaries, and aggregated group-level summary outputs). It **does NOT contain participant-level data**.

### What's in this repository (Public)

✅ **Analysis code** - All Python scripts to reproduce Tables/Figures and statistical analyses
✅ **ROI dictionaries** - Region of Interest mappings and classification rules
✅ **Aggregated outputs** - Group-level summary statistics required to verify manuscript claims
✅ **Publication figures** - Final figures with group-level visualizations

### What's NOT in this repository (Controlled Access)

❌ **Participant-level datasets** - Demographics, MMSE scores, eye-tracking features (N=60 participants)
❌ **Granular eye-tracking exports** - Task-level, ROI-level, event-level data
❌ **Out-of-fold predictions** - Per-participant cross-validation outputs
❌ **Raw VR recordings** - Original eye-tracking time-series data

**Why?** Our Research Data Management Plan (DMP) and Human Research Ethics approval classify this dataset as **re-identifiable**. Participant-level data requires controlled access via data sharing agreement.

### How to access controlled data

See [REQUEST_ACCESS.md](REQUEST_ACCESS.md) for detailed instructions on requesting participant-level derived datasets.

**Summary**:
1. Submit formal request to Dr. King Hann Lim (glkhann@curtin.edu.my)
2. Provide research proposal + institutional ethics approval
3. Execute data sharing agreement
4. Receive controlled data package from Curtin Research Data Collection

---

## Study Overview

**Cohort**: N=60 (20 Controls, 20 MCI, 20 AD)
**Task**: VR-based cognitive screening tool (VR-CS, 21-point scale)
**Eye-tracking**: Pico 4 Pro head-mounted display with integrated eye tracking (HMD refresh 90 Hz; eye-tracking sampled at 60 Hz; angular accuracy ~0.5°)
**Primary outcome**: MCI vs Controls classification using eye-tracking + cognitive features

**Note on naming**: In code/results, historical variable names such as `VR_MMSE_total` may appear; these correspond to the VR-CS total score (21-point) reported in the manuscript.

### Key Results

**Cross-validated ROC (MCI vs Controls)**:
- **AUC**: 1.00 (95% CI: 1.00–1.00, 10-fold stratified CV)
- **Sensitivity**: 100% (95% CI: 83.2%–100%, Clopper-Pearson exact)
- **Specificity**: 100% (95% CI: 83.2%–100%, Clopper-Pearson exact)

**Validation Evidence**:
- ✅ **Permutation test** (1,000 iterations): p < 0.001
- ✅ **Repeated CV** (100 repeats): Mean AUC = 1.00, SD = 0.00
- ✅ **Confounding baseline**: Demographics-only AUC = 0.44

**Feature Separation**: Two eye-tracking features show perfect group separation due to working memory deficits in MCI (5–6× more ROI entries/regressions). See [figures/feature_separation_focused.png](figures/feature_separation_focused.png).

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

**Requirements**: Python 3.8+, pandas, numpy, scikit-learn, matplotlib, seaborn, scipy

### 2. Reproduce aggregated results (without participant-level data)

You can verify manuscript claims using the **aggregated outputs** provided in `results_public/`:

```bash
# View ROC metrics (aggregated)
cat results_public/metrics.json

# View permutation test results
cat results_public/permutation_test_summary.json

# View repeated CV stability
cat results_public/repeated_cv_summary.json

# View confounding baseline
cat results_public/confounding_baseline_results.json
```

### 3. Run full analysis (requires controlled data)

**Note**: The following scripts require participant-level data (not included in this repository):

```bash
cd code

# Cross-validated ROC analysis
python 04_roc_cv_enhanced.py  # Requires participants_master.csv

# Generate Figure 3
python 05_figures.py  # Requires participant_task_level_metrics.csv

# Validation analyses
python 05_permutation_test.py
python 06_repeated_cv.py
python 07_confounding_baseline.py
python 08_feature_distribution_plot.py
```

To run these scripts, request access to the controlled data package via [REQUEST_ACCESS.md](REQUEST_ACCESS.md).

---

## Repository Structure

```
vr-cs-eye-tracking-analysis/
├── README.md                  # This file
├── REQUEST_ACCESS.md          # How to access controlled data
├── LICENSE                    # MIT License (code)
├── requirements.txt           # Python dependencies
├── .gitignore                 # Prevents accidental commit of sensitive data
├── code/                      # Analysis scripts (PUBLIC)
│   ├── 04_roc_cv.py                      # Basic ROC CV
│   ├── 04_roc_cv_enhanced.py             # Enhanced ROC CV (P0 artifacts)
│   ├── 05_figures.py                     # Generate Figure 3
│   ├── 05_permutation_test.py            # 1,000 permutations
│   ├── 06_repeated_cv.py                 # 100 repeated CVs
│   ├── 07_confounding_baseline.py        # Demographics-only baseline
│   └── 08_feature_distribution_plot.py   # Feature separation viz
├── docs/                      # Documentation (PUBLIC)
│   ├── ROI_Dictionary.txt                # Human-readable ROI descriptions
│   ├── ROI_Mapping.csv                   # Machine-readable ROI types
│   ├── roi_mapping_rules.md              # Classification algorithm
│   └── data_dictionary.csv               # Variable definitions
├── results_public/            # Aggregated outputs (PUBLIC, group-level only)
│   ├── metrics.json                      # ROC metrics (aggregated)
│   ├── software_versions.txt             # Environment snapshot
│   ├── mci_cv_roc_report.txt             # Human-readable ROC report
│   ├── permutation_test_summary.json     # Permutation test (aggregated)
│   ├── repeated_cv_summary.json          # Repeated CV (aggregated)
│   └── confounding_baseline_results.json # Confounding analysis (aggregated)
└── figures/                   # Publication figures (PUBLIC)
    ├── Figure3_Saccade_Amplitude_Publication.png  # Main figure
    ├── Fig3_source_data.csv                       # Group×task summary
    └── feature_separation_focused.png             # Feature separation plot
```

---

## ROI (Region of Interest) Classification

Eye-tracking data are classified into three ROI types:

- **KW (Keywords)**: Target words in each task (e.g., time words, location names)
- **INST (Instructions)**: Task instruction regions
- **BG (Background)**: Background regions (used to detect off-task attention)

See [docs/ROI_Dictionary.txt](docs/ROI_Dictionary.txt) and [docs/roi_mapping_rules.md](docs/roi_mapping_rules.md) for complete documentation.

---

## Reproducibility

### Version for Manuscript Reproduction

**Manuscript results reproduced using:**
- **GitHub Release**: [v2.0.1](https://github.com/HaiweiZuo/vr-cs-eye-tracking-analysis/releases/tag/v2.0.1)
- **Zenodo Package**: [10.5281/zenodo.18309467](https://doi.org/10.5281/zenodo.18309467)

This ensures reviewers can identify the exact code version corresponding to the manuscript.

### Fixed Parameters

- **Random seed**: 42 (for reproducible cross-validation splits)
- **CV method**: 10-fold stratified cross-validation
- **Standardization**: Leakage-safe (StandardScaler fit on training folds only)
- **Confidence intervals**: Clopper-Pearson exact (binomial)

---

## License and Citation

### Code License

**MIT License** - Free for academic and commercial use. See [LICENSE](LICENSE).

### Data License (Aggregated Outputs)

**CC BY 4.0** - Attribution required when reusing aggregated results in `results_public/`.

### Preferred Citation

**For this code repository**:
```
Zuo, H. (2026). VR-CS Eye-Tracking Analysis Code (v2.0) [Software].
GitHub. https://github.com/HaiweiZuo/vr-cs-eye-tracking-analysis
```

**For the Zenodo dataset** (aggregated data + full documentation):
```
Zuo, H. (2026). VR-CS + Eye-Tracking Dataset for Alzheimer's Disease
and MCI Detection (v2.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.18309467
```

**For controlled data** (if obtained via Curtin RDC):
Add to your Data Availability statement:
```
Participant-level derived datasets were obtained from Curtin Research Data
Collection under a data sharing agreement (contact: Dr. King Hann Lim,
glkhann@curtin.edu.my).
```

---

## Related Resources

- 📊 **Zenodo Open Package**: [https://doi.org/10.5281/zenodo.18309467](https://doi.org/10.5281/zenodo.18309467)
  *(Full documentation, aggregated results, pre-generated validation outputs)*
- 🔒 **Controlled Data**: [Curtin Research Data Collection](REQUEST_ACCESS.md)
  *(Participant-level datasets, available upon request)*
- 📄 **Manuscript**: *DADM submission (under review)*

---

## Contact

### For code/technical questions

**Haiwei Zuo**
PhD Candidate
Curtin Malaysia Research Institute, Curtin University Malaysia
Email: haiwei.zuo@postgrad.curtin.edu.my
ORCID: [0009-0009-7008-3028](https://orcid.org/0009-0009-7008-3028)

### For data access requests

**Dr. King Hann Lim**
Corresponding Author
Email: glkhann@curtin.edu.my

---

## Acknowledgments

This research was approved by the Curtin University Human Research Ethics Committee. Eye-tracking data were collected using a Pico 4 Pro head-mounted display with integrated eye tracking. VR environment developed in Unity.

---

**Last updated**: 2026-01-23
**Version**: 2.0
**GitHub**: https://github.com/HaiweiZuo/vr-cs-eye-tracking-analysis
**Zenodo**: https://doi.org/10.5281/zenodo.18309467

---

## Disclaimer

MMSE instrument content (items/forms) is not reproduced in this repository due to licensing/copyright restrictions. The MMSE is used only as an external comparator (total score) for concurrent validity.
