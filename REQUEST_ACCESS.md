# Requesting Access to Controlled Data

**Package**: VR-CS Eye-Tracking Dataset (Controlled Access Layer)
**Repository**: Curtin Research Data Collection
**Contact**: Dr. King Hann Lim (glkhann@curtin.edu.my)

---

## What data requires controlled access?

The **Controlled-access package** includes participant-level derived datasets and granular task×ROI eye-tracking exports that are not publicly available due to ethics and privacy constraints:

### Participant-level datasets (3 files)
- `data/participant_level/participants_master.csv` - Demographics, MMSE scores, aggregated eye-tracking features (N=60)
- `data/participant_level/participant_task_level_metrics.csv` - Task-level metrics (N=300, 60 participants × 5 tasks)
- `docs/cohort_and_exclusion_log.csv` - Cohort summary and exclusion tracking

### Group-level datasets with ADQ_ID identifiers (8 files)
- `data/group_level_adq_id/CTRL_All_Folders_ROI_Summary.csv`
- `data/group_level_adq_id/MCI_All_Folders_ROI_Summary.csv`
- `data/group_level_adq_id/AD_All_Folders_ROI_Summary.csv`
- `data/group_level_adq_id/CTRL_All_Folders_Events.csv`
- `data/group_level_adq_id/MCI_All_Folders_Events.csv`
- `data/group_level_adq_id/AD_All_Folders_Events.csv`
- `data/group_level_adq_id/CTRL_MCI_AD_All_Events.csv`
- `data/group_level_adq_id/CTRL_MCI_AD_All_ROI_Summary.csv`

### Cross-validation artifacts (3 files)
- `results/roc_cv/oof_predictions.csv` - Out-of-fold predictions for all 40 MCI/Control participants
- `results/roc_cv/cv_folds_seed_and_indices.csv` - Fold assignments and random seed
- `results/roc_cv/mci_cv_roc_results.csv` - Per-participant CV results

---

## Why is controlled access required?

Our Research Data Management Plan (DMP) and Human Research Ethics approval classify this dataset as **re-identifiable**. Key constraints include:

1. **Re-identification risk**: Combination of demographics (age, sex, education), cognitive scores, and behavioral patterns could potentially re-identify participants in small cohorts.
2. **Geographic restriction**: Our DMP states that human participant information is not to be sent overseas without explicit approval.
3. **Ethics approval scope**: Public sharing was not covered in the original ethics protocol; controlled access with data sharing agreements is the approved mechanism.

---

## How to request access

### Step 1: Prepare your request

Your request should include:

1. **Research proposal** (1-2 pages):
   - Research question and objectives
   - Why you need participant-level data (vs. aggregated summaries in the Open package)
   - Planned analyses and methods
   - Expected outputs (publications, theses, etc.)

2. **Institutional affiliation**:
   - University/research institution name
   - Position/role (faculty, postdoc, PhD student, etc.)
   - Supervisor contact (if applicable)

3. **Ethics documentation**:
   - Proof of institutional ethics approval for secondary data analysis, OR
   - Commitment to obtain ethics approval upon data access

4. **Data management plan**:
   - How the data will be stored and secured
   - Who will have access at your institution
   - Data retention and disposal plan
   - Commitment to not attempt re-identification

### Step 2: Submit formal request

Send your request to:

**Dr. King Hann Lim** (Corresponding Author)
Email: glkhann@curtin.edu.my
Subject: "VR-CS Controlled Data Access Request"

### Step 3: Execute data sharing agreement

If your request is approved, you will be required to sign a **Data Sharing Agreement** that includes:

- Permitted use (research purposes only, aligned with your proposal)
- Prohibited use (no re-identification attempts, no redistribution)
- Data security requirements (encrypted storage, access controls)
- Publication acknowledgment (cite the Zenodo Open package and controlled data source)
- Data destruction timeline (typically upon project completion or after a defined retention period)

### Step 4: Receive controlled data

Upon execution of the agreement, you will receive:

- `VR_MMSE_ControlledAccess_v2_0.zip` (566 KB, 15 files)
- MANIFEST.tsv with SHA256 checksums for file integrity verification
- Technical documentation on data structure and variable definitions

---

## Typical review timeline

- **Initial review**: 2-4 weeks from submission
- **Agreement execution**: 1-2 weeks (depending on institutional processes)
- **Data delivery**: Within 1 week after signed agreement

---

## What's included in the Open package (no access request needed)?

If you only need to verify manuscript claims or reproduce aggregated analyses, the **Zenodo Open package** provides:

- ✅ Analysis code (all Python scripts)
- ✅ ROI dictionaries and mapping rules
- ✅ Aggregated group-level summaries (`results/figures/Fig3_source_data.csv`)
- ✅ Pre-generated validation results (permutation tests, repeated CV, confounding baselines, feature visualizations)
- ✅ Aggregated metrics and reports (`results/roc_cv/metrics.json`, `mci_cv_roc_report.txt`, `software_versions.txt`)

Most manuscript verification and code review can be performed using only the Open package.

---

## Frequently Asked Questions

### Q1: Can I access the raw VR recordings or eye-tracking video?

**No.** Raw VR session recordings and eye-tracking videos are not available for sharing due to high re-identification risk. Only derived eye-tracking metrics (fixation durations, saccade amplitudes, ROI entries, etc.) are available in the Controlled package.

### Q2: Can I access data for teaching purposes?

**Maybe.** Educational use requires a separate data sharing agreement. Contact Dr. Lim with details of your course, institution, student access controls, and data security plan.

### Q3: Do I need ethics approval from my institution?

**Yes, typically.** Most institutions require ethics approval for secondary analysis of human participant data, even if de-identified. Check with your institutional review board.

### Q4: Can I share the controlled data with collaborators?

**No, not without explicit permission.** The data sharing agreement is institution-specific. Collaborators at other institutions must submit separate access requests.

### Q5: How should I cite the controlled data?

Cite both:

1. **Zenodo Open package** (for code and aggregated results):
   - See `CITATION.cff` in the Open package for BibTeX

2. **Controlled data source** in your Data Availability statement:
   - "Participant-level derived datasets were obtained from Curtin Research Data Collection under a data sharing agreement (contact: Dr. King Hann Lim, glkhann@curtin.edu.my)."

### Q6: What if my request is denied?

Requests may be denied if:
- Research purpose is not aligned with ethics approval scope
- Insufficient data security plan
- Lack of institutional ethics approval or commitment to obtain it
- Request from commercial entities without clear academic collaboration

If denied, you may revise and resubmit with additional documentation.

---

## Contact

For questions about the access request process:

**Dr. King Hann Lim**
Corresponding Author
Email: glkhann@curtin.edu.my
Affiliation: Curtin Malaysia Research Institute, Curtin University Malaysia

For technical questions about the Open package:

**Haiwei Zuo**
PhD Candidate
Email: haiwei.zuo@postgrad.curtin.edu.my
ORCID: 0009-0009-7008-3028

---

**Last updated**: 2026-01-20
**Package version**: 2.0
