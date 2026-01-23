# ROI Mapping Rules for VR-CS Eye-Tracking Study

**Version**: 1.0
**Date**: 2026-01-16

## Overview

This document describes the rules used to classify Regions of Interest (ROIs) into three types: **KW** (Keywords), **INST** (Instructions), and **BG** (Background). These classifications are critical for understanding the participant-level metrics reported in the manuscript.

## ROI Type Definitions

### 1. KW (Keywords)
**Definition**: Task-relevant target words or phrases that participants need to identify or remember.

**Examples**:
- **Q1 (Orientation to Time)**: Time-related words (e.g., "year", "month", "day", "date")
- **Q2 (Orientation to Place)**: Location names (e.g., "country", "province", "city", "hospital")
- **Q3 (Registration)**: Objects to be remembered (e.g., "apple", "table", "coin")
- **Q4 (Attention & Calculation)**: Numbers in calculation tasks
- **Q5 (Recall)**: Objects from Q3 presented again

**Naming pattern**: `KW_[task]_[index]` (e.g., `KW_n2q1_1`, `KW_n2q2_3`)

### 2. INST (Instructions)
**Definition**: Text regions containing task instructions or operation prompts.

**Examples**:
- "Please select the current year"
- "Click on the objects you remember"
- "Count backwards from 100 by 7s"

**Naming pattern**: `INST_[task]_[index]` (e.g., `INST_n2q1_1`, `INST_n2q3_2`)

**Subvariants**:
- `INST_[task]_1a`, `INST_[task]_1b`: Multiple instruction panels for same task
- `INST_[task]_2a`, `INST_[task]_2b`: Secondary instruction sets

### 3. BG (Background)
**Definition**: Non-task-related regions used to detect off-task attention or visual exploration outside target areas.

**Examples**:
- Empty screen areas
- Decorative elements
- Non-interactive VR environment components

**Naming pattern**: `BG_[task]` (e.g., `BG_n2q1`, `BG_n2q2`)

**Purpose**: High fixation time in BG regions may indicate:
- Distraction
- Task difficulty
- Cognitive load
- Visual search strategy

## Classification Algorithm

The ROI type is determined by parsing the ROI string:

```python
def classify_roi_type(roi_string):
    if roi_string.startswith('KW_'):
        return 'KW'
    elif roi_string.startswith('INST_'):
        return 'INST'
    elif roi_string.startswith('BG_'):
        return 'BG'
    else:
        return 'Unknown'
```

## Task-specific ROI Counts

| Task | KW ROIs | INST ROIs | BG ROIs | Total |
|------|---------|-----------|---------|-------|
| Q1   | 4-5     | 1-2       | 1       | 6-8   |
| Q2   | 4-5     | 1-2       | 1       | 6-8   |
| Q3   | 3       | 2-3       | 1       | 6-7   |
| Q4   | 1-2     | 2-4       | 1       | 4-7   |
| Q5   | 3       | 2-3       | 1       | 6-7   |

**Note**: Exact counts vary by participant based on which task components were presented.

## Data Aggregation Strategy

### For manuscript Tables and Figures:

1. **ROI-level → Participant-level aggregation**:
   - For each participant, calculate mean FixTime, EnterCount, RegressionCount across all ROIs within each task
   - This produces participant×task-level metrics (N=300 rows for 60 participants × 5 tasks)

2. **Task-level → Participant-level aggregation**:
   - For ROC models, further aggregate across tasks to produce single values per participant
   - Mean_FixTime = mean(FixTime across all ROIs and all tasks for this participant)

### Why this matters:

- **Avoid pseudoreplication**: Statistical inference must be at participant level (N=60), not ROI level (N=~1,600)
- **ROI types are used for**: Feature interpretation and sensitivity analyses (not shown in main manuscript)

## Validation Rules

To verify correct ROI type assignment:

1. **Coverage**: Every ROI string in group-level files must map to exactly one type (KW/INST/BG)
2. **Consistency**: Same ROI string always maps to same type across all participants
3. **Completeness**: No unclassified ROI strings (checked by `roi_mapping.csv`)

## Reference Files

- **Machine-readable mapping**: `roi_mapping.csv`
- **Human-readable descriptions**: `roi_dictionary.txt`

## Questions?

If you encounter an ROI string not covered by these rules, please refer to `roi_mapping.csv` for the complete lookup table.

---

**Last updated**: 2026-01-16
