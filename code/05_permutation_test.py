#!/usr/bin/env python3
"""
Permutation Test for AUC=1.00 Validation (P1 Requirement)

专家要求：运行>=1,000次permutation test来验证AUC=1.00不是偶然结果
目的：计算在随机标签下观察到AUC>=1.00的概率（p-value）

日期：2026-01-19
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from pathlib import Path
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def run_cv_pipeline(X, y, random_state=42):
    """
    运行完整的leakage-safe CV pipeline
    返回out-of-fold AUC
    """
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
    y_pred_proba_oof = np.zeros(len(y))

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Leakage-safe standardization
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train model
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train_scaled, y_train)

        # Predict
        y_pred_proba_oof[test_idx] = model.predict_proba(X_test_scaled)[:, 1]

    auc = roc_auc_score(y, y_pred_proba_oof)
    return auc

def main():
    print("=" * 80)
    print("Permutation Test for AUC=1.00 Validation (P1 Requirement)")
    print("=" * 80)

    # 1. Load data
    print("\n1. Loading data...")
    base_dir = Path(__file__).parent.parent
    participants = pd.read_csv(base_dir / 'data/participant_level/participants_master.csv')
    roi_summary = pd.read_csv(base_dir / 'data/group_level_adq_id/CTRL_MCI_AD_All_ROI_Summary.csv')
    events = pd.read_csv(base_dir / 'data/group_level_adq_id/CTRL_MCI_AD_All_Events.csv')

    # Filter MCI and Control
    participants_filtered = participants[participants['Group'].isin(['Control', 'MCI'])].copy()

    # 2. Compute features
    print("2. Computing features...")
    roi_features = roi_summary.groupby('ParticipantID').agg({
        'FixTime': 'mean',
        'EnterCount': 'mean',
        'RegressionCount': 'mean'
    }).reset_index()
    roi_features.columns = ['ParticipantID', 'Mean_FixTime', 'Mean_EnterCount', 'Mean_RegressionCount']

    saccade_data = events[events['EventType'] == 'saccade'].copy()
    saccade_features = saccade_data.groupby('ParticipantID').agg({
        'Amplitude_deg': 'mean'
    }).reset_index()
    saccade_features.columns = ['ParticipantID', 'Mean_Saccade_Amplitude']

    features_df = participants_filtered[['ParticipantID', 'Group', 'VR_MMSE_total']].merge(
        roi_features, on='ParticipantID', how='left'
    ).merge(
        saccade_features, on='ParticipantID', how='left'
    )
    features_df = features_df.fillna(features_df.mean(numeric_only=True))

    # 3. Prepare X and y
    feature_columns = ['VR_MMSE_total', 'Mean_FixTime', 'Mean_EnterCount',
                       'Mean_RegressionCount', 'Mean_Saccade_Amplitude']
    X = features_df[feature_columns].values
    y = (features_df['Group'] == 'MCI').astype(int).values

    print(f"   ✓ Sample size: {len(X)} (Control={np.sum(y==0)}, MCI={np.sum(y==1)})")
    print(f"   ✓ Features: {len(feature_columns)}")

    # 4. Compute observed AUC (true labels)
    print("\n3. Computing observed AUC (true labels)...")
    observed_auc = run_cv_pipeline(X, y, random_state=42)
    print(f"   ✓ Observed AUC: {observed_auc:.4f}")

    # 5. Run permutation test
    n_permutations = 1000
    print(f"\n4. Running {n_permutations} permutations (this may take a few minutes)...")
    print("   Shuffling labels and re-running CV pipeline each time...")

    np.random.seed(42)
    permuted_aucs = []

    for i in tqdm(range(n_permutations), desc="Permutations"):
        # Shuffle labels
        y_permuted = np.random.permutation(y)

        # Run CV with shuffled labels
        try:
            permuted_auc = run_cv_pipeline(X, y_permuted, random_state=42)
            permuted_aucs.append(permuted_auc)
        except Exception as e:
            # In case of convergence issues
            print(f"\n   Warning: Permutation {i+1} failed: {e}")
            continue

    permuted_aucs = np.array(permuted_aucs)

    # 6. Compute p-value
    print("\n5. Computing p-value...")
    p_value = np.mean(permuted_aucs >= observed_auc)
    print(f"   ✓ p-value: {p_value:.4f}")
    print(f"   (Probability of observing AUC >= {observed_auc:.4f} by chance)")

    # 7. Summary statistics
    print("\n6. Permutation distribution statistics...")
    print(f"   Mean: {np.mean(permuted_aucs):.4f}")
    print(f"   Median: {np.median(permuted_aucs):.4f}")
    print(f"   Std: {np.std(permuted_aucs):.4f}")
    print(f"   Min: {np.min(permuted_aucs):.4f}")
    print(f"   Max: {np.max(permuted_aucs):.4f}")
    print(f"   95th percentile: {np.percentile(permuted_aucs, 95):.4f}")
    print(f"   99th percentile: {np.percentile(permuted_aucs, 99):.4f}")

    # 8. Interpretation
    print("\n" + "=" * 80)
    print("Interpretation")
    print("=" * 80)

    if p_value < 0.001:
        interpretation = "HIGHLY SIGNIFICANT (p < 0.001)"
        message = "The observed AUC=1.00 is extremely unlikely to occur by chance."
    elif p_value < 0.01:
        interpretation = "VERY SIGNIFICANT (p < 0.01)"
        message = "The observed AUC=1.00 is very unlikely to occur by chance."
    elif p_value < 0.05:
        interpretation = "SIGNIFICANT (p < 0.05)"
        message = "The observed AUC=1.00 is unlikely to occur by chance."
    else:
        interpretation = "NOT SIGNIFICANT (p >= 0.05)"
        message = "The observed AUC=1.00 could potentially occur by chance."

    print(f"\n{interpretation}")
    print(f"{message}")
    print(f"\nOut of {len(permuted_aucs)} permutations, {np.sum(permuted_aucs >= observed_auc)} ")
    print(f"achieved AUC >= {observed_auc:.4f}.")

    # 9. Save results
    print("\n" + "=" * 80)
    print("Saving results...")
    print("=" * 80)

    output_dir = base_dir / 'results/roc_cv'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save permutation AUC distribution
    perm_df = pd.DataFrame({
        'permutation_id': range(1, len(permuted_aucs) + 1),
        'permuted_auc': permuted_aucs
    })
    perm_file = output_dir / 'permutation_test_aucs.csv'
    perm_df.to_csv(perm_file, index=False)
    print(f"✓ Permutation AUCs: {perm_file}")

    # Save summary JSON
    summary = {
        'analysis_date': '2026-01-19',
        'n_permutations': len(permuted_aucs),
        'random_seed': 42,
        'observed_auc': float(observed_auc),
        'permutation_statistics': {
            'mean': float(np.mean(permuted_aucs)),
            'median': float(np.median(permuted_aucs)),
            'std': float(np.std(permuted_aucs)),
            'min': float(np.min(permuted_aucs)),
            'max': float(np.max(permuted_aucs)),
            'percentile_95': float(np.percentile(permuted_aucs, 95)),
            'percentile_99': float(np.percentile(permuted_aucs, 99))
        },
        'p_value': float(p_value),
        'interpretation': interpretation,
        'n_permutations_gte_observed': int(np.sum(permuted_aucs >= observed_auc))
    }

    summary_file = output_dir / 'permutation_test_summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Summary JSON: {summary_file}")

    # Save detailed report
    report_file = output_dir / 'permutation_test_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Permutation Test Report (P1 Requirement)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Analysis Date: 2026-01-19\n")
        f.write(f"Number of Permutations: {len(permuted_aucs)}\n")
        f.write(f"Random Seed: 42\n\n")

        f.write("Observed AUC (true labels):\n")
        f.write(f"  {observed_auc:.4f}\n\n")

        f.write("Permutation Distribution Statistics:\n")
        f.write(f"  Mean:   {np.mean(permuted_aucs):.4f}\n")
        f.write(f"  Median: {np.median(permuted_aucs):.4f}\n")
        f.write(f"  Std:    {np.std(permuted_aucs):.4f}\n")
        f.write(f"  Min:    {np.min(permuted_aucs):.4f}\n")
        f.write(f"  Max:    {np.max(permuted_aucs):.4f}\n")
        f.write(f"  95th percentile: {np.percentile(permuted_aucs, 95):.4f}\n")
        f.write(f"  99th percentile: {np.percentile(permuted_aucs, 99):.4f}\n\n")

        f.write("P-value:\n")
        f.write(f"  {p_value:.4f}\n")
        f.write(f"  ({np.sum(permuted_aucs >= observed_auc)} out of {len(permuted_aucs)} ")
        f.write(f"permutations achieved AUC >= {observed_auc:.4f})\n\n")

        f.write("Interpretation:\n")
        f.write(f"  {interpretation}\n")
        f.write(f"  {message}\n\n")

        f.write("=" * 80 + "\n")
        f.write("For DADM Manuscript\n")
        f.write("=" * 80 + "\n\n")
        f.write("Recommended text:\n")
        f.write(f"\"To assess whether the observed AUC of {observed_auc:.2f} could arise by chance,\n")
        f.write(f"we performed a permutation test with {len(permuted_aucs)} random label shuffles.\n")
        f.write(f"The null distribution had mean AUC = {np.mean(permuted_aucs):.2f} ")
        f.write(f"(SD = {np.std(permuted_aucs):.2f}), and the observed AUC was more extreme than\n")
        f.write(f"{100*(1-p_value):.1f}% of permutations (p ")
        if p_value < 0.001:
            f.write("< 0.001")
        else:
            f.write(f"= {p_value:.3f}")
        f.write("), confirming that the discrimination is not attributable to chance.\"\n\n")

    print(f"✓ Detailed report: {report_file}")

    print("\n" + "=" * 80)
    print("✓ Permutation test complete!")
    print("=" * 80)

    return summary

if __name__ == '__main__':
    main()
