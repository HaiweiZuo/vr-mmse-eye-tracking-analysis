#!/usr/bin/env python3
"""
Enhanced MCI vs Controls Cross-Validated ROC Analysis
Includes all DADM reproducibility requirements (P0 mandatory artifacts)

按照DADM专家建议生成：
- Out-of-fold predictions (oof_predictions.csv)
- CV fold indices and seed (cv_folds_seed_and_indices.csv)
- Metrics JSON (metrics.json)
- Software versions (software_versions.txt)
- Binomial CIs for Sensitivity/Specificity (Clopper-Pearson)

日期：2026-01-19
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
import json
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def clopper_pearson_ci(k, n, alpha=0.05):
    """
    计算Clopper-Pearson精确置信区间（用于二项分布比例）
    k: 成功次数
    n: 总次数
    alpha: 显著性水平（0.05 for 95% CI）
    """
    from scipy import stats

    if k == 0:
        lower = 0.0
    else:
        lower = stats.beta.ppf(alpha/2, k, n-k+1)

    if k == n:
        upper = 1.0
    else:
        upper = stats.beta.ppf(1-alpha/2, k+1, n-k)

    return lower, upper

def bootstrap_ci(y_true, y_pred_proba, n_bootstrap=1000, ci=95):
    """计算AUC的bootstrap置信区间"""
    np.random.seed(42)
    aucs = []
    n_samples = len(y_true)

    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, n_samples, replace=True)
        if len(np.unique(y_true[indices])) < 2:
            continue
        auc = roc_auc_score(y_true[indices], y_pred_proba[indices])
        aucs.append(auc)

    alpha = (100 - ci) / 2
    ci_lower = np.percentile(aucs, alpha)
    ci_upper = np.percentile(aucs, 100 - alpha)

    return ci_lower, ci_upper

def calculate_youden_index(y_true, y_pred_proba):
    """计算Youden index并返回最佳阈值下的敏感度和特异度"""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    youden_index = tpr - fpr
    best_idx = np.argmax(youden_index)

    best_threshold = thresholds[best_idx]
    best_sensitivity = tpr[best_idx]
    best_specificity = 1 - fpr[best_idx]

    return best_sensitivity, best_specificity, best_threshold

def save_software_versions(output_dir):
    """保存软件版本信息（P0要求）"""
    import platform
    import sklearn
    import scipy

    versions = {
        'Python': sys.version,
        'Platform': platform.platform(),
        'NumPy': np.__version__,
        'Pandas': pd.__version__,
        'Scikit-learn': sklearn.__version__,
        'SciPy': scipy.__version__,
    }

    version_file = output_dir / 'software_versions.txt'
    with open(version_file, 'w', encoding='utf-8') as f:
        f.write("Software Versions for Reproducibility\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: 2026-01-19\n\n")
        for key, value in versions.items():
            f.write(f"{key}: {value}\n")

    print(f"✓ Software versions saved: {version_file}")
    return version_file

def main():
    print("=" * 80)
    print("Enhanced MCI vs Controls - Cross-Validated ROC Analysis")
    print("With DADM P0 Reproducibility Artifacts")
    print("=" * 80)

    # 1. 读取数据
    print("\n1. Loading data...")
    base_dir = Path(__file__).parent.parent
    participants = pd.read_csv(base_dir / 'data/participant_level/participants_master.csv')
    roi_summary = pd.read_csv(base_dir / 'data/group_level_adq_id/CTRL_MCI_AD_All_ROI_Summary.csv')
    events = pd.read_csv(base_dir / 'data/group_level_adq_id/CTRL_MCI_AD_All_Events.csv')

    # 只选择MCI和Control组
    participants_filtered = participants[participants['Group'].isin(['Control', 'MCI'])].copy()
    print(f"   ✓ Control: {(participants_filtered['Group'] == 'Control').sum()} participants")
    print(f"   ✓ MCI: {(participants_filtered['Group'] == 'MCI').sum()} participants")

    # 2. 计算participant-level特征
    print("\n2. Computing participant-level features...")

    # ROI特征
    roi_features = roi_summary.groupby('ParticipantID').agg({
        'FixTime': 'mean',
        'EnterCount': 'mean',
        'RegressionCount': 'mean'
    }).reset_index()
    roi_features.columns = ['ParticipantID', 'Mean_FixTime', 'Mean_EnterCount', 'Mean_RegressionCount']

    # Saccade特征
    saccade_data = events[events['EventType'] == 'saccade'].copy()
    saccade_features = saccade_data.groupby('ParticipantID').agg({
        'Amplitude_deg': 'mean'
    }).reset_index()
    saccade_features.columns = ['ParticipantID', 'Mean_Saccade_Amplitude']

    # 合并特征
    features_df = participants_filtered[['ParticipantID', 'Group', 'VR_MMSE_total']].merge(
        roi_features, on='ParticipantID', how='left'
    ).merge(
        saccade_features, on='ParticipantID', how='left'
    )

    features_df = features_df.fillna(features_df.mean(numeric_only=True))
    print(f"   ✓ Features: 5 (VR_MMSE_total + 4 eye-tracking features)")

    # 3. 准备X和y
    feature_columns = ['VR_MMSE_total', 'Mean_FixTime', 'Mean_EnterCount',
                       'Mean_RegressionCount', 'Mean_Saccade_Amplitude']

    X = features_df[feature_columns].values
    y = (features_df['Group'] == 'MCI').astype(int).values  # MCI=1, Control=0
    participant_ids = features_df['ParticipantID'].values

    print(f"   ✓ Sample size: {len(X)}")
    print(f"   ✓ Feature columns: {feature_columns}")

    # 4. 10-Fold Cross-Validation with leakage-safe standardization
    print("\n3. Running 10-Fold Cross-Validation (leakage-safe)...")
    print("   Strategy: StandardScaler fit on training folds only, then applied to test folds")

    cv_random_state = 42
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=cv_random_state)

    # 存储fold信息和predictions
    y_pred_proba_oof = np.zeros(len(y))
    fold_info = []
    fold_aucs = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Leakage-safe标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 训练模型
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train_scaled, y_train)

        # 预测测试集
        y_pred_proba_fold = model.predict_proba(X_test_scaled)[:, 1]
        y_pred_proba_oof[test_idx] = y_pred_proba_fold

        # 计算fold AUC
        fold_auc = roc_auc_score(y_test, y_pred_proba_fold)
        fold_aucs.append(fold_auc)

        # 记录fold信息
        for idx in test_idx:
            fold_info.append({
                'fold': fold_idx,
                'participant_id': participant_ids[idx],
                'index_in_dataset': int(idx),
                'train_or_test': 'test'
            })

        print(f"   Fold {fold_idx:2d}: AUC = {fold_auc:.4f} "
              f"(train={len(train_idx)}, test={len(test_idx)})")

    # 5. 计算总体out-of-fold AUC
    print("\n4. Computing Out-of-Fold AUC...")
    cv_auc = roc_auc_score(y, y_pred_proba_oof)
    print(f"   ✓ Cross-Validated AUC: {cv_auc:.4f}")
    print(f"   ✓ Mean Fold AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")

    # 6. 计算95% CI (bootstrap)
    print("\n5. Computing 95% Confidence Interval (Bootstrap)...")
    ci_lower, ci_upper = bootstrap_ci(y, y_pred_proba_oof, n_bootstrap=1000, ci=95)
    print(f"   ✓ 95% CI: [{ci_lower:.2f}–{ci_upper:.2f}]")

    # 7. 计算Youden index下的敏感度和特异度
    print("\n6. Computing Sensitivity and Specificity at Youden Index...")
    sensitivity, specificity, threshold = calculate_youden_index(y, y_pred_proba_oof)

    # 计算实际分类结果
    y_pred = (y_pred_proba_oof >= threshold).astype(int)

    # 计算TP, TN, FP, FN
    tp = np.sum((y == 1) & (y_pred == 1))  # True Positives (MCI正确识别为MCI)
    tn = np.sum((y == 0) & (y_pred == 0))  # True Negatives (Control正确识别为Control)
    fp = np.sum((y == 0) & (y_pred == 1))  # False Positives (Control误识别为MCI)
    fn = np.sum((y == 1) & (y_pred == 0))  # False Negatives (MCI误识别为Control)

    n_mci = np.sum(y == 1)
    n_control = np.sum(y == 0)

    print(f"   ✓ Optimal threshold: {threshold:.4f}")
    print(f"   ✓ Sensitivity: {sensitivity:.2%} (TP={tp}/{n_mci})")
    print(f"   ✓ Specificity: {specificity:.2%} (TN={tn}/{n_control})")
    print(f"   ✓ Youden Index: {sensitivity + specificity - 1:.4f}")

    # 8. 计算Sensitivity和Specificity的Clopper-Pearson置信区间（P0要求）
    print("\n7. Computing Binomial Confidence Intervals (Clopper-Pearson)...")

    sens_ci_lower, sens_ci_upper = clopper_pearson_ci(tp, n_mci, alpha=0.05)
    spec_ci_lower, spec_ci_upper = clopper_pearson_ci(tn, n_control, alpha=0.05)

    print(f"   ✓ Sensitivity 95% CI: [{sens_ci_lower:.3f}–{sens_ci_upper:.3f}]")
    print(f"   ✓ Specificity 95% CI: [{spec_ci_lower:.3f}–{spec_ci_upper:.3f}]")
    print(f"   Note: For {tp}/{n_mci} successes, lower bound ≈ {sens_ci_lower:.3f}")

    # 9. 保存输出文件（P0要求）
    print("\n" + "=" * 80)
    print("Saving P0 Mandatory Artifacts...")
    print("=" * 80)

    output_dir = base_dir / 'results/roc_cv'
    output_dir.mkdir(parents=True, exist_ok=True)

    # P0-1: Out-of-fold predictions
    oof_df = pd.DataFrame({
        'ParticipantID': participant_ids,
        'True_Label': y,
        'Predicted_Probability': y_pred_proba_oof,
        'Predicted_Class': y_pred,
        'Group': features_df['Group'].values
    })
    oof_file = output_dir / 'oof_predictions.csv'
    oof_df.to_csv(oof_file, index=False)
    print(f"✓ [P0-1] Out-of-fold predictions: {oof_file}")

    # P0-2: CV fold indices and seed
    fold_df = pd.DataFrame(fold_info)
    fold_metadata = pd.DataFrame([{
        'cv_strategy': '10-fold StratifiedKFold',
        'random_state': cv_random_state,
        'shuffle': True,
        'n_splits': 10
    }])
    fold_file = output_dir / 'cv_folds_seed_and_indices.csv'
    with open(fold_file, 'w') as f:
        f.write("# CV Metadata\n")
        fold_metadata.to_csv(f, index=False)
        f.write("\n# Fold Assignments\n")
        fold_df.to_csv(f, index=False)
    print(f"✓ [P0-2] CV fold indices and seed: {fold_file}")

    # P0-3: Metrics JSON
    metrics = {
        'analysis_date': '2026-01-19',
        'method': '10-fold Stratified Cross-Validation',
        'standardization': 'Leakage-safe (within-fold, training-only fit)',
        'sample_size': {
            'total': int(len(y)),
            'control': int(n_control),
            'mci': int(n_mci)
        },
        'features': feature_columns,
        'cv_random_state': int(cv_random_state),
        'results': {
            'cv_auc': float(cv_auc),
            'auc_ci_95': {
                'lower': float(ci_lower),
                'upper': float(ci_upper)
            },
            'mean_fold_auc': float(np.mean(fold_aucs)),
            'std_fold_auc': float(np.std(fold_aucs)),
            'fold_aucs': [float(x) for x in fold_aucs],
            'optimal_threshold': float(threshold),
            'sensitivity': float(sensitivity),
            'sensitivity_ci_95': {
                'lower': float(sens_ci_lower),
                'upper': float(sens_ci_upper)
            },
            'specificity': float(specificity),
            'specificity_ci_95': {
                'lower': float(spec_ci_lower),
                'upper': float(spec_ci_upper)
            },
            'youden_index': float(sensitivity + specificity - 1),
            'confusion_matrix': {
                'true_positives': int(tp),
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn)
            }
        }
    }

    metrics_file = output_dir / 'metrics.json'
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"✓ [P0-3] Metrics JSON: {metrics_file}")

    # P0-4: Software versions
    version_file = save_software_versions(output_dir)
    print(f"✓ [P0-4] Software versions: {version_file}")

    # 10. 生成详细报告
    print("\n" + "=" * 80)
    print("Generating Detailed Report...")
    print("=" * 80)

    report_file = output_dir / 'mci_cv_roc_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("MCI vs Controls - Enhanced Cross-Validated ROC Analysis Report\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Analysis Date: 2026-01-19\n")
        f.write(f"Method: 10-Fold Stratified Cross-Validation\n")
        f.write(f"Standardization: Leakage-safe (within-fold, training-only fit)\n")
        f.write(f"Random Seed: {cv_random_state}\n\n")

        f.write("Sample Size:\n")
        f.write(f"  Control: {n_control} participants\n")
        f.write(f"  MCI: {n_mci} participants\n")
        f.write(f"  Total: {len(y)} participants\n\n")

        f.write("Features Used:\n")
        for i, feat in enumerate(feature_columns, 1):
            f.write(f"  {i}. {feat}\n")
        f.write("\n")

        f.write("Cross-Validation Results:\n")
        f.write(f"  Cross-validated AUC: {cv_auc:.4f}\n")
        f.write(f"  95% CI (Bootstrap): [{ci_lower:.2f}–{ci_upper:.2f}]\n")
        f.write(f"  Mean Fold AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}\n\n")

        f.write("Fold-by-Fold AUCs:\n")
        for i, auc in enumerate(fold_aucs, 1):
            f.write(f"  Fold {i:2d}: {auc:.4f}\n")
        f.write("\n")

        f.write("Optimal Classification Threshold (Youden Index):\n")
        f.write(f"  Threshold: {threshold:.4f}\n")
        f.write(f"  Sensitivity: {sensitivity:.4f} ({sensitivity:.1%})\n")
        f.write(f"  Sensitivity 95% CI (Clopper-Pearson): [{sens_ci_lower:.3f}–{sens_ci_upper:.3f}]\n")
        f.write(f"  Specificity: {specificity:.4f} ({specificity:.1%})\n")
        f.write(f"  Specificity 95% CI (Clopper-Pearson): [{spec_ci_lower:.3f}–{spec_ci_upper:.3f}]\n")
        f.write(f"  Youden Index: {sensitivity + specificity - 1:.4f}\n\n")

        f.write("Confusion Matrix:\n")
        f.write(f"  True Positives (TP): {tp} / {n_mci}\n")
        f.write(f"  True Negatives (TN): {tn} / {n_control}\n")
        f.write(f"  False Positives (FP): {fp}\n")
        f.write(f"  False Negatives (FN): {fn}\n\n")

        f.write("=" * 80 + "\n")
        f.write("For DADM Manuscript (Copy-Paste Values)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"AUC: {cv_auc:.2f} (95% CI: {ci_lower:.2f}–{ci_upper:.2f})\n")
        f.write(f"Sensitivity: {sensitivity:.1%} (95% CI: {sens_ci_lower:.1%}–{sens_ci_upper:.1%})\n")
        f.write(f"Specificity: {specificity:.1%} (95% CI: {spec_ci_lower:.1%}–{spec_ci_upper:.1%})\n\n")

        f.write("DADM-Style Reporting (recommended by expert):\n")
        f.write(f"\"In this internally validated cohort, participant-level ROI entry and\n")
        f.write(f"regression measures showed non-overlapping ranges between MCI and controls,\n")
        f.write(f"yielding leakage-safe 10-fold cross-validated discrimination of AUC={cv_auc:.2f}\n")
        f.write(f"(95% CI {ci_lower:.2f}–{ci_upper:.2f}), with sensitivity of {sensitivity:.1%}\n")
        f.write(f"(95% CI {sens_ci_lower:.1%}–{sens_ci_upper:.1%}) and specificity of {specificity:.1%}\n")
        f.write(f"(95% CI {spec_ci_lower:.1%}–{spec_ci_upper:.1%}). Given the modest sample size\n")
        f.write(f"and clinically well-characterized recruitment, these estimates represent\n")
        f.write(f"internal validation and may overestimate real-world performance; external\n")
        f.write(f"validation in larger, more heterogeneous cohorts is required.\"\n\n")

        f.write("P0 Artifacts Generated:\n")
        f.write(f"  ✓ oof_predictions.csv\n")
        f.write(f"  ✓ cv_folds_seed_and_indices.csv\n")
        f.write(f"  ✓ metrics.json\n")
        f.write(f"  ✓ software_versions.txt\n\n")

    print(f"✓ Detailed report: {report_file}")

    print("\n" + "=" * 80)
    print("✓ Analysis Complete!")
    print("=" * 80)
    print("\nAll P0 mandatory artifacts have been generated:")
    print(f"  1. {oof_file.name}")
    print(f"  2. {fold_file.name}")
    print(f"  3. {metrics_file.name}")
    print(f"  4. {version_file.name}")

    return metrics

if __name__ == '__main__':
    main()
