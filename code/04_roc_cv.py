#!/usr/bin/env python3
"""
计算MCI vs Controls的交叉验证ROC分析
使用10-fold cross-validation with leakage-safe standardization
日期：2026-01-15
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

def bootstrap_ci(y_true, y_pred_proba, n_bootstrap=1000, ci=95):
    """计算AUC的bootstrap置信区间"""
    np.random.seed(42)
    aucs = []
    n_samples = len(y_true)

    for _ in range(n_bootstrap):
        # Bootstrap重采样
        indices = np.random.choice(n_samples, n_samples, replace=True)
        if len(np.unique(y_true[indices])) < 2:
            continue
        auc = roc_auc_score(y_true[indices], y_pred_proba[indices])
        aucs.append(auc)

    # 计算置信区间
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

def main():
    print("=" * 80)
    print("MCI vs Controls - 10-Fold Cross-Validated ROC Analysis")
    print("=" * 80)

    # 1. 读取数据
    print("\n1. 读取数据...")
    from pathlib import Path
    base_dir = Path(__file__).parent.parent
    participants = pd.read_csv(base_dir / 'data/participant_level/participants_master.csv')
    roi_summary = pd.read_csv(base_dir / 'data/group_level_adq_id/CTRL_MCI_AD_All_ROI_Summary.csv')
    events = pd.read_csv(base_dir / 'data/group_level_adq_id/CTRL_MCI_AD_All_Events.csv')

    # 只选择MCI和Control组
    participants_filtered = participants[participants['Group'].isin(['Control', 'MCI'])].copy()
    print(f"   ✓ Control组: {(participants_filtered['Group'] == 'Control').sum()}人")
    print(f"   ✓ MCI组: {(participants_filtered['Group'] == 'MCI').sum()}人")

    # 2. 计算participant-level特征（与AD分析保持一致）
    print("\n2. 计算participant-level特征...")

    # ROI特征（FixTime, EnterCount, RegressionCount的均值）
    roi_features = roi_summary.groupby('ParticipantID').agg({
        'FixTime': 'mean',
        'EnterCount': 'mean',
        'RegressionCount': 'mean'
    }).reset_index()
    roi_features.columns = ['ParticipantID', 'Mean_FixTime', 'Mean_EnterCount', 'Mean_RegressionCount']

    # Saccade amplitude的均值
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

    # 填充缺失值（如果有）
    features_df = features_df.fillna(features_df.mean(numeric_only=True))

    print(f"   ✓ 特征数量: 5个（VR_MMSE_total + 4个眼动特征）")

    # 3. 准备X和y
    feature_columns = ['VR_MMSE_total', 'Mean_FixTime', 'Mean_EnterCount',
                       'Mean_RegressionCount', 'Mean_Saccade_Amplitude']

    X = features_df[feature_columns].values
    y = (features_df['Group'] == 'MCI').astype(int).values  # MCI=1, Control=0
    participant_ids = features_df['ParticipantID'].values

    print(f"   ✓ 样本数: {len(X)}")
    print(f"   ✓ 特征列: {feature_columns}")

    # 4. 10-Fold Cross-Validation with leakage-safe standardization
    print("\n3. 执行10-Fold Cross-Validation（leakage-safe）...")
    print("   策略: 在每个fold内，StandardScaler仅用训练集拟合，再应用到测试集")

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # 存储out-of-fold predictions
    y_pred_proba_oof = np.zeros(len(y))
    fold_aucs = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Leakage-safe标准化：仅在训练集上拟合
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

        print(f"   Fold {fold_idx:2d}: AUC = {fold_auc:.4f} "
              f"(train={len(train_idx)}, test={len(test_idx)})")

    # 5. 计算总体out-of-fold AUC
    print("\n4. 计算Out-of-Fold AUC...")
    cv_auc = roc_auc_score(y, y_pred_proba_oof)
    print(f"   ✓ Cross-Validated AUC: {cv_auc:.4f}")
    print(f"   ✓ Mean Fold AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")

    # 6. 计算95% CI（bootstrap from out-of-fold predictions）
    print("\n5. 计算95%置信区间（Bootstrap from out-of-fold predictions）...")
    ci_lower, ci_upper = bootstrap_ci(y, y_pred_proba_oof, n_bootstrap=1000, ci=95)
    print(f"   ✓ 95% CI: [{ci_lower:.2f}–{ci_upper:.2f}]")

    # 7. 计算Youden index下的敏感度和特异度
    print("\n6. 计算Youden Index下的敏感度和特异度...")
    sensitivity, specificity, threshold = calculate_youden_index(y, y_pred_proba_oof)
    print(f"   ✓ 最佳阈值: {threshold:.4f}")
    print(f"   ✓ 敏感度 (Sensitivity): {sensitivity:.2%}")
    print(f"   ✓ 特异度 (Specificity): {specificity:.2%}")
    print(f"   ✓ Youden Index: {sensitivity + specificity - 1:.4f}")

    # 8. 生成报告
    print("\n" + "=" * 80)
    print("最终结果摘要（用于论文）")
    print("=" * 80)

    print(f"\nMCI vs Controls (N={len(y)}):")
    print(f"  Cross-validated AUC: {cv_auc:.2f}")
    print(f"  95% CI: {ci_lower:.2f}–{ci_upper:.2f}")
    print(f"  Sensitivity: {sensitivity:.1%}")
    print(f"  Specificity: {specificity:.1%}")

    print("\n用于替换文档中的占位符:")
    print(f"  [CV_AUC_MCI] = {cv_auc:.2f}")
    print(f"  [CI_low–CI_high] = {ci_lower:.2f}–{ci_upper:.2f}")
    print(f"  [Sens_MCI] = {sensitivity:.1%}")
    print(f"  [Spec_MCI] = {specificity:.1%}")

    # 9. 保存详细结果
    print("\n" + "=" * 80)
    print("保存详细结果...")
    print("=" * 80)

    # 保存out-of-fold predictions
    results_df = pd.DataFrame({
        'ParticipantID': participant_ids,
        'True_Label': y,
        'Predicted_Probability': y_pred_proba_oof,
        'Group': features_df['Group'].values
    })

    output_dir = base_dir / 'results/roc_cv'
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / 'mci_cv_roc_results.csv'
    results_df.to_csv(results_file, index=False)
    print(f"✓ Out-of-fold predictions已保存: {results_file}")

    # 保存摘要报告
    report_file = output_dir / 'mci_cv_roc_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("MCI vs Controls - Cross-Validated ROC Analysis Report\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Analysis Date: 2026-01-15\n")
        f.write(f"Method: 10-Fold Stratified Cross-Validation\n")
        f.write(f"Standardization: Leakage-safe (within-fold, training-only fit)\n\n")

        f.write("Sample Size:\n")
        f.write(f"  Control: {(y == 0).sum()} participants\n")
        f.write(f"  MCI: {(y == 1).sum()} participants\n")
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
        f.write(f"  Specificity: {specificity:.4f} ({specificity:.1%})\n")
        f.write(f"  Youden Index: {sensitivity + specificity - 1:.4f}\n\n")

        f.write("=" * 80 + "\n")
        f.write("For Manuscript (Copy-Paste Values)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"[CV_AUC_MCI] = {cv_auc:.2f}\n")
        f.write(f"[CI_low–CI_high] = {ci_lower:.2f}–{ci_upper:.2f}\n")
        f.write(f"[Sens_MCI] = {sensitivity:.1%}\n")
        f.write(f"[Spec_MCI] = {specificity:.1%}\n\n")

        f.write("Example sentence for Results:\n")
        f.write(f"\"For MCI vs controls, the cross-validated AUC was {cv_auc:.2f} \n")
        f.write(f"(95% CI {ci_lower:.2f}–{ci_upper:.2f}), with sensitivity of {sensitivity:.1%} \n")
        f.write(f"and specificity of {specificity:.1%} at the optimal threshold.\"\n\n")

    print(f"✓ 详细报告已保存: {report_file}")

    print("\n" + "=" * 80)
    print("✓ 分析完成！")
    print("=" * 80)

    return cv_auc, ci_lower, ci_upper, sensitivity, specificity

if __name__ == '__main__':
    main()
