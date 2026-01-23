#!/usr/bin/env python3
"""
Feature Distribution Visualization (P2 Requirement)

专家要求：创建图表展示non-overlapping ranges that yield AUC=1.00

展示：
1. Mean_EnterCount的Control vs MCI分布（完全分离）
2. Mean_RegressionCount的Control vs MCI分布（完全分离）
3. 其他特征的分布（有重叠）

日期：2026-01-19
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")

def main():
    print("=" * 80)
    print("Feature Distribution Visualization (P2 Requirement)")
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

    print(f"   ✓ Features computed for {len(features_df)} participants")

    # 3. Create visualizations
    print("\n3. Creating visualizations...")

    feature_list = [
        ('Mean_EnterCount', 'Mean ROI Entry Count'),
        ('Mean_RegressionCount', 'Mean Regression Count'),
        ('VR_MMSE_total', 'VR-MMSE Total Score'),
        ('Mean_FixTime', 'Mean Fixation Time (s)'),
        ('Mean_Saccade_Amplitude', 'Mean Saccade Amplitude (°)')
    ]

    # Create figure with 2x3 subplots (5 features + 1 summary)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Feature Distributions: MCI vs Controls\nShowing Complete Separation in Eye-Tracking Metrics',
                 fontsize=16, fontweight='bold', y=0.995)

    axes = axes.flatten()

    for idx, (feat, feat_label) in enumerate(feature_list):
        ax = axes[idx]

        # Get data for each group
        control_data = features_df[features_df['Group'] == 'Control'][feat].dropna()
        mci_data = features_df[features_df['Group'] == 'MCI'][feat].dropna()

        # Histogram + KDE
        ax.hist(control_data, bins=15, alpha=0.5, label='Control', color='#1f77b4', edgecolor='black')
        ax.hist(mci_data, bins=15, alpha=0.5, label='MCI', color='#ff7f0e', edgecolor='black')

        # Add vertical lines for ranges
        control_max = control_data.max()
        mci_min = mci_data.min()

        ax.axvline(control_max, color='#1f77b4', linestyle='--', linewidth=2,
                   label=f'Control max = {control_max:.2f}')
        ax.axvline(mci_min, color='#ff7f0e', linestyle='--', linewidth=2,
                   label=f'MCI min = {mci_min:.2f}')

        # Check for separation
        gap = mci_min - control_max
        if gap > 0:
            # Complete separation!
            ax.axvspan(control_max, mci_min, alpha=0.2, color='green',
                       label=f'Gap = {gap:.2f}')
            separation_text = f"✓ COMPLETE SEPARATION (gap={gap:.2f})"
            ax.text(0.5, 0.95, separation_text, transform=ax.transAxes,
                   ha='center', va='top', fontweight='bold', color='green',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        else:
            # Overlapping
            overlap = control_max - mci_min
            separation_text = f"Overlap = {overlap:.2f}"
            ax.text(0.5, 0.95, separation_text, transform=ax.transAxes,
                   ha='center', va='top', color='red',
                   bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

        ax.set_xlabel(feat_label, fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=11)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    # 6th subplot: Summary statistics table
    ax_summary = axes[5]
    ax_summary.axis('off')

    summary_data = []
    for feat, feat_label in feature_list:
        control_data = features_df[features_df['Group'] == 'Control'][feat].dropna()
        mci_data = features_df[features_df['Group'] == 'MCI'][feat].dropna()

        control_max = control_data.max()
        mci_min = mci_data.min()
        gap = mci_min - control_max

        if gap > 0:
            separation = f"✓ {gap:.2f}"
            color = 'green'
        else:
            separation = f"✗ {-gap:.2f}"
            color = 'red'

        summary_data.append([
            feat_label[:20],  # Truncate long names
            f"{control_max:.2f}",
            f"{mci_min:.2f}",
            separation
        ])

    # Create table
    table = ax_summary.table(cellText=summary_data,
                            colLabels=['Feature', 'Control\nMax', 'MCI\nMin', 'Gap'],
                            cellLoc='center',
                            loc='center',
                            bbox=[0, 0, 1, 1])

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Color code the Gap column
    for i in range(1, len(summary_data) + 1):
        gap_cell = table[(i, 3)]
        if '✓' in gap_cell.get_text().get_text():
            gap_cell.set_facecolor('lightgreen')
        else:
            gap_cell.set_facecolor('lightcoral')

    # Bold header
    for j in range(4):
        table[(0, j)].set_facecolor('lightgray')
        table[(0, j)].set_text_props(weight='bold')

    ax_summary.set_title('Summary: Feature Separation Analysis', fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout()

    # Save figure
    output_dir = base_dir / 'results/roc_cv'
    output_dir.mkdir(parents=True, exist_ok=True)

    fig_file = output_dir / 'feature_distribution_separation.png'
    plt.savefig(fig_file, dpi=300, bbox_inches='tight')
    print(f"   ✓ Figure saved: {fig_file}")

    # Also save high-quality PDF
    pdf_file = output_dir / 'feature_distribution_separation.pdf'
    plt.savefig(pdf_file, bbox_inches='tight')
    print(f"   ✓ PDF saved: {pdf_file}")

    plt.close()

    # 4. Create a focused plot showing only the two separating features
    print("\n4. Creating focused plot for the two separating features...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Complete Feature Separation: MCI vs Controls\nKey Eye-Tracking Metrics Driving AUC=1.00',
                 fontsize=14, fontweight='bold')

    separating_features = [
        ('Mean_EnterCount', 'Mean ROI Entry Count'),
        ('Mean_RegressionCount', 'Mean Regression Count')
    ]

    for idx, (feat, feat_label) in enumerate(separating_features):
        ax = axes[idx]

        control_data = features_df[features_df['Group'] == 'Control'][feat].dropna()
        mci_data = features_df[features_df['Group'] == 'MCI'][feat].dropna()

        # Box plot + strip plot
        data_for_plot = pd.concat([
            control_data.to_frame(name='value').assign(Group='Control'),
            mci_data.to_frame(name='value').assign(Group='MCI')
        ])

        sns.boxplot(data=data_for_plot, x='Group', y='value', ax=ax,
                   palette={'Control': '#1f77b4', 'MCI': '#ff7f0e'}, width=0.5)
        sns.stripplot(data=data_for_plot, x='Group', y='value', ax=ax,
                     color='black', alpha=0.5, size=6)

        # Add separation zone
        control_max = control_data.max()
        mci_min = mci_data.min()
        gap = mci_min - control_max

        ax.axhline(control_max, color='#1f77b4', linestyle='--', linewidth=2,
                   label=f'Control max = {control_max:.2f}')
        ax.axhline(mci_min, color='#ff7f0e', linestyle='--', linewidth=2,
                   label=f'MCI min = {mci_min:.2f}')
        ax.axhspan(control_max, mci_min, alpha=0.2, color='green')

        ax.text(0.5, (control_max + mci_min) / 2, f'Gap = {gap:.2f}\n(NO OVERLAP)',
               ha='center', va='center', fontweight='bold', fontsize=12,
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))

        ax.set_ylabel(feat_label, fontsize=12, fontweight='bold')
        ax.set_xlabel('Group', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    fig_focused = output_dir / 'feature_separation_focused.png'
    plt.savefig(fig_focused, dpi=300, bbox_inches='tight')
    print(f"   ✓ Focused figure saved: {fig_focused}")

    pdf_focused = output_dir / 'feature_separation_focused.pdf'
    plt.savefig(pdf_focused, bbox_inches='tight')
    print(f"   ✓ Focused PDF saved: {pdf_focused}")

    plt.close()

    # 5. Save feature statistics as CSV
    print("\n5. Saving feature statistics...")

    stats_data = []
    for feat, feat_label in feature_list:
        control_data = features_df[features_df['Group'] == 'Control'][feat].dropna()
        mci_data = features_df[features_df['Group'] == 'MCI'][feat].dropna()

        stats_data.append({
            'Feature': feat_label,
            'Control_Mean': control_data.mean(),
            'Control_SD': control_data.std(),
            'Control_Min': control_data.min(),
            'Control_Max': control_data.max(),
            'MCI_Mean': mci_data.mean(),
            'MCI_SD': mci_data.std(),
            'MCI_Min': mci_data.min(),
            'MCI_Max': mci_data.max(),
            'Gap': mci_data.min() - control_data.max(),
            'Complete_Separation': 'Yes' if (mci_data.min() - control_data.max()) > 0 else 'No'
        })

    stats_df = pd.DataFrame(stats_data)
    stats_file = output_dir / 'feature_separation_statistics.csv'
    stats_df.to_csv(stats_file, index=False)
    print(f"   ✓ Statistics CSV: {stats_file}")

    print("\n" + "=" * 80)
    print("Key Findings:")
    print("=" * 80)

    for row in stats_data:
        if row['Complete_Separation'] == 'Yes':
            print(f"\n✓ {row['Feature']}")
            print(f"  Control range: [{row['Control_Min']:.2f}, {row['Control_Max']:.2f}]")
            print(f"  MCI range: [{row['MCI_Min']:.2f}, {row['MCI_Max']:.2f}]")
            print(f"  Gap: {row['Gap']:.2f} (COMPLETE SEPARATION)")

    print("\n" + "=" * 80)
    print("✓ Feature distribution visualization complete!")
    print("=" * 80)

    return stats_df

if __name__ == '__main__':
    main()
