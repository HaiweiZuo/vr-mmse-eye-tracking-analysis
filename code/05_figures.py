#!/usr/bin/env python3
"""
Figure 3 - 论文风格版本（柱状图）
模仿原始论文的可视化风格
颜色：淡绿色、淡黄色、淡红色
日期：2026-01-15
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
from pathlib import Path

# 读取数据
base_dir = Path(__file__).parent.parent
data = pd.read_csv(base_dir / 'results/figures/Fig3_source_data.csv')

# 确保顺序
data['Group'] = pd.Categorical(data['Group'],
                                categories=['Control', 'MCI', 'AD'],
                                ordered=True)
data['Task'] = pd.Categorical(data['Task'],
                               categories=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'],
                               ordered=True)
data = data.sort_values(['Task', 'Group'])

# 定义颜色方案（从image.png精确提取）
colors = {
    'Control': '#496698',  # 深蓝色（RGB: 73, 102, 152）
    'MCI': '#F6A934',      # 橙黄色（RGB: 246, 169, 52）
    'AD': '#EE3233'        # 鲜红色（RGB: 238, 50, 51）
}

# 创建5个子图（每个任务一个）
fig, axes = plt.subplots(1, 5, figsize=(15, 4), sharey=True)

# 为每个任务创建柱状图
tasks = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
task_titles = [
    'Q1: Orientation to Time',
    'Q2: Orientation to Place',
    'Q3: Registration',
    'Q4: Attention & Calculation',
    'Q5: Recall'
]

x = np.arange(3)  # 三个组
width = 0.6  # 柱子宽度

for idx, (task, ax) in enumerate(zip(tasks, axes)):
    task_data = data[data['Task'] == task].copy()

    # 提取数据
    means = task_data['Mean_Amplitude_deg'].values
    sems = task_data['SEM_Amplitude_deg'].values
    groups = task_data['Group'].values

    # 创建柱状图
    bars = ax.bar(x, means, width,
                   color=[colors[g] for g in groups],
                   edgecolor='gray',
                   linewidth=0.8,
                   alpha=0.9)

    # 添加误差棒
    ax.errorbar(x, means, yerr=sems,
                fmt='none',
                ecolor='black',
                capsize=4,
                capthick=1.5,
                linewidth=1.5,
                alpha=0.7)

    # 设置标题
    ax.set_title(task_titles[idx], fontsize=11, fontweight='normal', pad=10)

    # 设置x轴
    ax.set_xticks(x)
    ax.set_xticklabels(['Control\nGroup', 'MCI\nGroup', 'AD\nGroup'],
                       fontsize=9)
    ax.set_xlabel('Group', fontsize=10, fontweight='normal')

    # 网格线
    ax.yaxis.grid(True, linestyle='-', alpha=0.2, color='gray')
    ax.set_axisbelow(True)

    # 移除顶部和右侧边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 设置y轴范围
    ax.set_ylim(0, max(data['Mean_Amplitude_deg']) * 1.2)

# 只在第一个子图显示y轴标签
axes[0].set_ylabel('Mean Saccade Amplitude (deg)',
                   fontsize=11, fontweight='bold')

# 调整子图间距
plt.tight_layout()

# 保存图片
output_dir = base_dir / 'results/figures'
output_dir.mkdir(parents=True, exist_ok=True)
output_base = 'Figure3_Saccade_Amplitude_Publication'
plt.savefig(output_dir / f'{output_base}.pdf', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / f'{output_base}.png', dpi=300, bbox_inches='tight')

print("=" * 70)
print("✓ Figure 3 (论文风格版本) 已生成！")
print("=" * 70)
print(f"\n生成文件:")
print(f"  - {output_base}.pdf (矢量格式)")
print(f"  - {output_base}.png (高分辨率)")

print("\n颜色方案（严格匹配您的配色图）:")
print(f"  Control: 深蓝色 {colors['Control']}")
print(f"  MCI: 深橙黄色 {colors['MCI']}")
print(f"  AD: 鲜红色 {colors['AD']}")

print("\n样式特点:")
print("  ✓ 柱状图（匹配原始论文）")
print("  ✓ 5个子图（每个任务独立）")
print("  ✓ 淡色配色方案")
print("  ✓ 误差棒（SEM）")
print("  ✓ 清晰的标签")

# 打印统计摘要
print("\n" + "=" * 70)
print("统计摘要（按任务）:")
print("=" * 70)
for task in tasks:
    task_data = data[data['Task'] == task]
    print(f"\n{task}:")
    for _, row in task_data.iterrows():
        print(f"  {row['Group']:8s}: {row['Mean_Amplitude_deg']:5.2f} ± {row['SEM_Amplitude_deg']:4.2f} deg")

plt.close()

print("\n" + "=" * 70)
print("完成！")
print("=" * 70)
