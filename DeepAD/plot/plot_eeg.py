import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from scipy.stats import wilcoxon

# Loading data
data = pd.read_csv(r'D:\1PHD\methods\2method_nature\ADR\experiments\eeg\cli2.csv')

# Preprocessing data, organizing into long format
data_long = pd.melt(data, var_name='video_version', value_name='eeg_score')

# Splitting video and version into two separate columns
data_long['video'] = data_long['video_version'].apply(lambda x: x.split('_')[0])
data_long['version'] = data_long['video_version'].apply(lambda x: x.split('_')[1])

# Print the restructured data
print(data_long.head())

def plot_eeg_boxplot(data, save_path):
    plt.rcParams.update({
        'font.size': 7,
        'font.family': 'sans-serif',
        'figure.figsize': (2, 2)
    })

    fig, ax = plt.subplots()
    grouped = data.groupby('video')
    p_values = {}
    order = ["Number", "Chainsaw"]

    for name in order:
        group = data[data["video"] == name]
        data_2d = group[group["version"] == "2d"]["eeg_score"].to_numpy()
        data_3d = group[group["version"] == "3d"]["eeg_score"].to_numpy()

        # 配对 Wilcoxon（同一被试两条件）
        stat, p = wilcoxon(data_2d, data_3d, alternative="two-sided")
        p_values[name] = p

    sns.boxplot(x='video', y='eeg_score', hue='version', data=data, order=order, ax=ax, palette=['#f8766d', '#00bfc4'],
                width=0.7,  # Control the width of the boxes
                whiskerprops=dict(linestyle='-', color='black', linewidth=1),  # Reduced whisker line width
                medianprops=dict(linestyle='-', linewidth=1, color='black'))     
    ax.set_ylabel('Cognitive load index', fontsize=7)
    ax.set_xlabel('')
    legend = plt.legend(loc='lower right', fontsize=5, frameon=True)
    frame = legend.get_frame()
    frame.set_linewidth(0.5)
    frame.set_edgecolor('black')
    frame.set_linestyle('dashed')
    ax.set_ylim(-0.2, 13.7)
    
    ylim = ax.get_ylim()
    for i, (video, p) in enumerate(p_values.items()):
        ax.text(i, ylim[1] - (ylim[1] - ylim[0]) * 0.1, f'P = {p:.2e}',
                color='black', ha='center', fontsize=6)

    sns.despine()
    plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()

plot_eeg_boxplot(data_long, 'D:\\1PHD\\methods\\2method_nature\\ADR\\experiments\\eeg\\eeg_scores_boxplot1.png')
