from __future__ import print_function
from matplotlib import pyplot as plt
import numpy as np

from scipy.stats import pearsonr
import pandas as pd
import seaborn as sns

def main():
    # 读取CSV文件corr.csv
    data = pd.read_csv("experiments/ablation/corr.csv")
    
    # 提取指定列的数据
    # column1 = data['sss_mmse'].values  # 请根据实际情况替换 'column_name1'
    # column2 = data['sss_moca'].values  # 请根据实际情况替换 'column_name2'
    # column3 = data['mmse'].values  # 请根据实际情况替换 'column_name2'
    # column4 = data['moca'].values  # 请根据实际情况替换 'column_name2'
    # column5 = data['cdt'].values  # 请根据实际情况替换 'column_name2'
    # column6 = data['npi'].values  # 请根据实际情况替换 'column_name2'
    # column7 = data['cdr'].values  # 请根据实际情况替换 'column_name2'
    # column8 = data['hamd'].values  # 请根据实际情况替换 'column_name2'
    # column9 = data['adl'].values  # 请根据实际情况替换 'column_name2'

    # # 绘制热图
    # features = [column1, column2, column3, column4, column5, column6, column7, column8, column9]
    plot_correlation_matrix(data, ['Pred_MMSE', 'Pred_MoCA', 'MMSE', 'MoCA', 'CDT', 'NPI', 'CDR-SB', 'HAMD', 'ADL'], 'experiments/ablation/correlation_heatmap.png')

def calculate_plcc(outputs, targets):
    # Calculate PLCC (Pearson Linear Correlation Coefficient)
    plcc_values = []
    # outputs = outputs.detach().cpu().numpy()
    # targets = targets.detach().cpu().numpy()
    plcc_col, p_plcc = pearsonr(outputs, targets)
    return plcc_col, p_plcc



def plot_correlation_matrix(data, features, save_path):
    """
    Computes the correlation matrix from given data and plots it with an aesthetic suitable for high-impact publications.
    
    Parameters:
        data (np.array or pd.DataFrame): Array or DataFrame containing feature data.
        features (list): List of feature names corresponding to the columns in data.
        save_path (str): Path and filename where the heatmap will be saved.
    """
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)
    
    corr_matrix = data.corr()

    # Define a sophisticated color palette
    cmap = sns.diverging_palette(220, 20, as_cmap=True)

    # Create the plot with adjusted aesthetics
    plt.figure(figsize=(6, 5))
    plt.rcParams.update({
        'font.size': 8, 
        'font.family': 'sans-serif',  # More typical in publications
        'axes.labelsize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8
    })

    # Create the heatmap
    ax = sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap=cmap, cbar_kws={'label': 'PLCC'},
                     xticklabels=features, yticklabels=features, square=True, linewidths=0, linecolor='gray', cbar=True)

    # Adjust color bar size and location
    cbar = ax.collections[0].colorbar
    cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), rotation=0)

    # Rotate the tick labels for better visibility
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    # Adjust margins and layout to make the plot square
    plt.subplots_adjust(left=0.2, right=0.95, top=0.95, bottom=0.2)

    # Save the heatmap
    plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
if __name__ == '__main__':
    main()