import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from .misc import *   
import seaborn as sns
import pandas as pd
from sklearn.metrics import roc_curve, auc
from matplotlib.colors import LinearSegmentedColormap

__all__ = ['make_image', 'show_batch', 'show_mask', 'show_mask_single']

# functions to show an image
def make_image(img, mean=(0,0,0), std=(1,1,1)):
    for i in range(0, 3):
        img[i] = img[i] * std[i] + mean[i]    # unnormalize
    npimg = img.numpy()
    return np.transpose(npimg, (1, 2, 0))

def gauss(x,a,b,c):
    return torch.exp(-torch.pow(torch.add(x,-b),2).div(2*c*c)).mul(a)

def colorize(x):
    ''' Converts a one-channel grayscale image to a color heatmap image '''
    if x.dim() == 2:
        torch.unsqueeze(x, 0, out=x)
    if x.dim() == 3:
        cl = torch.zeros([3, x.size(1), x.size(2)])
        cl[0] = gauss(x,.5,.6,.2) + gauss(x,1,.8,.3)
        cl[1] = gauss(x,1,.5,.3)
        cl[2] = gauss(x,1,.2,.3)
        cl[cl.gt(1)] = 1
    elif x.dim() == 4:
        cl = torch.zeros([x.size(0), 3, x.size(2), x.size(3)])
        cl[:,0,:,:] = gauss(x,.5,.6,.2) + gauss(x,1,.8,.3)
        cl[:,1,:,:] = gauss(x,1,.5,.3)
        cl[:,2,:,:] = gauss(x,1,.2,.3)
    return cl

def show_batch(images, Mean=(2, 2, 2), Std=(0.5,0.5,0.5)):
    images = make_image(torchvision.utils.make_grid(images), Mean, Std)
    plt.imshow(images)
    plt.show()

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
import os

def plot_corr(x, y, label_class=None, title='Correlation Plot', x_label='X-axis', y_label='Y-axis', save_path=None, fig_size=(1.7, 2), r_value=None, p_value=None):

    plt.rcParams.update({'font.size': 7, 'font.family': 'sans-serif'})
    # Convert x to a numpy array to avoid TypeError in multiplication
    x = np.array(x).flatten()
    y = np.array(y).flatten()
    label_class = np.array(label_class).flatten() if label_class is not None else None

    # Calculate linear regression if not provided
    if r_value is None or p_value is None:
        slope, intercept, _, p_value, std_err = stats.linregress(x, y)
    else:
        slope, intercept = np.polyfit(x, y, 1)
    line = slope * x + intercept

    # Create scatter plot
    fig, ax = plt.subplots(figsize=fig_size)
    # Draw each point with differentiation by label_class using custom colors
    class_colors = ['blue', 'salmon', 'red', 'purple', 'orange', 'yellow']  # Customizable list of colors for different classes
    labels = ['AD', 'HC']
    if label_class is not None:
        unique_classes = np.unique(label_class)
        for idx, cls in enumerate(unique_classes):
            cls_mask = label_class == cls
            color = class_colors[idx % len(class_colors)]  # Cycle through class_colors list
            sns.scatterplot(x=x[cls_mask], y=y[cls_mask], color=color, label=labels[idx], s=10, ax=ax)
    else:
        sns.scatterplot(x=x, y=y, color='black', s=10, ax=ax)  # Default scatter plot without class differentiation
    ax.fill_between([0], [0], [0], color='red', alpha=0.2, label='95% CI', edgecolor='none')

    # Add linear regression line and confidence interval
    # ax.plot(x, line, 'r', label=f'95% CI', linewidth=0.5)
    sns.regplot(x=x, y=y, scatter=False, ci=95, color='red', line_kws={'linewidth': 0.5}, ax=ax)
    ax.spines['left'].set_linewidth(0.5)   # 设置左侧轴线的粗细
    ax.spines['bottom'].set_linewidth(0.5) # 设置底部轴线的粗细
    # Add text and labels with Nature Medicine style
    ax.set_xlabel(x_label, fontsize=6)
    ax.set_ylabel(y_label, fontsize=6)
    ax.margins(x=0.05, y=0.05)
    # Adjust tick parameters for smaller font size
    ax.tick_params(axis='both', which='major', length=2.5, width=0.5, labelsize=6)

    # Set legend position and size, and remove frame
    ax.legend(loc='lower right', fontsize=5, frameon=False)

    # Annotate R and P values in the top left corner
    ax.annotate(f'R = {r_value:.2f}, P = {p_value:.1e}', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=6, ha='left', va='top')
    print(f'R = {r_value:.2f}, P = {p_value:.1e}')
    # Legend with dashed border
    legend = ax.legend(loc='lower right', fontsize=5, frameon=True)
    frame = legend.get_frame()
    frame.set_linewidth(0.5)
    frame.set_edgecolor('black')
    frame.set_linestyle('dashed')  # Set the frame to dashed line
    # Remove top and right borders
    sns.despine()
    # Adjust layout to make room for labels
    plt.subplots_adjust(left=0.18, right=0.95, top=0.85, bottom=0.18)

    # Set axis limits and ticks
    ax.set_xlim(2, 31)
    ax.set_ylim(6, 33)
    ax.set_xticks(np.arange(5, 31, 5))
    ax.set_yticks(np.arange(10, 31, 5))

    # Save the figure with a high resolution
    if save_path is not None:
        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0.1)
        
def plot_corr_adnc(x, y, label_class=None, title='Correlation Plot', x_label='X-axis', y_label='Y-axis', save_path=None, fig_size=(2, 2), r_value=None, p_value=None):

    plt.rcParams.update({'font.size': 7, 'font.family': 'sans-serif'})
    # Convert x to a numpy array to avoid TypeError in multiplication
    x = np.array(x).flatten()
    y = np.array(y).flatten()
    label_class = np.array(label_class).flatten() if label_class is not None else None
    # Calculate linear regression if not provided
    if r_value is None or p_value is None:
        slope, intercept, _, p_value, std_err = stats.linregress(x, y)
    else:
        slope, intercept = np.polyfit(x, y, 1)
    line = slope * x + intercept
    # Create scatter plot
    fig, ax = plt.subplots(figsize=fig_size)
    # Draw each point with differentiation by label_class using custom colors
    class_colors = ['salmon', 'blue', 'red', 'purple', 'orange', 'yellow']  # Customizable list of colors for different classes
    labels = ['HC', 'AD']
    if label_class is not None:
        unique_classes = np.unique(label_class)
        for idx, cls in enumerate(unique_classes):
            cls_mask = label_class == cls
            color = class_colors[idx % len(class_colors)]  # Cycle through class_colors list
            sns.scatterplot(x=x[cls_mask], y=y[cls_mask], color=color, label=labels[idx], s=10, ax=ax)
    else:
        sns.scatterplot(x=x, y=y, color='black', s=10, ax=ax)  # Default scatter plot without class differentiation
    ax.fill_between([0], [0], [0], color='red', alpha=0.2, label='95% CI', edgecolor='none')

    # Add linear regression line and confidence interval
    # ax.plot(x, line, 'r', label=f'95% CI', linewidth=0.5)
    sns.regplot(x=x, y=y, scatter=False, ci=95, color='red', line_kws={'linewidth': 0.5}, ax=ax)
    ax.spines['left'].set_linewidth(0.5)   # 设置左侧轴线的粗细
    ax.spines['bottom'].set_linewidth(0.5) # 设置底部轴线的粗细
    # Add text and labels with Nature Medicine style
    ax.set_xlabel(x_label, fontsize=6)
    ax.set_ylabel(y_label, fontsize=6)
    ax.margins(x=0.05, y=0.05)
    # Adjust tick parameters for smaller font size
    ax.tick_params(axis='both', which='major', labelsize=5)

    # Set legend position and size, and remove frame
    ax.legend(loc='lower right', fontsize=5, frameon=False)

    # Annotate R and P values in the top left corner
    ax.annotate(f'R = {r_value:.2f}, P = {p_value:.1e}', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=6, ha='left', va='top')

    legend = ax.legend(loc='lower right', fontsize=5, frameon=True)
    frame = legend.get_frame()
    frame.set_linewidth(0.5)
    frame.set_edgecolor('black')
    frame.set_linestyle('dashed')  # Set the frame to dashed line
    # Remove top and right borders
    sns.despine()
    # Adjust layout to make room for labels
    plt.subplots_adjust(left=0.18, right=0.95, top=0.85, bottom=0.18)

    # Set axis limits and ticks
    ax.set_xlim(7, 32)
    ax.set_ylim(7, 34)
    ax.set_xticks(np.arange(10, 31, 5))
    ax.set_yticks(np.arange(10, 31, 5))

    # Save the figure with a high resolution
    if save_path is not None:
        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0.1)


def plot_roc(all_targets_binary, all_preds, roc_auc, text, save_path):
    """
    Plots a ROC curve with the given binary targets and predictions.
    
    Parameters:
    all_targets_binary (array-like): Binary target values.
    all_preds (array-like): Predicted values.
    save_path (str): Path to save the ROC curve image.
    """
    fpr, tpr, _ = roc_curve(all_targets_binary, all_preds)
    if roc_auc is None:
     roc_auc = auc(fpr, tpr)

    specificity = 1 - fpr

    plt.rcParams.update({'font.size': 7, 'font.family': 'sans-serif'})

    fig, ax = plt.subplots(figsize=(2, 2))
    # color=['teal', darkorange, mediumseagreen, indianred,coral, lightsteelblue,tomato,darkseagreen, lightgreen 'magenta', 'orange', 'green']
    ax.plot(specificity, tpr, color='lightsteelblue', lw=1)
    ax.fill_between(specificity, tpr, alpha=0.2, color='lightsteelblue')
    ax.set_xlim([1.05, -0.05])  # Reverse the x-axis
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('Specificity', fontsize=6)
    ax.set_ylabel('Sensitivity', fontsize=6)
    
    ax.margins(x=0.05, y=0.05)
    ax.set_xticks(np.arange(1.0, -0.2, -0.2))
    ax.set_yticks(np.arange(0.0, 1.2, 0.2))

    # Add some padding between the plot and the axis
    ax.margins(x=0.05, y=0.05)

    # Place AUC text inside the plot
    ax.text(0.5, 0.4, text, fontsize=7, ha='center', va='center', transform=ax.transAxes)
    ax.text(0.5, 0.3, f'AUC: {roc_auc:.3f}', fontsize=7, ha='center', va='center', transform=ax.transAxes)
    
    # Adjust tick parameters for smaller font size and sans-serif font
    ax.tick_params(axis='both', which='major', length=2.5, width=0.5, labelsize=6)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}'))
    ax.spines['left'].set_linewidth(0.5)   # 设置左侧轴线的粗细
    ax.spines['bottom'].set_linewidth(0.5) # 设置底部轴线的粗细
    # Remove top and right borders, and grid
    sns.despine()

    # Adjust layout to make room for labels
    plt.subplots_adjust(left=0.18, right=0.95, top=0.85, bottom=0.18)

    # Save the figure with a high resolution
    plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()


def plot_roc2(csv_path, prediction_columns, labels, save_path):
    data = pd.read_csv(csv_path)
    all_targets_binary = data['target'].values
    all_preds_list = [data[col].values for col in prediction_columns]

    roc_auc_list = []
    for preds in all_preds_list:
        fpr, tpr, _ = roc_curve(all_targets_binary, preds)
        roc_auc = auc(fpr, tpr)
        roc_auc_list.append(roc_auc)
    plt.rcParams.update({'font.size': 7, 'font.family': 'sans-serif'})
    fig, ax = plt.subplots(figsize=(2, 2))

    colors = ['teal', 'magenta', 'orange', 'green']  # Different colors for different ROC curves

    for preds, aauc, label, color in zip(all_preds_list, roc_auc_list, labels, colors):
        fpr, tpr, _ = roc_curve(all_targets_binary, preds)
        ax.plot(1 - fpr, tpr, color=color, lw=2, label=f'{label} (AUC: {aauc:.3f})')
        ax.fill_between(1 - fpr, tpr, alpha=0.2, color=color)

    ax.set_xlim([1.05, -0.05])  # Reverse the x-axis
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('Specificity', fontsize=6)
    ax.set_ylabel('Sensitivity', fontsize=6)
    
    ax.margins(x=0.05, y=0.05)
    ax.set_xticks(np.arange(1.0, -0.2, -0.2))
    ax.set_yticks(np.arange(0.0, 1.2, 0.2))

    ax.tick_params(axis='both', which='major', labelsize=5)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}'))
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    sns.despine()

    plt.subplots_adjust(left=0.18, right=0.95, top=0.85, bottom=0.18)
    ax.legend(loc='lower right', fontsize=5, frameon=True)

    plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

def boxplot(*data_groups, save_path=None):
    """
    Creates a styled boxplot for the given data groups and calculates p-values
    using the Kruskal-Wallis test.

    Parameters:
    - data_groups: variable number of array-like. Each represents a group.
    """

    # Flatten the input arrays to 1D in case they are n*1
    flat_data_groups = [np.ravel(group) for group in data_groups]

    if len(flat_data_groups) > 2:
        # Perform Kruskal-Wallis test
        h_value, p_value_kruskal = stats.kruskal(*flat_data_groups)
        print(f"Kruskal-Wallis p-value: {p_value_kruskal:.2e}")
    else:
        # Perform Mann-Whitney U test for two groups
        u_value, p_value_kruskal = stats.mannwhitneyu(flat_data_groups[0], flat_data_groups[1])
        print(f"Mann-Whitney U p-value: {p_value_kruskal:.2e}")

    # Create the boxplot
    fig, ax = plt.subplots(figsize=(4, 2.65))
    if len(flat_data_groups) > 2:
        ax.text(0.5, 0.95, f'Kruskal-Wallis, P = {p_value_kruskal:.1e}', 
                transform=ax.transAxes, fontsize=7, va='top', ha='center')
    else:
        ax.text(0.5, 0.95, f'Mann-Whitney U, P = {p_value_kruskal:.1e}',
                transform=ax.transAxes, fontsize=7, va='top', ha='center')
    
    boxprops = dict(linestyle='-', linewidth=1, color='black')
    medianprops = dict(linestyle='-', linewidth=1, color='black')
    whiskerprops = dict(linestyle='-', linewidth=1, color='black')
    capprops = dict(linestyle='-', linewidth=1, color='black')
    flierprops = dict(marker='o', markerfacecolor='none', markeredgecolor='black', linewidth=0.5, markersize=3, markeredgewidth=1)

    bp = ax.boxplot(flat_data_groups, patch_artist=True,
                    boxprops=boxprops,
                    medianprops=medianprops,
                    whiskerprops=whiskerprops,
                    capprops=capprops,
                    flierprops=flierprops,
                    widths=0.4)
    # Customizing the color and style
    colors = ['#D7191C', '#2C7BB6', '#FDAE61', '#ABDDA4']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    # # Customizing the whiskers, fliers, caps, and median lines
    # for whisker in bp['whiskers']:
    #     whisker.set(color='#000000', linewidth=2.0)
    # for cap in bp['caps']:
    #     cap.set(color='#000000', linewidth=2.0)
    # for median in bp['medians']:
    #     median.set(color='#000000', linewidth=2.0)
    # for flier in bp['fliers']:
    #     flier.set(marker='o', color='#000000', alpha=1.0)

    # Setting the y-axis and x-axis labels
    ax.set_ylabel('Value')
    ax.set_xticklabels([f'Group {i}' for i in range(1, len(data_groups) + 1)], fontsize=6)
    
    # Removing the top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Adjusting the limits if necessary
    data_max = max(np.concatenate(flat_data_groups))
    data_min = min(np.concatenate(flat_data_groups))
    upper_ylim = data_max + (data_max * 0.1)  # Add 10% headroom above the max data point
    lower_ylim = data_min - (data_min * 0.1)  # Add 10% headroom below the min data point
    ax.set_ylim(lower_ylim, upper_ylim)
    ax.tick_params(axis='y', labelsize=6)
    ax.tick_params(axis='x', which='major', pad=2)
    plt.savefig(save_path, format='png', dpi=300)

def boxplot2(scores, categories, save_path=None):
    plt.rcParams.update({
        'font.size': 7,
        'font.family': 'sans-serif',
        'figure.figsize': (2, 2)
    })

    fig, ax = plt.subplots()

    grouped_data = {0: [], 1: []}
    for score, category in zip(scores, categories):
        grouped_data[category].append(score)

    flat_data_groups = [np.ravel(group) for group in grouped_data.values()]
    if len(flat_data_groups) > 2:
        # Perform Kruskal-Wallis test
        h_value, p_value = stats.kruskal(*flat_data_groups)
    else:
        # Perform Mann-Whitney U test for two groups
        u_value, p_value = stats.mannwhitneyu(flat_data_groups[0], flat_data_groups[1])
    ax.text(0.5, 1.05, f'P = {p_value:.1e}', transform=ax.transAxes, ha='center', va='top', fontsize=7)

    _, p_value = stats.mannwhitneyu(*flat_data_groups)

    # 确保颜色设置正确
    colors = ['#f8766d', '#00bfc4']  # 绿色和紫色的十六进制代码
    bp = ax.boxplot(flat_data_groups, patch_artist=True,
                    widths=0.4, whiskerprops=dict(linestyle='-', color='black'),
                    medianprops=dict(linestyle='-', linewidth=2, color='black'))

    # 为每个箱形图设置颜色
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_xticks([1, 2])  # 设置刻度位置
    ax.set_xticklabels(['HC', 'AD'], fontsize=7)
    ax.set_ylabel('Values', fontsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # 展示或保存图像
    if save_path:
        plt.savefig(save_path, format='png', dpi=300)


def violinplot2(scores, categories, save_path=None):
    plt.rcParams.update({
        'font.size': 7,
        'font.family': 'sans-serif',
        'figure.figsize': (2, 2)
    })

    fig, ax = plt.subplots()

    grouped_data = {0: [], 1: []}
    for score, category in zip(scores, categories):
        grouped_data[category].append(score)

    flat_data_groups = [np.ravel(group) for group in grouped_data.values()]

    # 进行 Mann-Whitney U 测试
    if len(flat_data_groups) > 2:
        h_value, p_value = stats.kruskal(*flat_data_groups)
    else:
        u_value, p_value = stats.mannwhitneyu(flat_data_groups[0], flat_data_groups[1])

    # 显示 P 值
    ax.text(0.5, 1.05, f'P = {p_value:.1e}', transform=ax.transAxes, ha='center', va='top', fontsize=7)

    # 绘制小提琴图
    vp = ax.violinplot(flat_data_groups, showmeans=False, showmedians=False, showextrema=False, widths=0.6)

    # 自定义小提琴图部件颜色
    colors = ['#f8766d', '#00bfc4']  # 更新颜色
    for pc, color in zip(vp['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)  # 轻微透明度提高美观性

    # 在小提琴图内添加小的箱线图，不显示帽子线
    ax.boxplot(flat_data_groups, widths=0.1, patch_artist=True, 
               boxprops=dict(facecolor='white', color='black'),
               medianprops=dict(color='black'), whiskerprops=dict(color='black'),
               capprops=dict(color='black', visible=False),  # 不显示帽子线
               flierprops=dict(marker='o', color='red', markersize=5))

    ax.set_xticks([1, 2])  # 确保刻度准确位置
    ax.set_xticklabels(['HC', 'AD'], fontsize=7)
    ax.set_ylabel('Values', fontsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 展示或保存图像
    plt.savefig(save_path, format='png', dpi=300)


def barplot(data, categories, labels, save_path=None):
    plt.rcParams.update({
        'font.size': 7,
        'font.family': 'sans-serif',
        'figure.figsize': (2, 2)
    })

    fig, ax = plt.subplots()

    grouped_data = {label: [] for label in labels}
    for value, category in zip(data, categories):
        grouped_data[category].append(value)

    means = [np.mean(grouped_data[label]) for label in labels]
    stds = [np.std(grouped_data[label]) for label in labels]

    # 绘制柱状图
    bars = ax.bar(labels, means, yerr=stds, capsize=5, color=['#66c2a5', '#fc8d62', '#8da0cb'], edgecolor='black')

    # 添加数据点
    for i, (label, mean, std) in enumerate(zip(labels, means, stds)):
        y = np.random.normal(mean, std, len(grouped_data[label]))  # 使用正态分布生成数据点
        x = np.random.normal(i, 0.04, len(grouped_data[label]))   # 添加小的随机扰动以使点散开
        ax.scatter(x, y, color='black', zorder=10)

    # 设置标签和标题
    ax.set_ylabel('Pearson Correlation')
    ax.set_xlabel('PD Severity Prediction')
    ax.set_ylim(0, 1)
    ax.set_xticks(range(len(labels)))  # 设置刻度位置
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_title('Pearson correlation of the PD severity prediction and MDS-UPDRS')

    # 添加图例
    colors = ['#66c2a5', '#fc8d62', '#8da0cb']
    legend_labels = ['All', 'w/o Multi-task', 'w/o Transfer']
    patches = [plt.plot([],[], marker="o", ms=10, ls="", mec=None, color=colors[i], 
                label="{:s}".format(legend_labels[i]) )[0]  for i in range(len(legend_labels))]
    plt.legend(handles=patches, loc='upper right', title='Condition')

    plt.savefig(save_path, format='png', dpi=300)


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
def plottsne(features, labels, title, n_components_pca=50, save_path=None):
    if features.shape[1] > n_components_pca:
        pca = PCA(n_components=n_components_pca)
        features_reduced = pca.fit_transform(features)
        print("PCA applied, reduced dimensions to:", features_reduced.shape[1])
    else:
        features_reduced = features
        print("PCA not applied due to feature dimension less than or equal to pca components.")
        
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(features_reduced)

    plt.figure(figsize=(2, 2))
    plt.rcParams.update({'font.size': 7, 'font.family': 'sans-serif'})
    class_colors = ['salmon', 'blue']
    labels_text = ['HC', 'AD']
    for idx, label in enumerate(np.unique(labels)):
        mask = labels == label
        plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], color=class_colors[idx], label=labels_text[idx], s=20, alpha=0.8)
    
    plt.title(title, fontsize=7)
    # plt.xlabel('t-SNE 1', fontsize=5)
    # plt.ylabel('t-SNE 2', fontsize=5)
    plt.xticks([])
    plt.yticks([])
    plt.legend(loc='lower right', fontsize=5, frameon=False)
    
    legend = plt.legend(loc='lower right', fontsize=5, frameon=True)
    frame = legend.get_frame()
    frame.set_linewidth(0.5)
    frame.set_edgecolor('black')
    frame.set_linestyle('dashed')  # Set the frame to dashed line
    # plt.tick_params(axis='both', which='major', labelsize=5)
    plt.savefig(save_path, format='png', dpi=300)


from scipy.stats import pearsonr, spearmanr
def plot_ablation(labels, predictions_list, n_folds, save_path):
    """
    Plot and save a bar chart with Pearson Correlation Coefficients for models using n-fold cross-validation.

    Parameters:
    - labels: The true values.
    - predictions_list: A list containing predictions from each model (each model's predictions should be a list as well).
    - n_folds: The number of folds used in the cross-validation.
    - save_path: Path to save the figure.
    """
    # Ensure labels and predictions are NumPy arrays for splitting
    labels = np.array(labels)
    predictions_list = [np.array(predictions) for predictions in predictions_list]

    # Split the labels and predictions according to the folds
    fold_size = len(labels) // n_folds
    correlations = []
    std_devs = []

    for predictions in predictions_list:
        fold_correlations = []
        for fold in range(n_folds):
            start = fold * fold_size
            end = (fold + 1) * fold_size if fold != n_folds - 1 else len(labels)
            fold_labels = labels[start:end]
            fold_predictions = predictions[start:end]
            corr, _ = pearsonr(fold_labels, fold_predictions)
            fold_correlations.append(corr)
        # Calculate mean and standard deviation of correlations across folds
        fold_mean = np.mean(fold_correlations)
        fold_std = np.std(fold_correlations)
        correlations.append(fold_mean)
        std_devs.append(fold_std)

    # Plot settings
    categories = ['Model 1', 'Model 2', 'Model 3']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # Create the bar chart
    fig, ax = plt.subplots()
    ax.bar(categories, correlations, yerr=std_devs, capsize=10, color=colors, edgecolor='black')
    ax.set_ylabel('Pearson Correlation')
    ax.set_title('PD Severity Prediction (Cross-Validated)')
    ax.set_ylim(0, 1.0)  # Pearson Correlation range

    # Save the plot
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)  # Save for publication quality
    plt.close()  # Close the figure to prevent display if not necessary

    return correlations, std_devs  # Return the computed metrics

# # Example usage
# labels = np.random.rand(100)  # Replace with actual labels
# predictions_1 = np.random.rand(100)  # Replace with actual predictions from model 1
# predictions_2 = np.random.rand(100)  # Replace with actual predictions from model 2
# predictions_3 = np.random.rand(100)  # Replace with actual predictions from model 3
# predictions_list = [predictions_1, predictions_2, predictions_3]
# n_folds = 5  # Example fold number
# save_path = '/mnt/data/cross_validated_pearson_correlation_chart.png'

# # Calculate and plot the cross-validated Pearson correlations
# correlations, std_devs = plot_ablation(
#     labels, predictions_list, n_folds, save_path
# )

# # Display the calculated correlations and standard deviations for each model
# for i, (corr, std_dev) in enumerate(zip(correlations, std_devs)):
#     print(f"Model {i+1}: Pearson Correlation = {corr:.2f}, Standard Deviation = {std_dev:.2f}")

# # Display saved path for download
# save_path

    # Show the plot
# x = torch.zeros(1, 3, 3)
# out = colorize(x)
# out_im = make_image(out)
# plt.imshow(out_im)
# plt.show()
import matplotlib.pyplot as plt
import numpy as np
import torch

def visualize_feature_map(inputs, model, selected_layer, use_cuda=False):
    """
    Visualize the feature maps of a selected layer in a deep learning model.

    Parameters:
    - img: Input image, should be a torch Variable with shape (1, C, H, W)
    - model: Pre-trained deep learning model
    - selected_layer: The layer of the model we want to visualize
    - use_cuda: If True, use GPU for computation

    Returns:
    - None: This function will plot the feature maps directly
    """
    
    # Move input and model to GPU if required
    if use_cuda:
        inputs, targets = [x.cuda() for x in inputs], targets.cuda()
        model = model.cuda()
    
    # Set model to evaluation mode
    model.eval()
    
    # Register hook to access to feature maps
    features_blobs = []
    
    def hook_feature(module, input, output):
        features_blobs.append(output.data.cpu().numpy())

    # Hook the selected layer
    model._modules.get(selected_layer).register_forward_hook(hook_feature)

    # Forward pass to get feature maps
    model(inputs)
    
    # Get feature maps from the hooked layer
    feature_maps = features_blobs[0]
    num_feature_maps = feature_maps.shape[1]  # Get the number of feature maps

    # Plot the feature maps
    rows = np.ceil(np.sqrt(num_feature_maps)).astype(int)
    cols = np.ceil(num_feature_maps / rows).astype(int)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    fig.suptitle(f'Feature Maps of layer {selected_layer}')

    for i, ax in enumerate(axes.flat):
        if i < num_feature_maps:
            ax.imshow(feature_maps[0, i, :, :], cmap='viridis')
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()

    # plt.tight_layout(pad=0.5)
    # plt.savefig(os.path.join(save_path, 'Boxline2.png'), format='png', dpi=300)
    # plt.close()
