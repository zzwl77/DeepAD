import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_model_comparison(csv_path, selected_columns, metric_name, bar_names, save_path=None):

    plt.rcParams.update({
        'font.size': 7, 
        'font.family': 'sans-serif',  # More typical in publications
        'figure.figsize': (3, 3),
        'axes.labelsize': 7,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7
    })

    # Load data from CSV
    data = pd.read_csv(csv_path)
    
    # Select specific columns based on model names
    data = data[selected_columns]
    
    # Calculate means and standard deviations
    means = data.mean()
    stds = data.std()
    
    # Define bar positions and width
    n_groups = len(means)  # number of groups
    index = np.arange(n_groups)  # category positions
    bar_width = 0.8  # width of bars to make them adjacent

    # Create the bar plot
    fig, ax = plt.subplots()
    # Professional color palette
    colors = ['lightseagreen', 'darkseagreen', 'peachpuff']  # Greens, blues, and oranges
    bars = ax.bar(index, means, bar_width, yerr=stds, capsize=5, color=colors, label=bar_names, edgecolor='none', error_kw={'elinewidth':1, 'ecolor':'black'})

    # Adding individual data points as small circles
    for i, col in zip(index, selected_columns):
        y = data[col]
        x = np.random.normal(i, 0.04, size=len(y))  # Small jitter to the x-coordinates for visibility
        ax.scatter(x, y, color=colors[i % len(colors)], edgecolor='black', zorder=3, s=12)
    # Setting labels and titles
    plt.subplots_adjust(left=0.18, right=0.95, top=0.85, bottom=0.18)
    ax.tick_params(axis='both',length=0)

    ax.set_ylabel(metric_name)
    ax.set_xticks(index)
    ax.set_xticklabels(bar_names)  # Use column names as x-tick labels
    ax.set_ylim(0, 1.1)  # Adjust y-limit to fit the error bars
    ax.legend(loc='upper right')

### Save or display the figure
    if save_path:
        plt.savefig(save_path, format='png', dpi=300)

# Example usage
csv_path = 'experiments\\ablation\\cls.csv'    #adreg，adncreg
selected_columns = ['scls_auc', 'mt_auc', 'svm_auc']  # Replace with your actual model column names
metric_name = 'AUC'  # Replace with your metric
bar_names = ['DeepAD', 'ViT', 'SVM']  # Replace with your actual model names
save_path = 'experiments\\ablation\\cls.png'
plot_model_comparison(csv_path, selected_columns, metric_name, bar_names, save_path)

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# def plot_two_groups_comparison(csv_path, group1_columns, group2_columns, metric_name, bar_names, save_path):
#     # Load data from CSV
#     data = pd.read_csv(csv_path)
    
#     # Calculate means and standard deviations for both groups
#     means_group1 = data[group1_columns].mean()
#     stds_group1 = data[group1_columns].std()
#     means_group2 = data[group2_columns].mean()
#     stds_group2 = data[group2_columns].std()
    
#     # Define matplotlib parameters for aesthetics
#     plt.rcParams.update({
#         'font.size': 7,
#         'font.family': 'sans-serif',
#         'figure.figsize': (4, 3),
#         'axes.labelsize': 7,
#         'xtick.labelsize': 7,
#         'ytick.labelsize': 7
#     })

#     fig, ax = plt.subplots()
    
#     # Set positions for groups and bars
#     n_models = len(bar_names)
#     index_group1 = np.arange(n_models)  # MMSE model positions
#     index_group2 = np.arange(n_models) + n_models + 1  # MOCA model positions
#     bar_width = 1  # Width of the bars
    
#     # Color scheme for the bars
#     colors = ['lightseagreen', 'darkseagreen', 'peachpuff']
    
#     # Plotting MMSE bars and points
#     for i, (mean, std, col) in enumerate(zip(means_group1, stds_group1, group1_columns)):
#         ax.bar(index_group1[i], mean, bar_width, label=bar_names[i], color=colors[i],
#                yerr=std, capsize=5, edgecolor='none')        
#         y_mmse = data[col]
#         x_mmse = np.random.normal(index_group1[i], 0.04, size=len(y_mmse))
#         ax.scatter(x_mmse, y_mmse, color=colors[i], edgecolor='black', zorder=3, s=12)

#     # Plotting MOCA bars and points
#     for i, (mean, std, col) in enumerate(zip(means_group2, stds_group2, group2_columns)):
#         ax.bar(index_group2[i], mean, bar_width, color=colors[i], yerr=std, capsize=5, edgecolor='none')
#         y_moca = data[col]
#         x_moca = np.random.normal(index_group2[i], 0.04, size=len(y_moca))
#         ax.scatter(x_moca, y_moca, color=colors[i], edgecolor='black', zorder=3, s=12)

#     # Customizing the axes and plot layout
#     plt.subplots_adjust(left=0.18, right=0.95, top=0.85, bottom=0.18)
#     ax.set_ylabel(metric_name)
#     group1_center = (index_group1[-1] + index_group1[0]) / 2
#     group2_center = (index_group2[-1] + index_group2[0]) / 2
#     ax.set_xticks([group1_center, group2_center])
#     ax.tick_params(axis='both',length=0)
#     ax.set_xticklabels(['MMSE', 'MoCA'])
#     ax.set_ylim(0, 1.05)
#     ax.legend()
    
#     # Optionally save or show the plot
#     if save_path:
#         plt.savefig(save_path, format='png', dpi=300)
    
#     plt.show()

# ### Function call with specified parameters
# # plot_two_groups_comparison(
# #     csv_path='experiments\\ablation\\adreg.csv',
# #     group1_columns=['sss_mmse', 'mt_mmse', 'svm_mmse'],
# #     group2_columns=['sss_moca', 'mt_moca', 'svm_moca'],
# #     metric_name='PLCC',
# #     bar_names=['DeepAD', 'ViT', 'SVM'],
# #     save_path='experiments\\ablation\\adreg.png'
# # )
# plot_two_groups_comparison(
#     csv_path='experiments\\ablation\\adncreg.csv',
#     group1_columns=['sss_mmse', 'mt_mmse', 'svm_mmse'],
#     group2_columns=['sss_moca', 'mt_moca', 'svm_moca'],
#     metric_name='PLCC',
#     bar_names=['DeepAD', 'ViT', 'SVM'],
#     save_path='experiments\\ablation\\adncreg.png'
# )