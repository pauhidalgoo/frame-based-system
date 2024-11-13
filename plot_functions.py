import matplotlib.pyplot as plt

def plot_name_metrics(df, name, metric='val_acc', group_by=None, show_runs=True, show_mean=True):
    """
    Plot metrics for a given architecture name with options for grouping and displaying individual runs.
    
    Parameters:
    - df (DataFrame): The dataset containing metrics.
    - name (str): The architecture name to filter data.
    - metric (str): The metric to plot, e.g., 'val_acc' or 'val_f1'.
    - group_by (list): Columns to group by, in addition to 'epoch'. Default is ['epoch'] if None.
    - show_runs (bool): Whether to show individual runs.
    - show_mean (bool): Whether to show mean over all runs.
    """
    arch_data = df[df['architecture_name'] == name]
    
    if arch_data.empty:
        print(f"No data found for architecture: {name}")
        return
    
    if not show_mean and not show_runs:
        print("You must plot something.")
        return
    
    # Set default grouping if group_by is not provided
    if group_by is None:
        group_by = ['epoch']
    else:
        # Ensure 'epoch' is always included in the grouping
        if 'epoch' not in group_by:
            group_by.append('epoch')
    
    # Group by the specified columns and calculate mean of the metric
    grouped = arch_data.groupby(group_by)[metric].mean().reset_index()
    
    # Calculate mean per epoch if showing mean
    mean_per_epoch = grouped.groupby('epoch')[metric].mean()
    
    plt.figure(figsize=(12, 6))
    
    # Plot individual runs if requested
    if show_runs:
        unique_combinations = grouped.drop_duplicates(subset=[col for col in group_by if col != 'epoch'])
        for _, combination in unique_combinations.iterrows():
            condition = (grouped[list(combination.index)] == combination.values).all(axis=1)
            run_data = grouped[condition]
            label = " - ".join([f"{col}={val}" for col, val in combination.items() if col != 'epoch'])
            plt.plot(run_data['epoch'], run_data[metric], label=f'Run {label} {run_data[metric].values[-1]:.4f}', alpha=0.5, linestyle='--')
    
    # Plot mean over all runs if requested
    if show_mean:
        plt.plot(mean_per_epoch.index, mean_per_epoch.values, label=f'Mean {mean_per_epoch.values[-1]:.4f}', color='red', linewidth=2)
    
    plt.title(f'{metric} Evolution for {name}')
    plt.xlabel('Epoch')
    plt.ylabel(metric)
    plt.legend()
    plt.grid()
    plt.show()


def plot_highest_bar(df, metric='val_acc', group_by='architecture_name'):
    """
    Plots a bar chart of the highest values of a given metric, grouped by a specified column.
    
    Parameters:
    - df (DataFrame): Dataset containing metrics.
    - metric (str): The metric to plot (e.g., 'val_acc').
    - group_by (str): The column to group by (default is 'architecture_name').
    """
    # Group by the specified column and get the max of the metric
    max_metrics = df.groupby(group_by)[metric].max().reset_index()
    max_metrics = max_metrics.sort_values(by=metric, ascending=False)
    
    # Color palette for the bars
    colors = [
        "#FFADAD", "#FFD6A5", "#FDFFB6", "#CAFFBF", "#9BF6FF",
        "#A0C4FF", "#BDB2FF", "#FFC6FF", "#FFB5A7", "#FCD5CE",
        "#D9F0FF", "#BEE1E6", "#A2D2FF", "#FFAFCC", "#C3F0CA",
        "#FFE5D9", "#FDE2E4", "#E2F0CB", "#FFCAD4", "#B8F2E6",
        "#FFC8A2", "#B6E0FE", "#D9C5FF", "#F9F3DF", "#E4C1F9"
    ]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(max_metrics[group_by], max_metrics[metric], color=colors[:len(max_metrics)])

    # Add labels above each bar
    for bar, value in zip(bars, max_metrics[metric]):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.4f}', 
                 ha='center', va='bottom')

    # Configure plot
    plt.title(f'Highest {metric} Reached for Each {group_by.capitalize()}')
    plt.xlabel(group_by.replace('_', ' ').capitalize())
    plt.ylabel(f'Max {metric}')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    legend_labels = [f"{name}: {value:.4f}" for name, value in zip(max_metrics[group_by], max_metrics[metric])]
    plt.legend(bars, legend_labels, title=f'{group_by.capitalize()} (Max Value)', loc='upper left')

    plt.tight_layout()
    plt.show()


def plot_highest_bar_old(df, metric='val_acc', filter_substring=None):
    # Filter data by the given substring if specified
    if filter_substring:
        df = df[df['architecture_name'].str.contains(filter_substring, case=False)]
    
    # Group by architecture name and find the maximum value for the specified metric
    max_metrics = df.groupby('architecture_name')[metric].max().reset_index()
    max_metrics = max_metrics.sort_values(by=metric, ascending=False)
    
    # Define colors
    colors = [
        "#FFADAD",  "#FFD6A5", "#FDFFB6", "#CAFFBF", "#9BF6FF", "#A0C4FF", 
        "#BDB2FF",  "#FFC6FF", "#FFB5A7", "#FCD5CE", "#D9F0FF", "#BEE1E6",
        "#A2D2FF",  "#FFAFCC", "#C3F0CA", "#FFE5D9", "#FDE2E4", "#E2F0CB", 
        "#FFCAD4",  "#B8F2E6", "#FFC8A2", "#B6E0FE", "#D9C5FF", "#F9F3DF", 
        "#E4C1F9"
    ]

    # Create the bar plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(max_metrics['architecture_name'], max_metrics[metric], color=colors[:len(max_metrics)])

    # Add value labels to each bar
    for bar, value in zip(bars, max_metrics[metric]):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.4f}', 
                 ha='center', va='bottom')

    # Configure plot
    plt.title(f'Highest {metric} Reached for Each Architecture' + 
              (f" (Filtered by '{filter_substring}')" if filter_substring else ''))
    plt.xlabel('Architecture Name')
    plt.ylabel(f'Max {metric}')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Create legend labels
    legend_labels = [f"{name}: {value:.4f}" for name, value in zip(max_metrics['architecture_name'], max_metrics[metric])]
    plt.legend(bars, legend_labels, title='Architecture (Max Value)') 

    plt.tight_layout()
    plt.show()


def plot_compare_architectures(df, architecture_names=None, metric='val_acc', best_run=True):
    plt.figure(figsize=(12, 6))
    if architecture_names is None:
        architecture_names = df['architecture_name'].unique()

    for arch in architecture_names:
        arch_data = df[df['architecture_name'] == arch]
        
        if arch_data.empty:
            print(f"No data found for architecture: {arch}")
            continue

        grouped = arch_data.groupby(['run_number', 'epoch'])[metric].mean().reset_index()

        if best_run:
            if 'loss' in metric.lower():
                best_run_number = grouped.groupby('run_number')[metric].last().idxmin()
            else:
                best_run_number = grouped.groupby('run_number')[metric].last().idxmax()

            run_data = grouped[grouped['run_number'] == best_run_number]
            label = f'{arch} - Final {metric}: {run_data[metric].values[-1]:.4f}'
        else:
            mean_per_epoch = grouped.groupby('epoch')[metric].mean()
            run_data = pd.DataFrame({'epoch': mean_per_epoch.index, metric: mean_per_epoch.values})
            label = f'{arch} - Final {metric}: {mean_per_epoch.values[-1]:.4f}'

        if len(run_data) == 1:
            plt.scatter(run_data['epoch'], run_data[metric], label=label, s=100)
        else:
            plt.plot(run_data['epoch'], run_data[metric], label=label, linewidth=2)

    plt.title(f'Comparison of {metric} Across Architectures')
    plt.xlabel('Epoch')
    plt.ylabel(metric)
    plt.legend()
    plt.grid()
    plt.show()

def plot_train_val_comparison(df, architecture_name, metric='acc', comparison_type='mean'):
    assert metric in ["acc", "f1", "loss"], "The metric should be 'acc', 'loss' or 'f1'"
    train_metric = f'training_{metric}'
    val_metric = f'val_{metric}'

    arch_data = df[df['architecture_name'] == architecture_name]

    if arch_data.empty:
        print(f"No data found for architecture: {architecture_name}")
        return
    
    grouped = arch_data.groupby(['run_number', 'epoch'])[[train_metric, val_metric]].mean().reset_index()

    plt.figure(figsize=(12, 6))

    if comparison_type == 'mean':
        mean_per_epoch = grouped.groupby('epoch')[[train_metric, val_metric]].mean()
        plt.plot(mean_per_epoch.index, mean_per_epoch[train_metric], label=f'Mean Training {mean_per_epoch[f"training_{metric}"].values[-1]:.4f}', color='blue', linewidth=2)
        plt.plot(mean_per_epoch.index, mean_per_epoch[val_metric], label=f'Mean Validation {mean_per_epoch[f"val_{metric}"].values[-1]:.4f}', color='red', linewidth=2)

    elif comparison_type == 'best':
        last_epoch = grouped['epoch'].max()
        best_run = grouped[grouped['epoch'] == last_epoch].sort_values(val_metric, ascending=False).iloc[0]['run_number']
        best_run_data = grouped[grouped['run_number'] == best_run]
        plt.plot(best_run_data['epoch'], best_run_data[train_metric], label=f'Best Training (Run {int(best_run)}, {metric} {best_run_data[train_metric].values[-1]:.4f})', color='blue', linewidth=2)
        plt.plot(best_run_data['epoch'], best_run_data[val_metric], label=f'Best Validation (Run {int(best_run)}, {metric}  {best_run_data[val_metric].values[-1]:.4f})', color='red', linewidth=2)

    elif comparison_type == 'all':
        for run in grouped['run_number'].unique():
            run_data = grouped[grouped['run_number'] == run]
            plt.plot(run_data['epoch'], run_data[train_metric], alpha=0.3, linestyle='--', color='blue')
            plt.plot(run_data['epoch'], run_data[val_metric], alpha=0.3, linestyle='--', color='red')
        mean_per_epoch = grouped.groupby('epoch')[[train_metric, val_metric]].mean()

        label = f'Mean Training {metric}: {mean_per_epoch[f"training_{metric}"].values[-1]:.4f}'
        plt.plot(mean_per_epoch.index, mean_per_epoch[train_metric], label=label, color='blue', linewidth=2)
        label = f'Mean Validation {metric}: {mean_per_epoch[f"val_{metric}"].values[-1]:.4f}'
        plt.plot(mean_per_epoch.index, mean_per_epoch[val_metric], label=label, color='red', linewidth=2)

    # Configure plot
    plt.title(f'Training vs. Validation {metric.capitalize()} Evolution for {architecture_name}')
    plt.xlabel('Epoch')
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.grid()
    plt.show()
