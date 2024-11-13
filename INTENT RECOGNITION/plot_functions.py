import matplotlib.pyplot as plt

def plot_name_metrics(df, name, metric='val_acc', show_runs=True, show_mean=True):
    arch_data = df[df['architecture_name'] == name]
    
    if arch_data.empty:
        print(f"No data found for architecture: {name}")
        return
    
    if show_mean == False and show_runs == False:
        print(f"You must plot something bro")
        return

    grouped = arch_data.groupby(['run_number', 'epoch'])[metric].mean().reset_index()

    mean_per_epoch = grouped.groupby('epoch')[metric].mean()

    plt.figure(figsize=(12, 6))

    if show_runs:
        for run in grouped['run_number'].unique():
            run_data = grouped[grouped['run_number'] == run]
            plt.plot(run_data['epoch'], run_data[metric], label=f'Run {run} {run_data[metric].values[-1]:.4f}', alpha=0.5, linestyle='--')

    if show_mean:
        plt.plot(mean_per_epoch.index, mean_per_epoch.values, label=f'Mean {mean_per_epoch.values[-1]:.4f}', color='red', linewidth=2)

    plt.title(f'{metric} Evolution for {name}')
    plt.xlabel('epoch')
    plt.ylabel(metric)
    plt.legend()
    plt.grid()
    plt.show()


def plot_highest_bar(df, metric='val_acc', top_n=False):
    max_metrics = df.groupby('architecture_name')[metric].max().reset_index()
    max_metrics = max_metrics.sort_values(by=metric, ascending=False)

    if top_n:
        max_metrics = max_metrics.head(top_n)
    
    colors = [
    "#FFADAD",  # Soft Pink
    "#FFD6A5",  # Peach
    "#FDFFB6",  # Light Yellow
    "#CAFFBF",  # Pale Green
    "#9BF6FF",  # Sky Blue
    "#A0C4FF",  # Light Periwinkle
    "#BDB2FF",  # Lavender
    "#FFC6FF",  # Light Magenta
    "#FFB5A7",  # Pastel Coral
    "#FCD5CE",  # Warm Blush
    "#D9F0FF",  # Powder Blue
    "#BEE1E6",  # Light Teal
    "#A2D2FF",  # Soft Blue
    "#FFAFCC",  # Pinkish Rose
    "#C3F0CA",  # Mint
    "#FFE5D9",  # Soft Apricot
    "#FDE2E4",  # Light Pinkish
    "#E2F0CB",  # Pale Lime
    "#FFCAD4",  # Faded Pink
    "#B8F2E6",  # Aqua Mint
    "#FFC8A2",  # Light Tangerine
    "#B6E0FE",  # Light Baby Blue
    "#D9C5FF",  # Pastel Lavender
    "#F9F3DF",  # Off White Yellow
    "#E4C1F9"   # Pale Lilac
]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(max_metrics['architecture_name'], max_metrics[metric], color=colors)

    for bar, value in zip(bars, max_metrics[metric]):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.4f}', 
                 ha='center', va='bottom')

    # Configure plot
    plt.title(f'Highest {metric} Reached for Each Architecture')
    plt.xlabel('Architecture Name')
    plt.ylabel(f'Max {metric}')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    legend_labels = [f"{name}: {value:.4f}" for name, value in zip(max_metrics['architecture_name'], max_metrics[metric])]
    plt.legend(bars, legend_labels, title='Architecture (Max Value)') #  bbox_to_anchor=(1, 1), loc='upper left'

    plt.tight_layout() # Es veu millor
    plt.show()

def plot_highest_bar_no_name(df, metric='val_acc', top_n=None):
    max_metrics = df.groupby('architecture_name')[metric].max().reset_index()
    max_metrics = max_metrics.sort_values(by=metric, ascending=False)
    
    if top_n:
        max_metrics = max_metrics.head(top_n)
    
    colors = [
        "#FFADAD", "#FFD6A5", "#FDFFB6", "#CAFFBF", "#9BF6FF", "#A0C4FF", "#BDB2FF", "#FFC6FF",
        "#FFB5A7", "#FCD5CE", "#D9F0FF", "#BEE1E6", "#A2D2FF", "#FFAFCC", "#C3F0CA", "#FFE5D9",
        "#FDE2E4", "#E2F0CB", "#FFCAD4", "#B8F2E6", "#FFC8A2", "#B6E0FE", "#D9C5FF", "#F9F3DF",
        "#E4C1F9"
    ]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(max_metrics['architecture_name'], max_metrics[metric], color=colors[:len(max_metrics)])

    for bar, value in zip(bars, max_metrics[metric]):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.4f}', 
                 ha='center', va='bottom')

    plt.title(f'Highest {metric} Reached for Each Architecture')
    plt.xlabel('')
    plt.ylabel(f'Max {metric}')
    plt.xticks([])
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    legend_labels = [f"{name}: {value:.4f}" for name, value in zip(max_metrics['architecture_name'], max_metrics[metric])]
    plt.legend(bars, legend_labels, title='Architecture (Max Value)', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

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
