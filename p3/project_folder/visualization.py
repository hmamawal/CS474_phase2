import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

def graph_results(results, outfile="graph.png", log_scale=True):
    """
    Create a comprehensive visualization of the experiment results.
    
    Parameters:
    - results: List of experiment results [size, index, satisfies_spec, time, mode]
    - outfile: Output file path for the graph image
    - log_scale: Whether to use log-log scaling
    """
    # Convert results to DataFrame for easier manipulation
    df = pd.DataFrame(results, columns=["Size", "DFA Index", "Satisfies Spec?", "Time (seconds)", "Mode"])
    
    # Set up the figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('DFA Suffix Inclusion Algorithm Performance Analysis', fontsize=16)
    
    # Plot 1: Scatterplot with trend lines
    ax = axes[0, 0]
    modes = df["Mode"].unique()
    colors = ['blue', 'green']
    markers = ['o', 's']
    
    for i, mode in enumerate(modes):
        mode_data = df[df["Mode"] == mode]
        # Scatter plot
        ax.scatter(
            mode_data["Size"], 
            mode_data["Time (seconds)"], 
            alpha=0.5, 
            label=f"{mode} DFAs", 
            color=colors[i], 
            marker=markers[i]
        )
        
        # Fit a curve to the data
        sizes = mode_data["Size"].unique()
        avg_times = [mode_data[mode_data["Size"] == size]["Time (seconds)"].mean() for size in sizes]
        
        # Calculate regression line
        if log_scale:
            # Linear regression on log-log scale
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                np.log(sizes), np.log(avg_times)
            )
            # Plot power law fit
            x_range = np.linspace(min(sizes), max(sizes), 100)
            y_fit = np.exp(intercept) * x_range ** slope
            ax.plot(
                x_range, 
                y_fit, 
                '--', 
                color=colors[i], 
                label=f'{mode} fit: O(n^{slope:.3f})'
            )
        else:
            # Linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                sizes, avg_times
            )
            # Plot linear fit
            x_range = np.linspace(min(sizes), max(sizes), 100)
            y_fit = intercept + slope * x_range
            ax.plot(
                x_range, 
                y_fit, 
                '--', 
                color=colors[i], 
                label=f'{mode} fit: {slope:.3e}x + {intercept:.3e}'
            )
    
    if log_scale:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title('Computation Time vs DFA Size (Log-Log Scale)')
    else:
        ax.set_title('Computation Time vs DFA Size')
    ax.set_xlabel('Number of States')
    ax.set_ylabel('Computation Time (seconds)')
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend()
    
    # Plot 2: Box plot by size and mode
    ax = axes[0, 1]
    sizes = sorted(df["Size"].unique())
    box_positions = []
    box_data = []
    box_colors = []
    labels = []
    
    width = 0.35  # Width of box plots
    
    for i, mode in enumerate(modes):
        for size in sizes:
            mode_size_data = df[(df["Mode"] == mode) & (df["Size"] == size)]
            if not mode_size_data.empty:
                box_positions.append(size + (i - 0.5 + 0.5) * width)
                box_data.append(mode_size_data["Time (seconds)"].values)
                box_colors.append(colors[i])
                labels.append(f"{size}-{mode[:3]}")
    
    ax.boxplot(box_data, positions=box_positions, widths=width, patch_artist=True,
               boxprops=dict(alpha=0.6), medianprops=dict(color='red'))
    
    for i, box in enumerate(ax.artists):
        box.set_facecolor(box_colors[i % len(colors)])
    
    ax.set_title('Distribution of Computation Times by Size and Mode')
    ax.set_xlabel('DFA Size')
    ax.set_ylabel('Time (seconds)')
    ax.set_xscale('log' if log_scale else 'linear')
    ax.grid(True, which="both", ls="--", alpha=0.5)
    
    # Add a legend for the box plot
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[i], label=modes[i]) for i in range(len(modes))]
    ax.legend(handles=legend_elements)
    
    # Plot 3: Bar chart with error bars showing mean and std dev
    ax = axes[1, 0]
    bar_width = 0.35
    bar_positions = np.arange(len(sizes))
    
    for i, mode in enumerate(modes):
        means = []
        std_devs = []
        
        for size in sizes:
            mode_size_data = df[(df["Mode"] == mode) & (df["Size"] == size)]
            means.append(mode_size_data["Time (seconds)"].mean())
            std_devs.append(mode_size_data["Time (seconds)"].std())
        
        positions = bar_positions + (i - 0.5 + 0.5) * bar_width
        
        ax.bar(positions, means, bar_width, label=mode, alpha=0.7, color=colors[i])
        ax.errorbar(positions, means, yerr=std_devs, fmt='none', capsize=5, ecolor='black', alpha=0.7)
    
    ax.set_xlabel('DFA Size')
    ax.set_ylabel('Mean Computation Time (seconds)')
    ax.set_title('Mean Computation Time with Standard Deviation')
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(sizes, rotation=45)
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.legend()
    
    # Plot 4: Line plot showing scaling behavior
    ax = axes[1, 1]
    
    for i, mode in enumerate(modes):
        # Group and calculate means
        mode_sizes = []
        mode_means = []
        
        for size in sizes:
            mode_size_data = df[(df["Mode"] == mode) & (df["Size"] == size)]
            if not mode_size_data.empty:
                mode_sizes.append(size)
                mode_means.append(mode_size_data["Time (seconds)"].mean())
        
        ax.plot(mode_sizes, mode_means, 'o-', label=mode, color=colors[i], linewidth=2, markersize=8)
        
        # Add theoretical bounds
        # Assuming O(n^2) complexity for visualization
        scaled_factor = mode_means[-1] / (mode_sizes[-1]**2)
        theoretical_curve = [scaled_factor * (size**2) for size in mode_sizes]
        ax.plot(mode_sizes, theoretical_curve, '--', color=colors[i], alpha=0.5, 
                label=f'{mode} theoretical O(n²)')
    
    if log_scale:
        ax.set_xscale('log')
        ax.set_yscale('log')
    ax.set_xlabel('DFA Size')
    ax.set_ylabel('Mean Computation Time (seconds)')
    ax.set_title('Scaling Behavior Analysis')
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend()
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    return fig