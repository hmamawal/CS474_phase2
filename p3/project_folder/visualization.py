import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm
import os
import csv

def load_results_from_csv(csv_path):
    """
    Load results from a CSV file.
    Expected format: Size,DFA Index,Satisfies Spec?,Time (seconds),Mode
    """
    print(f"📊 Loading data from {csv_path}...")
    results = []
    
    try:
        with open(csv_path, 'r', newline='') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header row
            for row in reader:
                # Convert values to appropriate types
                size = int(row[0])
                index = int(row[1])
                result = row[2].lower() == 'true'
                time = float(row[3])
                mode = row[4] if len(row) > 4 else "default"
                results.append([size, index, result, time, mode])
                
        print(f"✅ Successfully loaded {len(results)} data points from {csv_path}")
        return results
    except Exception as e:
        print(f"❌ Error loading CSV file: {e}")
        return []

def analyze_and_save_plots(results, title="DFA Computation Time vs DFA Size", 
                         outfile="graph.png", log_scale=True, stats_display="text"):
    """
    Analyzes the time complexity and graphs the computation time statistics using matplotlib.
    Includes both regular and log-log plots with complexity analysis.
    
    Args:
        results: List of results in the format [size, index, result, elapsed_time, mode_name]
        title: Title for the graph
        outfile: Output file path for the graph image
        log_scale: Whether to include a log-log plot
        stats_display: Where to display statistics - "display" (on graph) or "text" (to file)
    """
    print(f"📊 Analyzing time complexity for {len(results)} data points...")
    
    # Check for different modes in the results
    modes = set(row[4] for row in results if len(row) > 4)
    
    # Organize results by size and mode
    data = {}
    for row in tqdm(results, desc="Processing result data", unit="entry"):
        size = row[0]
        time = row[3]
        mode = row[4] if len(row) > 4 else "default"
        
        if (size, mode) not in data:
            data[(size, mode)] = []
        data[(size, mode)].append(time)
    
    # Extract unique sizes and modes
    unique_sizes = sorted(set(key[0] for key in data.keys()))
    unique_modes = sorted(set(key[1] for key in data.keys()))
    
    print(f"Found {len(unique_sizes)} unique DFA sizes and {len(unique_modes)} modes.")
    
    # Generate main plot
    plt.figure(figsize=(12, 8))
    
    # Color mapping for different modes
    colors = ['b', 'r', 'g', 'c', 'm', 'y']
    
    # Process each mode separately
    complexity_results = {}
    for i, mode in enumerate(unique_modes):
        print(f"Plotting data for mode: {mode}")
        sizes = []
        means = []
        std_devs = []
        min_times = []
        max_times = []
        
        for size in unique_sizes:
            if (size, mode) in data:
                times = data[(size, mode)]
                sizes.append(size)
                means.append(np.mean(times))
                std_devs.append(np.std(times))
                min_times.append(np.min(times))
                max_times.append(np.max(times))
        
        color = colors[i % len(colors)]
        label = f"{mode} DFAs"
        
        # Plot average times with error bars showing min and max
        plt.errorbar(
            sizes, means,
            yerr=[
                np.array(means) - np.array(min_times),  
                np.array(max_times) - np.array(means)
            ],
            fmt=f'o-', color=color, capsize=5, label=label, alpha=0.7
        )
        
        # Analyze complexity for this mode
        if len(sizes) > 1:  # Need at least 2 points for regression
            complexity_results[mode] = analyze_complexity(sizes, means)
    
    # Set scales for main plot
    plt.title(title)
    plt.xlabel("DFA Size (number of states)")
    plt.ylabel("Computation Time (seconds)")
    plt.grid(True, which="both", ls="--", alpha=0.3)
    
    # Add legend only if multiple modes
    if len(unique_modes) > 1:
        plt.legend(loc='best')
    
    # Save main plot
    standard_plot_file = outfile
    plt.tight_layout()
    plt.savefig(standard_plot_file, dpi=300)
    print(f"✅ Standard plot saved to {standard_plot_file}")
    
    # Create log-log plot for complexity analysis if requested
    if log_scale:
        plt.figure(figsize=(12, 8))
        
        for i, mode in enumerate(unique_modes):
            sizes = []
            means = []
            
            for size in unique_sizes:
                if (size, mode) in data:
                    times = data[(size, mode)]
                    sizes.append(size)
                    means.append(np.mean(times))
            
            if len(sizes) > 1:  # Need at least 2 points
                color = colors[i % len(colors)]
                label = f"{mode} DFAs"
                complexity = complexity_results.get(mode, {})
                
                # Plot measured data points
                plt.loglog(sizes, means, f'o-', color=color, label=f"Measured {label}")
                
                # Add fitted line if we have complexity data
                if complexity:
                    fit_line = np.exp(complexity["intercept"]) * np.array(sizes)**complexity["slope"]
                    plt.loglog(
                        sizes, fit_line, '--', color='red',#color, 
                        label=f"Fitted: O(n^{complexity['slope']:.2f}), R²={complexity['r_squared']:.4f}"
                    )
        
        plt.title(f"{title} (Log-Log Scale)")
        plt.xlabel("DFA Size (log scale)")
        plt.ylabel("Computation Time (log scale)")
        plt.grid(True, which="both", ls="--", alpha=0.3)
        plt.legend(loc='best')
        
        # Save log-log plot
        log_plot_file = os.path.splitext(outfile)[0] + "_log_log.png"
        plt.tight_layout()
        plt.savefig(log_plot_file, dpi=300)
        print(f"✅ Log-log plot saved to {log_plot_file}")
    
    # Generate comprehensive statistics
    stats_file = os.path.splitext(outfile)[0] + "_statistics.txt"
    with open(stats_file, 'w') as f:
        f.write(f"TIME COMPLEXITY ANALYSIS FOR {title}\n")
        f.write("=" * 50 + "\n\n")
        
        # Write complexity analysis
        f.write("COMPLEXITY ANALYSIS:\n")
        f.write("-" * 30 + "\n")
        for mode, complexity in complexity_results.items():
            f.write(f"Mode: {mode}\n")
            f.write(f"  Measured complexity: O(n^{complexity['slope']:.4f})\n")
            f.write(f"  R-squared value: {complexity['r_squared']:.4f}\n")
            f.write(f"  Algorithm appears to be {complexity['complexity']}\n\n")
        
        # Write detailed statistics for each mode and size
        f.write("\nDETAILED STATISTICS:\n")
        f.write("-" * 30 + "\n")
        for mode in unique_modes:
            f.write(f"Mode: {mode}\n")
            f.write(f"{'Size':<10} {'Avg Time (s)':<15} {'Std Dev':<15} {'Min':<15} {'Max':<15} {'Median':<15} {'Count':<10}\n")
            
            for size in unique_sizes:
                if (size, mode) in data:
                    times = data[(size, mode)]
                    stats = {
                        'avg': np.mean(times),
                        'std_dev': np.std(times),
                        'min': np.min(times),
                        'max': np.max(times),
                        'median': np.median(times),
                        'count': len(times)
                    }
                    f.write(f"{size:<10} {stats['avg']:<15.6f} {stats['std_dev']:<15.6f} "
                            f"{stats['min']:<15.6f} {stats['max']:<15.6f} "
                            f"{stats['median']:<15.6f} {stats['count']:<10}\n")
            f.write("\n")
    
    print(f"📝 Detailed statistics and complexity analysis saved to {stats_file}")
    
    # Print a summary of complexity findings to the console
    print("\nCOMPLEXITY ANALYSIS SUMMARY:")
    print("=" * 30)
    for mode, complexity in complexity_results.items():
        print(f"Mode: {mode}")
        print(f"  Measured complexity: O(n^{complexity['slope']:.4f})")
        print(f"  R-squared value: {complexity['r_squared']:.4f}")
        print(f"  Algorithm appears to be {complexity['complexity']}")
    
    return complexity_results

def analyze_complexity(sizes, times):
    """
    Calculate complexity using linear regression on log-log data.
    """
    log_sizes = np.log(sizes)
    log_times = np.log(times)
    slope, intercept = np.polyfit(log_sizes, log_times, 1)
    r_squared = np.corrcoef(log_sizes, log_times)[0, 1]**2
    
    # Interpret the complexity class
    if 0.8 <= slope <= 1.2:
        complexity = "approximately linear - O(n)"
    elif 1.8 <= slope <= 2.2:
        complexity = "approximately quadratic - O(n²)"
    elif 2.8 <= slope <= 3.2:
        complexity = "approximately cubic - O(n³)"
    else:
        complexity = f"O(n^{slope:.2f})"
        
    return {
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_squared,
        "complexity": complexity
    }

def graph_results(results, title="DFA Computation Time vs DFA Size", outfile="graph.png", log_scale=False, stats_display="display"):
    """
    Graphs the computation time statistics for each DFA size using matplotlib.
    Includes options for log-log scaling and where to display statistics.
    
    Args:
        results: List of results in the format [size, index, result, elapsed_time, mode_name]
        title: Title for the graph
        outfile: Output file path for the graph image
        log_scale: Whether to use logarithmic scaling for axes
        stats_display: Where to display statistics - "display" (on graph) or "text" (to file)
    """
    print("⚠️ This function is deprecated in favor of analyze_and_save_plots which provides comprehensive complexity analysis.")
    print("Redirecting to analyze_and_save_plots...")
    return analyze_and_save_plots(results, title, outfile, log_scale, stats_display)