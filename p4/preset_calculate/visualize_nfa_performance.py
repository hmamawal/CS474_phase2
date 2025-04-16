import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def load_data(csv_file):
    """Load performance data from CSV file."""
    return pd.read_csv(csv_file)

def analyze_complexity(sizes, times):
    """Calculate complexity using linear regression on log-log data."""
    log_sizes = np.log(sizes)
    log_times = np.log(times)
    slope, intercept = np.polyfit(log_sizes, log_times, 1)
    r_squared = np.corrcoef(log_sizes, log_times)[0, 1]**2
    return slope, intercept, r_squared

def plot_performance_graphs(data, output_dir=None):
    """Generate performance visualization graphs."""
    plt.figure(figsize=(12, 10))
    
    # Extract data
    sizes = data['size'].values
    mean_times = data['mean_time'].values
    std_devs = data['std_dev'].values
    
    # Plot 1: Average time vs size with error bars
    plt.subplot(2, 1, 1)
    plt.errorbar(sizes, mean_times, yerr=std_devs, fmt='o-', capsize=5)
    plt.xlabel('Number of states')
    plt.ylabel('Average time (seconds)')
    plt.title('NFA Finite Acceptance Check - Average Time with Standard Deviation')
    plt.grid(True)
    
    # Plot 2: Log-log plot to identify complexity class
    plt.subplot(2, 1, 2)
    plt.loglog(sizes, mean_times, 'o-', label='Measured times')
    
    # Calculate complexity
    slope, intercept, r_squared = analyze_complexity(sizes, mean_times)
    
    # Generate fitted line for visualization
    fit_line = np.exp(intercept) * np.array(sizes)**slope
    plt.loglog(sizes, fit_line, 'r--', 
               label=f'Fitted line: O(n^{slope:.2f}), R² = {r_squared:.4f}')
    
    plt.xlabel('Number of states (log scale)')
    plt.ylabel('Time (log scale)')
    plt.title('Log-Log Plot to Identify Computational Complexity')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save if output directory specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'nfa_performance.png'))
    
    plt.show()
    
    # Print summary statistics
    print_summary(data, slope, r_squared)

def print_summary(data, slope, r_squared):
    """Print summary statistics and complexity analysis."""
    print("\nSummary Statistics:")
    print("==================")
    for _, row in data.iterrows():
        size = row['size']
        print(f"Size {int(size)}:")
        print(f"  Mean time: {row['mean_time']:.6f} seconds")
        print(f"  Median time: {row['median_time']:.6f} seconds")
        print(f"  Std dev: {row['std_dev']:.6f} seconds")
        print(f"  95% CI: ({row['ci_lower']:.6f}, {row['ci_upper']:.6f})")
        print(f"  Finite results: {row['finite_percent']}%")
    
    print("\nComplexity Analysis:")
    print("==================")
    print(f"Measured complexity: O(n^{slope:.4f})")
    print(f"R-squared value: {r_squared:.4f}")
    
    # Interpret the complexity class
    if 0.8 <= slope <= 1.2:
        complexity = "approximately linear - O(n)"
    elif 1.8 <= slope <= 2.2:
        complexity = "approximately quadratic - O(n²)"
    elif 2.8 <= slope <= 3.2:
        complexity = "approximately cubic - O(n³)"
    else:
        complexity = f"O(n^{slope:.2f})"
    
    print(f"Algorithm appears to be {complexity}")
    print(f"Coefficient of determination (R²): {r_squared:.4f}")
    print(f"The R² value indicates how well the power law model fits the data (closer to 1.0 is better)")

if __name__ == "__main__":
    # File path
    csv_file = "nfa_performance_data_run1.csv"
    
    # Load data
    data = load_data(csv_file)
    
    # Create output directory
    output_dir = "results/visualization"
    
    # Generate plots and print summary
    plot_performance_graphs(data, output_dir)