"""
Visualization module for plotting and analyzing NFA performance.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

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

def generate_plots(test_sizes, means, stdevs, run_number, results_dir):
    """
    Generate plots for the NFA performance data.
    """
    # Plot 1: Average time vs size with error bars
    plt.figure(figsize=(12, 8))
    plt.errorbar(test_sizes, means, yerr=stdevs, fmt='o-', capsize=5)
    plt.xlabel('Number of states')
    plt.ylabel('Average time (seconds)')
    plt.title('NFA Finite Acceptance Check - Average Time with Standard Deviation')
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'Figure_1.png'))
    
    # Plot 2: Log-log plot to identify complexity class
    plt.figure(figsize=(12, 8))
    plt.loglog(test_sizes, means, 'o-', label='Measured times')
    
    # Calculate complexity
    complexity_info = analyze_complexity(test_sizes, means)
    
    # Generate fitted line for visualization
    fit_line = np.exp(complexity_info["intercept"]) * np.array(test_sizes)**complexity_info["slope"]
    plt.loglog(test_sizes, fit_line, 'r--', 
               label=f'Fitted line: O(n^{complexity_info["slope"]:.2f}), R² = {complexity_info["r_squared"]:.4f}')
    
    plt.xlabel('Number of states (log scale)')
    plt.ylabel('Time (log scale)')
    plt.title('Log-Log Plot to Identify Computational Complexity')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'Figure_2_big_o.png'))
    
    return complexity_info

def print_summary_statistics(test_sizes, all_results):
    """
    Print summary statistics of test results to terminal.
    """
    print("\nSummary Statistics:")
    print("==================")
    
    for i, size in enumerate(test_sizes):
        times = [r[2] for r in all_results[i]]
        finite_count = sum(1 for r in all_results[i] if r[1])
        print(f"Size {size}:")
        print(f"  Mean time: {np.mean(times):.6f} seconds")
        print(f"  Median time: {np.median(times):.6f} seconds")
        print(f"  Std dev: {np.std(times):.6f} seconds")
        print(f"  95% CI: ({np.mean(times) - 1.96*np.std(times)/np.sqrt(len(times)):.6f}, "
              f"{np.mean(times) + 1.96*np.std(times)/np.sqrt(len(times)):.6f})")
        print(f"  Finite results: {finite_count}/{len(all_results[i])} ({finite_count/len(all_results[i])*100:.1f}%)")

def print_complexity_analysis(complexity_info):
    """
    Print complexity analysis to terminal.
    """
    print("\nComplexity Analysis:")
    print("==================")
    print(f"Measured complexity: O(n^{complexity_info['slope']:.4f})")
    print(f"R-squared value: {complexity_info['r_squared']:.4f}")
    print(f"Algorithm appears to be {complexity_info['complexity']}")
    print(f"Coefficient of determination (R²): {complexity_info['r_squared']:.4f}")
    print(f"The R² value indicates how well the power law model fits the data (closer to 1.0 is better)")