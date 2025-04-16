"""
Data management module for handling file operations and results storage.
"""
import os
import csv
import numpy as np

def get_next_run_number():
    """Determine the next run number by checking existing directories."""
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        return 1
    
    existing_runs = [d for d in os.listdir(results_dir) 
                    if os.path.isdir(os.path.join(results_dir, d)) and d.startswith('run')]
    
    if not existing_runs:
        return 1
    
    run_numbers = [int(run.replace('run', '')) for run in existing_runs 
                  if run.replace('run', '').isdigit()]
    
    return max(run_numbers) + 1 if run_numbers else 1

def save_results_to_csv(all_results, test_sizes, run_number):
    """Save results to CSV files."""
    # Create results directory
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', f'run{run_number}')
    os.makedirs(results_dir, exist_ok=True)
    
    # Save individual trials data
    individual_file = os.path.join(results_dir, f'individual_times.csv')
    with open(individual_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['size', 'trial', 'finite', 'time'])
        for i, size_results in enumerate(all_results):
            for j, (size, finite, elapsed) in enumerate(size_results):
                writer.writerow([size, j+1, int(finite), elapsed])
    
    # Save summary statistics
    summary_file = os.path.join(results_dir, f'nfa_performance_data_run{run_number}.csv')
    with open(summary_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['size', 'mean_time', 'median_time', 'std_dev', 'ci_lower', 'ci_upper', 'finite_percent'])
        
        for i, size in enumerate(test_sizes):
            times = [r[2] for r in all_results[i]]
            finite_count = sum(1 for r in all_results[i] if r[1])
            mean_time = np.mean(times)
            std_dev = np.std(times)
            ci_lower = mean_time - 1.96 * std_dev / np.sqrt(len(times))
            ci_upper = mean_time + 1.96 * std_dev / np.sqrt(len(times))
            finite_percent = finite_count / len(all_results[i]) * 100
            
            writer.writerow([
                size,
                f"{mean_time:.6f}",
                f"{np.median(times):.6f}",
                f"{std_dev:.6f}",
                f"{ci_lower:.6f}",
                f"{ci_upper:.6f}",
                f"{finite_percent:.1f}"
            ])
    
    # Save summary statistics as text
    summary_text_file = os.path.join(results_dir, f'summary_statistics.txt')
    with open(summary_text_file, 'w') as f:
        f.write("Summary Statistics:\n")
        f.write("==================\n")
        
        for i, size in enumerate(test_sizes):
            times = [r[2] for r in all_results[i]]
            finite_count = sum(1 for r in all_results[i] if r[1])
            mean_time = np.mean(times)
            std_dev = np.std(times)
            ci_lower = mean_time - 1.96 * std_dev / np.sqrt(len(times))
            ci_upper = mean_time + 1.96 * std_dev / np.sqrt(len(times))
            
            f.write(f"Size {size}:\n")
            f.write(f"  Mean time: {mean_time:.6f} seconds\n")
            f.write(f"  Median time: {np.median(times):.6f} seconds\n")
            f.write(f"  Std dev: {std_dev:.6f} seconds\n")
            f.write(f"  95% CI: ({ci_lower:.6f}, {ci_upper:.6f})\n")
            f.write(f"  Finite results: {finite_count}/{len(all_results[i])} ({finite_count/len(all_results[i])*100:.1f}%)\n")
    
    return summary_file, results_dir