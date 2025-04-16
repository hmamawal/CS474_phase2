"""
Main module for the NFA finite acceptance checker.
"""
import random
import time
import numpy as np
from models import SPEC_DFA
from generator import generate_random_nfa
from algorithms import check_finite_acceptance
from data_manager import get_next_run_number, save_results_to_csv
from visualization import generate_plots, print_summary_statistics, print_complexity_analysis
from tests import run_unit_tests

def run_performance_tests(num_trials=30):
    """
    Run performance tests on NFA finite acceptance checking algorithm.
    
    Args:
        num_trials: Number of trials to run for each size.
        
    Returns:
        List of test results for each size.
    """
    run_number = get_next_run_number()
    print(f"Starting test run #{run_number}")
    
    test_sizes = [1000, 5000, 10000, 25000]#, 50000, 100000, 250000, 500000]
    all_results = []
    
    for size in test_sizes:
        size_results = []
        print(f"Testing NFAs with {size} states...")
        
        for trial in range(num_trials):
            # Use different seeds for true randomness
            random.seed(trial * 100 + size)  # Different seed each trial
            nfa = generate_random_nfa(size, "AC0")
            
            start_time = time.perf_counter()
            finite = check_finite_acceptance(nfa, SPEC_DFA)
            elapsed = time.perf_counter() - start_time
            
            size_results.append((size, finite, elapsed))
            print(f"  Trial {trial+1}/{num_trials}: {elapsed:.6f} seconds")
        
        all_results.append(size_results)
    
    # Process and save the results
    summary_file, results_dir = save_results_to_csv(all_results, test_sizes, run_number)
    
    # Calculate means and standard deviations for plotting
    means = []
    stdevs = []
    for size_results in all_results:
        times = [r[2] for r in size_results]
        means.append(np.mean(times))
        stdevs.append(np.std(times))
    
    # Generate plots and get complexity info
    complexity_info = generate_plots(test_sizes, means, stdevs, run_number, results_dir)
    
    # Print summary statistics and complexity analysis
    print_summary_statistics(test_sizes, all_results)
    print_complexity_analysis(complexity_info)
    
    print(f"\nResults saved to: {results_dir}")
    
    return all_results

if __name__ == '__main__':
    # Run unit tests first to verify correctness
    run_unit_tests()
    
    # Then run the performance tests
    run_performance_tests()