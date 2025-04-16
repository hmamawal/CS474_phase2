import time
import csv
import numpy as np
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dfa_algorithms import check_suffix_inclusion
from dfa_generator import generate_random_dfa, generate_random_dfa_satisfying_spec

def process_dfa(size, satisfy_spec, accept_prob):
    """
    Generates a DFA of the given size and tests it.
    Returns a tuple: (size, result, elapsed_time).
    """
    if satisfy_spec:
        D = generate_random_dfa_satisfying_spec(size)
    else:
        D = generate_random_dfa(size, accept_prob)
    start = time.perf_counter()
    result = check_suffix_inclusion(D)
    end = time.perf_counter()
    elapsed = end - start
    return size, result, elapsed

def process_dfa_batch_parallel(params):
    """
    Process a batch of DFAs in parallel.
    
    Args:
        params: A tuple of (size, satisfy_spec, accept_prob, count, start_index)
        
    Returns:
        List of results in the format [size, index, result, elapsed_time, mode_name]
    """
    size, satisfy_spec, accept_prob, count, start_index, mode_name = params
    results = []
    
    for i in range(count):
        index = start_index + i
        size_result, result, elapsed = process_dfa(size, satisfy_spec, accept_prob)
        results.append([size_result, index, result, elapsed, mode_name])
    
    return results

def process_dfas_in_parallel(sizes, num_dfas_per_size, satisfy_spec, accept_prob, mode_name, max_workers=None):
    """
    Process DFAs in parallel using multiple processes.
    
    Args:
        sizes: List of DFA sizes to test
        num_dfas_per_size: Number of DFAs to generate per size
        satisfy_spec: Whether to generate DFAs that satisfy the spec
        accept_prob: Probability of accepting states in random DFAs
        mode_name: Name of the mode (for reporting)
        max_workers: Maximum number of worker processes to use
        
    Returns:
        List of results in the format [size, index, result, elapsed_time, mode_name]
    """
    # Determine max_workers based on CPU count if not provided
    if max_workers is None:
        max_workers = multiprocessing.cpu_count()
    
    all_results = []
    
    # Prepare tasks for parallel processing - one task per DFA size
    tasks = []
    for size in sizes:
        # Process all DFAs of a given size in one task
        tasks.append((size, satisfy_spec, accept_prob, num_dfas_per_size, 1, mode_name))
    
    # Process the tasks in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results_lists = list(executor.map(process_dfa_batch_parallel, tasks))
        
    # Flatten the results
    for result_list in results_lists:
        all_results.extend(result_list)
    
    return all_results

def save_results_to_csv(results, filename="results.csv"):
    """
    Save benchmark results to a CSV file.
    """
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Size", "DFA Index", "Satisfies Spec?", "Time (seconds)", "Mode"])
        for row in results:
            writer.writerow(row)

def calculate_statistics(results):
    """
    Calculate statistics for each DFA size in the results.
    Returns a dictionary with size as key and statistics as values.
    """
    stats = {}
    # Group results by size
    size_groups = {}
    for row in results:
        size = row[0]
        time = row[3]
        if size not in size_groups:
            size_groups[size] = []
        size_groups[size].append(time)
    
    # Calculate statistics for each size
    for size, times in size_groups.items():
        times_array = np.array(times)
        stats[size] = {
            'mean': np.mean(times_array),
            'std_dev': np.std(times_array),
            'min': np.min(times_array),
            'max': np.max(times_array),
            'median': np.median(times_array),
            'count': len(times_array)
        }
    
    return stats