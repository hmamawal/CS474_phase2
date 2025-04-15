import time
import csv
import numpy as np
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