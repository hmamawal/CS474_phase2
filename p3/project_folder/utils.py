import time
import csv
import numpy as np
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from dfa_algorithms import check_suffix_inclusion
from dfa_generator import generate_random_dfa, generate_random_dfa_satisfying_spec

def process_dfa(size, satisfy_spec, accept_prob):
    """
    Generates a DFA of the given size and tests it.
    Returns a tuple: (size, result, elapsed_time).
    """
    generation_start = time.perf_counter()
    
    # Split timing between generation and testing
    if satisfy_spec:
        D = generate_random_dfa_satisfying_spec(size)
    else:
        D = generate_random_dfa(size, accept_prob)
    
    generation_end = time.perf_counter()
    generation_time = generation_end - generation_start
    
    # Now time the actual testing
    test_start = time.perf_counter()
    result = check_suffix_inclusion(D)
    test_end = time.perf_counter()
    test_time = test_end - test_start
    
    total_time = generation_time + test_time
    
    # Only output detailed timing for large DFAs to avoid terminal spam
    if size >= 20000:
        mode = "satisfying" if satisfy_spec else "non-satisfying"
        print(f"  📊 DFA size {size} ({mode}): generation={generation_time:.3f}s, testing={test_time:.3f}s, total={total_time:.3f}s")
    
    return size, result, total_time

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
    
    batch_start = time.perf_counter()
    print(f"🔄 Starting batch: {count} {mode_name} DFAs of size {size}")
    
    # Process each DFA in the batch
    for i in range(count):
        index = start_index + i
        dfa_start = time.perf_counter()
        size_result, result, elapsed = process_dfa(size, satisfy_spec, accept_prob)
        results.append([size_result, index, result, elapsed, mode_name])
        
        # Show periodic updates for larger DFA sizes
        if size >= 30000 and (i+1) % max(1, count//5) == 0:  # Show progress at 20%, 40%, 60%, 80%, 100%
            percent_done = ((i+1) / count) * 100
            print(f"  → {mode_name} DFAs of size {size}: {percent_done:.0f}% complete ({i+1}/{count})")
    
    # Final status message for this batch
    batch_time = time.perf_counter() - batch_start
    total_time = sum(r[3] for r in results)
    avg_time = total_time / len(results) if results else 0
    
    print(f"✓ Completed {count} {mode_name} DFAs of size {size}")
    print(f"  → Average time: {avg_time:.6f}s per DFA")
    print(f"  → Batch total time: {batch_time:.3f}s")
    
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
    
    # Display global progress information
    print(f"🔄 Starting parallel processing of {len(tasks)} DFA sizes with {max_workers} workers")
    print(f"📊 Total DFAs to process: {len(tasks) * num_dfas_per_size}")
    
    # Create a progress bar for the overall process
    with tqdm(total=len(tasks), desc=f"Processing {mode_name} DFA sizes", unit="size") as pbar:
        # Process the tasks in parallel
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for i, result_list in enumerate(executor.map(process_dfa_batch_parallel, tasks)):
                all_results.extend(result_list)
                # Update progress bar after each size is completed
                pbar.update(1)
                # Display completion percentage
                percent_complete = (i + 1) / len(tasks) * 100
                print(f"Progress: {percent_complete:.1f}% complete ({i + 1}/{len(tasks)} sizes)")
                
                # For satisfying DFAs, add extra information about estimated time remaining
                if satisfy_spec and i < len(tasks) - 1:
                    # Calculate average time per DFA so far
                    avg_time = sum(r[3] for r in result_list) / len(result_list)
                    # Estimate remaining time
                    remaining_dfas = (len(tasks) - (i + 1)) * num_dfas_per_size
                    est_remaining_seconds = remaining_dfas * avg_time / max_workers
                    est_remaining_minutes = est_remaining_seconds / 60
                    print(f"⏱️  Estimated time remaining: {est_remaining_minutes:.1f} minutes")
    
    print(f"✅ Finished processing all {mode_name} DFAs")
    return all_results

def save_results_to_csv(results, filename="results.csv"):
    """
    Save benchmark results to a CSV file.
    """
    print(f"📝 Saving {len(results)} results to {filename}...")
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Size", "DFA Index", "Satisfies Spec?", "Time (seconds)", "Mode"])
        for row in results:
            writer.writerow(row)
    print(f"✅ Results saved successfully to {filename}")

def calculate_statistics(results):
    """
    Calculate statistics for each DFA size in the results.
    Returns a dictionary with size as key and statistics as values.
    """
    print("📊 Calculating statistics...")
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