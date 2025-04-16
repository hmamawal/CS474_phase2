import argparse
import random
import numpy as np
import time
from datetime import timedelta
from tqdm import tqdm
from utils import process_dfa, save_results_to_csv, calculate_statistics, process_dfas_in_parallel
from visualization import graph_results
import multiprocessing

def format_time(seconds):
    """Format time in a human-readable format"""
    return str(timedelta(seconds=int(seconds)))

def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(
        description="Test DFA suffix inclusion algorithm on random DFAs."
    )
    parser.add_argument("--sizes", nargs="+", type=int, 
                        default=[5000, 7500, 10000, 15000, 20000, 30000, 40000, 50000, 75000, 100000],
                        help="List of DFA sizes to test")
    parser.add_argument("--num_dfas", type=int, default=30,
                        help="Number of DFAs to generate per size")
    parser.add_argument("--accept_prob", type=float, default=0.2,
                        help="Probability that a state is accepting in randomly generated DFAs")
    parser.add_argument("--run_both_modes", action="store_true", default=True,
                        help="Run experiments with both satisfying and non-satisfying DFAs")
    parser.add_argument("--satisfy_spec", action="store_true",
                        help="Generate DFAs that satisfy the specification (L(D) ⊆ L(S)).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--outfile", type=str, default="results.csv",
                        help="Output CSV file for results")
    parser.add_argument("--graph_out", type=str, default="graph.png",
                        help="Output file for the graph image")
    parser.add_argument("--log_scale", action="store_true", default=True,
                        help="Plot results using log-log scale")
    parser.add_argument("--parallel", action="store_true", default=True,
                        help="Enable parallel processing")
    parser.add_argument("--max_workers", type=int, default=None,
                        help="Maximum number of worker processes for parallel execution")
    parser.add_argument("--progress", action="store_true", default=True,
                        help="Show detailed progress information")
    args = parser.parse_args()
    
    # Print experiment configuration
    print("\n" + "="*60)
    print("📋 DFA SUFFIX INCLUSION EXPERIMENT CONFIGURATION")
    print("="*60)
    print(f"🔢 DFA sizes to test: {args.sizes}")
    print(f"🔄 Number of DFAs per size: {args.num_dfas}")
    print(f"🎲 Random seed: {args.seed}")
    print(f"📊 Accept probability: {args.accept_prob}")
    print(f"📈 Using log scale for plotting: {args.log_scale}")
    print(f"⚙️ Parallel processing: {args.parallel}")
    print("="*60 + "\n")
    
    # Estimate total execution time (very rough estimate)
    total_dfas = len(args.sizes) * args.num_dfas
    if args.run_both_modes:
        total_dfas *= 2
    est_time_per_dfa = 0.05  # seconds, rough estimate
    est_total_time = total_dfas * est_time_per_dfa
    if args.parallel:
        cpu_count = multiprocessing.cpu_count()
        max_workers = args.max_workers if args.max_workers is not None else cpu_count
        est_total_time /= max_workers
    
    print(f"⏱️ Estimated execution time: {format_time(est_total_time)} (rough estimate)")
    
    random.seed(args.seed)
    sizes = args.sizes
    num_dfas_per_size = args.num_dfas
    all_results = []
    
    if args.run_both_modes:
        # Run with both modes - satisfying and non-satisfying DFAs
        modes = [False, True]
        mode_names = ["non-satisfying", "satisfying"]
    else:
        # Run with just one mode based on --satisfy_spec
        modes = [args.satisfy_spec]
        mode_names = ["non-satisfying" if not args.satisfy_spec else "satisfying"]
    
    # Print CPU information
    cpu_count = multiprocessing.cpu_count()
    max_workers = args.max_workers if args.max_workers is not None else cpu_count
    print(f"🖥️ System has {cpu_count} CPU cores available.")
    print(f"🧵 Using up to {max_workers} worker processes for parallel execution.\n")
    
    experiment_start = time.time()
    
    # Process DFAs for each mode
    for mode_idx, satisfy_spec in enumerate(modes):
        mode_name = mode_names[mode_idx]
        results = []
        
        mode_start = time.time()
        print(f"\n{'='*30}")
        print(f"🔄 PROCESSING {mode_name.upper()} DFAs:")
        print(f"{'='*30}")
        
        if args.parallel:
            # Process DFAs in parallel
            print(f"🚀 Using parallel processing with up to {max_workers} workers")
            
            # We don't use tqdm here because process_dfas_in_parallel handles all sizes at once
            results = process_dfas_in_parallel(
                sizes, 
                num_dfas_per_size, 
                satisfy_spec, 
                args.accept_prob,
                mode_name,
                max_workers
            )
            
            # Extract elapsed times for overall statistics
            overall_times = [row[3] for row in results]
            
        else:
            # Process DFAs sequentially (original method)
            overall_times = []
            total_dfas = len(sizes) * num_dfas_per_size
            pbar = tqdm(total=total_dfas, desc=f"Processing {mode_name} DFAs")
            
            for size in sizes:
                size_times = []
                for index in range(num_dfas_per_size):
                    size, result, elapsed = process_dfa(size, satisfy_spec, args.accept_prob)
                    results.append([size, index + 1, result, elapsed, mode_name])
                    overall_times.append(elapsed)
                    size_times.append(elapsed)
                    pbar.update(1)
                
                # Print progress after each size is complete
                if args.progress:
                    avg_time = sum(size_times) / len(size_times)
                    print(f"✓ Completed size {size}: Avg time = {avg_time:.6f}s")
            
            pbar.close()
        
        # Calculate statistics for this mode
        stats = calculate_statistics(results)
        print(f"\n📊 Statistics for {mode_name} DFAs:")
        for size in sizes:
            size_stats = stats[size]
            print(f"  Size {size}: Mean = {size_stats['mean']:.6f}s, Std Dev = {size_stats['std_dev']:.6f}s")
            if args.progress:
                print(f"  → Min = {size_stats['min']:.6f}s, Max = {size_stats['max']:.6f}s")
        
        # Save results for this mode to separate file
        mode_suffix = "satisfying" if satisfy_spec else "non_satisfying"
        save_results_to_csv(results, f"results_{mode_suffix}.csv")
        
        # Mode timing information
        mode_elapsed = time.time() - mode_start
        print(f"\n⏱️ {mode_name} processing completed in {format_time(mode_elapsed)}")
        
        # Add results to combined list
        all_results.extend(results)
    
    # Save combined results if both modes were run
    if args.run_both_modes:
        save_results_to_csv(all_results, args.outfile)
        print(f"📊 Combined results saved to {args.outfile}")
    
    # Graph all results together
    graph_results(all_results, outfile=args.graph_out, log_scale=args.log_scale)
    print(f"📈 Graph saved to {args.graph_out}")
    
    # Print final timing information
    total_elapsed = time.time() - experiment_start
    print("\n" + "="*60)
    print(f"✅ EXPERIMENT COMPLETED SUCCESSFULLY!")
    print(f"⏱️ Total execution time: {format_time(total_elapsed)}")
    print(f"📊 Processed {len(all_results)} DFAs across {len(sizes)} different sizes.")
    if args.run_both_modes:
        print(f"🔄 Tested both satisfying and non-satisfying DFAs.")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()