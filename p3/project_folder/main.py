import argparse
import random
import numpy as np
import time
from datetime import timedelta
from tqdm import tqdm
from utils import process_dfa, save_results_to_csv, calculate_statistics, process_dfas_in_parallel
from visualization import graph_results, load_results_from_csv
import multiprocessing
import os

def format_time(seconds):
    """Format time in a human-readable format"""
    return str(timedelta(seconds=int(seconds)))

def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(
        description="Test DFA suffix inclusion algorithm on random DFAs."
    )
    # Common parameters
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
    parser.add_argument("--stats_display", type=str, choices=["display", "text"], default="display",
                        help="Where to display processing statistics: 'display' (on graph) or 'text' (to file)")
    parser.add_argument("--csv_to_graph", type=str, default=None,
                        help="Path to an existing CSV file to graph without processing DFAs")
    
    # DFA mode selection - added new group for mode selection
    mode_group = parser.add_argument_group('DFA Mode Selection')
    mode_group.add_argument("--mode", type=str, choices=['non_satisfying', 'satisfying', 'both'], 
                           default='non_satisfying',
                           help="Which DFA types to run: non_satisfying, satisfying, or both")
    
    # Non-satisfying DFA parameters
    non_satisfying_group = parser.add_argument_group('Non-satisfying DFA Parameters')
    non_satisfying_group.add_argument("--non_satisfying_sizes", nargs="+", type=int, 
                                     default=[5000, 7500, 10000, 15000, 20000, 30000, 40000, 50000, 75000, 100000],
                                     help="List of DFA sizes to test for non-satisfying DFAs")
    non_satisfying_group.add_argument("--non_satisfying_num_dfas", type=int, default=30,
                                     help="Number of non-satisfying DFAs to generate per size")
    non_satisfying_group.add_argument("--accept_prob", type=float, default=0.2,
                                     help="Probability that a state is accepting in randomly generated DFAs")
    
    # Satisfying DFA parameters 
    satisfying_group = parser.add_argument_group('Satisfying DFA Parameters')
    satisfying_group.add_argument("--satisfying_sizes", nargs="+", type=int, 
                                 default=[100, 200, 300, 500, 1000, 2000],
                                 help="List of DFA sizes to test for satisfying DFAs")
    satisfying_group.add_argument("--satisfying_num_dfas", type=int, default=10,
                                 help="Number of satisfying DFAs to generate per size")
    
    args = parser.parse_args()
    
    # Check if we're only generating a graph from a CSV file
    if args.csv_to_graph:
        if not os.path.exists(args.csv_to_graph):
            print(f"❌ ERROR: CSV file not found: {args.csv_to_graph}")
            return
            
        print("\n" + "="*60)
        print("📊 GENERATING GRAPH FROM EXISTING CSV FILE")
        print("="*60)
        print(f"📂 CSV File: {args.csv_to_graph}")
        print(f"📈 Output Graph: {args.graph_out}")
        print(f"📊 Statistics Display: {args.stats_display}")
        print(f"🔢 Log Scale: {args.log_scale}")
        print("="*60 + "\n")
        
        results = load_results_from_csv(args.csv_to_graph)
        if results:
            # Count accepting DFAs in the CSV file
            accepting_count = sum(1 for row in results if row[2] == True)
            total_dfas = len(results)
            print(f"📊 Found {accepting_count} accepting DFAs out of {total_dfas} total ({accepting_count/total_dfas*100:.1f}%)")
            
            graph_results(results, outfile=args.graph_out, log_scale=args.log_scale, 
                          stats_display=args.stats_display)
            print(f"✅ Graph generation completed successfully")
        return
    
    # Print experiment configuration
    print("\n" + "="*60)
    print("📋 DFA SUFFIX INCLUSION EXPERIMENT CONFIGURATION")
    print("="*60)
    print(f"🔄 Mode: {args.mode}")
    
    if args.mode in ['non_satisfying', 'both']:
        print(f"🔢 Non-satisfying DFA sizes: {args.non_satisfying_sizes}")
        print(f"🔄 Number of non-satisfying DFAs per size: {args.non_satisfying_num_dfas}")
        print(f"📊 Accept probability: {args.accept_prob}")
    
    if args.mode in ['satisfying', 'both']:
        print(f"🔢 Satisfying DFA sizes: {args.satisfying_sizes}")
        print(f"🔄 Number of satisfying DFAs per size: {args.satisfying_num_dfas}")
    
    print(f"🎲 Random seed: {args.seed}")
    print(f"📈 Using log scale for plotting: {args.log_scale}")
    print(f"⚙️ Parallel processing: {args.parallel}")
    print(f"📊 Statistics display mode: {args.stats_display}")
    print("="*60 + "\n")
    
    # Estimate total execution time (very rough estimate)
    total_dfas = 0
    if args.mode in ['non_satisfying', 'both']:
        total_dfas += len(args.non_satisfying_sizes) * args.non_satisfying_num_dfas
    
    if args.mode in ['satisfying', 'both']:
        total_dfas += len(args.satisfying_sizes) * args.satisfying_num_dfas
        
    est_time_per_dfa = 0.05  # seconds, rough estimate for non-satisfying
    est_time_per_satisfying_dfa = 0.5  # seconds, rough estimate for satisfying
    
    est_total_time = 0
    if args.mode in ['non_satisfying', 'both']:
        est_total_time += len(args.non_satisfying_sizes) * args.non_satisfying_num_dfas * est_time_per_dfa
    
    if args.mode in ['satisfying', 'both']:
        est_total_time += len(args.satisfying_sizes) * args.satisfying_num_dfas * est_time_per_satisfying_dfa
        
    if args.parallel:
        cpu_count = multiprocessing.cpu_count()
        max_workers = args.max_workers if args.max_workers is not None else cpu_count
        est_total_time /= max_workers
    
    print(f"⏱️ Estimated execution time: {format_time(est_total_time)} (rough estimate)")
    
    random.seed(args.seed)
    all_results = []
    
    # Track the number of accepting DFAs for each mode
    accepting_counts = {
        'non-satisfying': 0, 
        'satisfying': 0
    }
    total_counts = {
        'non-satisfying': 0, 
        'satisfying': 0
    }
    
    # Print CPU information
    cpu_count = multiprocessing.cpu_count()
    max_workers = args.max_workers if args.max_workers is not None else cpu_count
    print(f"🖥️ System has {cpu_count} CPU cores available.")
    print(f"🧵 Using up to {max_workers} worker processes for parallel execution.\n")
    
    experiment_start = time.time()
    
    # Process non-satisfying DFAs if requested
    if args.mode in ['non_satisfying', 'both']:
        mode_name = "non-satisfying"
        results = []
        
        mode_start = time.time()
        print(f"\n{'='*30}")
        print(f"🔄 PROCESSING {mode_name.upper()} DFAs:")
        print(f"{'='*30}")
        
        if args.parallel:
            # Process DFAs in parallel
            print(f"🚀 Using parallel processing with up to {max_workers} workers")
            
            results = process_dfas_in_parallel(
                args.non_satisfying_sizes, 
                args.non_satisfying_num_dfas, 
                False,  # satisfy_spec = False
                args.accept_prob,
                mode_name,
                max_workers
            )
            
            # Extract elapsed times for overall statistics
            overall_times = [row[3] for row in results]
            
        else:
            # Process DFAs sequentially (original method)
            overall_times = []
            total_dfas = len(args.non_satisfying_sizes) * args.non_satisfying_num_dfas
            pbar = tqdm(total=total_dfas, desc=f"Processing {mode_name} DFAs")
            
            for size in args.non_satisfying_sizes:
                size_times = []
                for index in range(args.non_satisfying_num_dfas):
                    size, result, elapsed = process_dfa(size, False, args.accept_prob)
                    results.append([size, index + 1, result, elapsed, mode_name])
                    overall_times.append(elapsed)
                    size_times.append(elapsed)
                    pbar.update(1)
                
                # Print progress after each size is complete
                if args.progress:
                    avg_time = sum(size_times) / len(size_times)
                    print(f"✓ Completed size {size}: Avg time = {avg_time:.6f}s")
            
            pbar.close()
        
        # Update statistics for accepting DFAs
        accepting_counts[mode_name] = sum(1 for row in results if row[2] == True)
        total_counts[mode_name] = len(results)
        
        # Calculate statistics for this mode
        stats = calculate_statistics(results)
        print(f"\n📊 Statistics for {mode_name} DFAs:")
        for size in args.non_satisfying_sizes:
            if size in stats:
                size_stats = stats[size]
                print(f"  Size {size}: Mean = {size_stats['mean']:.6f}s, Std Dev = {size_stats['std_dev']:.6f}s")
                if args.progress:
                    print(f"  → Min = {size_stats['min']:.6f}s, Max = {size_stats['max']:.6f}s")
        
        # Save results for this mode to separate file
        save_results_to_csv(results, f"results_non_satisfying.csv")
        
        # Mode timing information
        mode_elapsed = time.time() - mode_start
        print(f"\n⏱️ {mode_name} processing completed in {format_time(mode_elapsed)}")
        
        # Add results to combined list
        all_results.extend(results)
    
    # Process satisfying DFAs if requested
    if args.mode in ['satisfying', 'both']:
        mode_name = "satisfying"
        results = []
        
        mode_start = time.time()
        print(f"\n{'='*30}")
        print(f"🔄 PROCESSING {mode_name.upper()} DFAs:")
        print(f"{'='*30}")
        
        if args.parallel:
            # Process DFAs in parallel
            print(f"🚀 Using parallel processing with up to {max_workers} workers")
            
            results = process_dfas_in_parallel(
                args.satisfying_sizes, 
                args.satisfying_num_dfas, 
                True,  # satisfy_spec = True
                args.accept_prob,
                mode_name,
                max_workers
            )
            
            # Extract elapsed times for overall statistics
            overall_times = [row[3] for row in results]
            
        else:
            # Process DFAs sequentially (original method)
            overall_times = []
            total_dfas = len(args.satisfying_sizes) * args.satisfying_num_dfas
            pbar = tqdm(total=total_dfas, desc=f"Processing {mode_name} DFAs")
            
            for size in args.satisfying_sizes:
                size_times = []
                for index in range(args.satisfying_num_dfas):
                    size, result, elapsed = process_dfa(size, True, args.accept_prob)
                    results.append([size, index + 1, result, elapsed, mode_name])
                    overall_times.append(elapsed)
                    size_times.append(elapsed)
                    pbar.update(1)
                
                # Print progress after each size is complete
                if args.progress:
                    avg_time = sum(size_times) / len(size_times)
                    print(f"✓ Completed size {size}: Avg time = {avg_time:.6f}s")
            
            pbar.close()
        
        # Update statistics for accepting DFAs
        accepting_counts[mode_name] = sum(1 for row in results if row[2] == True)
        total_counts[mode_name] = len(results)
        
        # Calculate statistics for this mode
        stats = calculate_statistics(results)
        print(f"\n📊 Statistics for {mode_name} DFAs:")
        for size in args.satisfying_sizes:
            if size in stats:
                size_stats = stats[size]
                print(f"  Size {size}: Mean = {size_stats['mean']:.6f}s, Std Dev = {size_stats['std_dev']:.6f}s")
                if args.progress:
                    print(f"  → Min = {size_stats['min']:.6f}s, Max = {size_stats['max']:.6f}s")
        
        # Save results for this mode to separate file
        save_results_to_csv(results, f"results_satisfying.csv")
        
        # Mode timing information
        mode_elapsed = time.time() - mode_start
        print(f"\n⏱️ {mode_name} processing completed in {format_time(mode_elapsed)}")
        
        # Add results to combined list
        all_results.extend(results)
    
    # Save combined results if both modes were run
    if args.mode == 'both':
        save_results_to_csv(all_results, args.outfile)
        print(f"📊 Combined results saved to {args.outfile}")
    
    # Graph results
    graph_results(all_results, outfile=args.graph_out, log_scale=args.log_scale, stats_display=args.stats_display)
    print(f"📈 Graph saved to {args.graph_out}")
    
    # Total accepting DFAs across all modes
    total_accepting = sum(accepting_counts.values())
    total_dfas_processed = len(all_results)
    
    # Print final timing information
    total_elapsed = time.time() - experiment_start
    print("\n" + "="*60)
    print(f"✅ EXPERIMENT COMPLETED SUCCESSFULLY!")
    print(f"⏱️ Total execution time: {format_time(total_elapsed)}")
    print(f"📊 Processed {total_dfas_processed} DFAs in total")
    
    # Print accepting DFA statistics
    print("\n📋 DFA ACCEPTANCE STATISTICS:")
    print(f"{'='*30}")
    if args.mode in ['non_satisfying', 'both'] and total_counts['non-satisfying'] > 0:
        non_sat_percent = (accepting_counts['non-satisfying'] / total_counts['non-satisfying']) * 100
        print(f"⚠️ Non-Satisfying DFAs: {accepting_counts['non-satisfying']} / {total_counts['non-satisfying']} ({non_sat_percent:.1f}%) were accepting")
        if accepting_counts['non-satisfying'] > 0:
            print(f"   Note: Non-satisfying DFAs should ideally not satisfy the spec, but some randomly do")
    
    if args.mode in ['satisfying', 'both'] and total_counts['satisfying'] > 0:
        sat_percent = (accepting_counts['satisfying'] / total_counts['satisfying']) * 100
        print(f"✓ Satisfying DFAs: {accepting_counts['satisfying']} / {total_counts['satisfying']} ({sat_percent:.1f}%) were accepting")
        if accepting_counts['satisfying'] < total_counts['satisfying']:
            print(f"   Note: All satisfying DFAs should satisfy the spec; this indicates a possible issue")
    
    if args.mode == 'both':
        total_percent = (total_accepting / total_dfas_processed) * 100
        print(f"📊 Total: {total_accepting} / {total_dfas_processed} ({total_percent:.1f}%) DFAs were accepting")
    
    print("="*60 + "\n")

if __name__ == "__main__":
    main()