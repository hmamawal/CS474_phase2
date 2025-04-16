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
    print(f"📊 Generating visualization for {len(results)} data points...")
    
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
    
    plt.figure(figsize=(12, 8))
    
    # Color mapping for different modes
    colors = ['b', 'r', 'g', 'c', 'm', 'y']
    
    # Process each mode separately
    for i, mode in enumerate(unique_modes):
        print(f"Plotting data for mode: {mode}")
        sizes = []
        avg_times = []
        std_devs = []
        min_times = []
        max_times = []
        
        for size in unique_sizes:
            if (size, mode) in data:
                times = data[(size, mode)]
                sizes.append(size)
                avg_times.append(np.mean(times))
                std_devs.append(np.std(times))
                min_times.append(np.min(times))
                max_times.append(np.max(times))
        
        color = colors[i % len(colors)]
        label = f"{mode} DFAs"
        
        # Plot average times with error bars showing min and max
        plt.errorbar(
            sizes, avg_times,
            yerr=[
                np.array(avg_times) - np.array(min_times),  
                np.array(max_times) - np.array(avg_times)
            ],
            fmt=f'o-', color=color, capsize=5, label=label, alpha=0.7
        )
    
    # Set scales
    if log_scale:
        print("Using logarithmic scale for both axes")
        plt.xscale('log')
        plt.yscale('log')
        plt.title(f"{title} (Log-Log Scale)")
    else:
        plt.title(title)
    
    plt.xlabel("DFA Size (number of states)")
    plt.ylabel("Computation Time (seconds)")
    plt.grid(True, which="both", ls="--", alpha=0.3)
    
    # Add legend only if multiple modes
    if len(unique_modes) > 1:
        plt.legend(loc='best')
    
    # Generate statistics text
    stats_text = "Processing Statistics:\n"
    detailed_stats = {}
    
    for mode in unique_modes:
        detailed_stats[mode] = {}
        for size in unique_sizes:
            if (size, mode) in data:
                times = data[(size, mode)]
                avg = np.mean(times)
                std_dev = np.std(times)
                min_time = np.min(times)
                max_time = np.max(times)
                median = np.median(times)
                count = len(times)
                
                if mode == unique_modes[0]:  # Only add basic stats for first mode to the graph
                    stats_text += f"{size} states: {avg:.6f}s avg\n"
                
                detailed_stats[mode][size] = {
                    'avg': avg,
                    'std_dev': std_dev,
                    'min': min_time,
                    'max': max_time,
                    'median': median,
                    'count': count
                }
    
    # Add statistics to graph or save to file based on preference
    if stats_display.lower() == "display":
        plt.figtext(0.02, 0.02, stats_text, fontsize=9)
    else:  # stats_display == "text"
        # Generate stats file path from the graph path
        stats_file = os.path.splitext(outfile)[0] + "_statistics.txt"
        with open(stats_file, 'w') as f:
            f.write(f"PROCESSING STATISTICS FOR {title}\n")
            f.write("=" * 50 + "\n\n")
            
            for mode in unique_modes:
                f.write(f"Mode: {mode}\n")
                f.write("-" * 30 + "\n")
                f.write(f"{'Size':<10} {'Avg Time (s)':<15} {'Std Dev':<15} {'Min':<15} {'Max':<15} {'Median':<15} {'Count':<10}\n")
                
                for size in unique_sizes:
                    if (size, mode) in data:
                        stats = detailed_stats[mode][size]
                        f.write(f"{size:<10} {stats['avg']:<15.6f} {stats['std_dev']:<15.6f} "
                                f"{stats['min']:<15.6f} {stats['max']:<15.6f} "
                                f"{stats['median']:<15.6f} {stats['count']:<10}\n")
                
                f.write("\n")
            
        print(f"📝 Detailed statistics saved to {stats_file}")
    
    # Save and close
    print(f"💾 Saving graph to {outfile}...")
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    print(f"✅ Graph saved successfully!")
    
    return outfile