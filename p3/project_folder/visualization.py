import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

def graph_results(results, title="DFA Computation Time vs DFA Size", outfile="graph.png", log_scale=False):
    """
    Graphs the computation time statistics for each DFA size using matplotlib.
    Includes options for log-log scaling.
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
    
    # Add statistics as text
    stats_text = "Processing Statistics:\n"
    for i, size in enumerate(unique_sizes):
        if (size, unique_modes[0]) in data:  # Just use the first mode for stats
            avg = np.mean(data[(size, unique_modes[0])])
            stats_text += f"{size} states: {avg:.6f}s avg\n"
    
    plt.figtext(0.02, 0.02, stats_text, fontsize=9)
    
    # Save and close
    print(f"💾 Saving graph to {outfile}...")
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    print(f"✅ Graph saved successfully!")
    
    return outfile