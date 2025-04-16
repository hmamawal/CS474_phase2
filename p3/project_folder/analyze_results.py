#!/usr/bin/env python3
"""
Script to analyze time complexity of existing P3 DFA results.
"""
import os
import sys
from visualization import load_results_from_csv, analyze_and_save_plots

def main():
    """
    Main function to process CSV files and generate complexity analysis.
    """
    # Check command line arguments for CSV file path
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        # Default to the satisfying results path
        csv_path = "../results/satisfying/results_satisfying.csv"
    
    # Check if the file exists
    if not os.path.exists(csv_path):
        print(f"❌ Error: CSV file not found: {csv_path}")
        return
    
    # Determine output directory - same as input file
    output_dir = os.path.dirname(csv_path)
    
    # Determine if this is satisfying or non-satisfying data
    is_satisfying = "satisfying" in csv_path.lower()
    mode_text = "Satisfying" if is_satisfying else "Non-Satisfying"
    
    print(f"\n{'='*60}")
    print(f"📊 DFA {mode_text.upper()} TIME COMPLEXITY ANALYSIS")
    print(f"{'='*60}")
    print(f"📂 CSV File: {csv_path}")
    print(f"📁 Output Directory: {output_dir}")
    print(f"{'='*60}\n")
    
    # Load the CSV data
    results = load_results_from_csv(csv_path)
    
    if not results:
        print("❌ No results loaded. Exiting.")
        return
    
    # Define output files
    base_name = os.path.join(output_dir, f"dfa_time_complexity_{mode_text.lower()}")
    outfile = f"{base_name}.png"
    
    # Analyze and plot results
    print(f"🔍 Analyzing time complexity for {mode_text} DFAs...")
    title = f"DFA {mode_text} Time Complexity Analysis"
    
    complexity_results = analyze_and_save_plots(
        results=results,
        title=title,
        outfile=outfile,
        log_scale=True,
        stats_display="text"
    )
    
    print(f"\n✅ Analysis complete! Files saved to {output_dir}")
    print(f"📊 Standard Plot: {outfile}")
    print(f"📈 Log-Log Plot: {base_name}_log_log.png")
    print(f"📝 Statistics: {base_name}_statistics.txt")

if __name__ == "__main__":
    main()