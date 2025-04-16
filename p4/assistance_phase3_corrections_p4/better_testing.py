# Improving Statistical Significance in Your Experiments

To obtain statistically significant results for this NFA finite acceptance algorithm, I recommend the following modifications to your experimental setup:

## 1. Increase Number of Runs

```python
def run_tests(num_trials=30):  # Run each size multiple times
    test_sizes = [1000, 5000, 10000, 25000, 50000, 100000, 250000, 500000]
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
    
    # Process and display statistics
    analyze_results(all_results, test_sizes)
    return all_results
```

## 2. Add Statistical Analysis

```python
def analyze_results(all_results, test_sizes):
    import numpy as np
    
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Average time vs size with error bars
    plt.subplot(2, 1, 1)
    means = []
    stdevs = []
    
    for size_results in all_results:
        times = [r[2] for r in size_results]
        means.append(np.mean(times))
        stdevs.append(np.std(times))
    
    plt.errorbar(test_sizes, means, yerr=stdevs, fmt='o-', capsize=5)
    plt.xlabel('Number of states')
    plt.ylabel('Average time (seconds)')
    plt.title('NFA Finite Acceptance Check - Average Time with Standard Deviation')
    plt.grid(True)
    
    # Plot 2: Log-log plot to identify complexity class
    plt.subplot(2, 1, 2)
    plt.loglog(test_sizes, means, 'o-')
    plt.xlabel('Number of states (log scale)')
    plt.ylabel('Time (log scale)')
    plt.title('Log-Log Plot to Identify Computational Complexity')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("==================")
    for i, size in enumerate(test_sizes):
        times = [r[2] for r in all_results[i]]
        finite_count = sum(1 for r in all_results[i] if r[1])
        print(f"Size {size}:")
        print(f"  Mean time: {np.mean(times):.6f} seconds")
        print(f"  Median time: {np.median(times):.6f} seconds")
        print(f"  Std dev: {np.std(times):.6f} seconds")
        print(f"  95% CI: ({np.mean(times) - 1.96*np.std(times)/np.sqrt(len(times)):.6f}, "
              f"{np.mean(times) + 1.96*np.std(times)/np.sqrt(len(times)):.6f})")
        print(f"  Finite results: {finite_count}/{len(all_results[i])} ({finite_count/len(all_results[i])*100:.1f}%)")
```

## 3. Key Recommendations

1. **More samples**: Run 20-30 trials per size for reliable statistics
2. **More data points**: Test 8-10 different NFA sizes, evenly distributed 
3. **Include smaller sizes**: Start from 1,000 states to better visualize growth
4. **Use different random seeds**: Ensure diversity in the generated NFAs
5. **Statistical metrics**: Calculate means, medians, standard deviations, and confidence intervals
6. **Log-log plots**: Help identify the actual computational complexity class

This approach will give you much more statistically rigorous results that better characterize the algorithm's performance across different input sizes.