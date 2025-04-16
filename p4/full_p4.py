import random
import time
import os
import csv
from collections import deque
from copy import deepcopy
import matplotlib.pyplot as plt  # added for graphing
import numpy as np

random.seed(42)  # added for reproducibility

# --------------------------
# Previously defined functions
# --------------------------

def dfa_to_nfa(dfa: dict) -> dict:
    """
    Convert a DFA into an equivalent NFA.
    """
    nfa = {
        'alphabet': dfa['alphabet'].copy(),
        'states': dfa['states'].copy(),
        'initial_states': { dfa['initial_state'] },
        'accepting_states': dfa['accepting_states'].copy(),
        'transitions': dict()
    }
    for (state, symbol), next_state in dfa['transitions'].items():
        nfa['transitions'][(state, symbol)] = { next_state }
    return nfa

def nfa_intersection(nfa1: dict, nfa2: dict) -> dict:
    """
    Compute the intersection of two NFAs using a product construction.
    """
    intersection = {
        'alphabet': nfa1['alphabet'].intersection(nfa2['alphabet']),
        'states': set(),
        'initial_states': set(),
        'accepting_states': set(),
        'transitions': dict()
    }
    # Initialize initial states (all pairs)
    for init1 in nfa1['initial_states']:
        for init2 in nfa2['initial_states']:
            intersection['initial_states'].add((init1, init2))
    intersection['states'].update(intersection['initial_states'])
    
    # Use a worklist to build reachable states and transitions.
    boundary = set(intersection['initial_states'])
    while boundary:
        (s1, s2) = boundary.pop()
        if s1 in nfa1['accepting_states'] and s2 in nfa2['accepting_states']:
            intersection['accepting_states'].add((s1, s2))
        for a in intersection['alphabet']:
            if (s1, a) in nfa1['transitions'] and (s2, a) in nfa2['transitions']:
                for next1 in nfa1['transitions'][(s1, a)]:
                    for next2 in nfa2['transitions'][(s2, a)]:
                        next_state = (next1, next2)
                        if next_state not in intersection['states']:
                            intersection['states'].add(next_state)
                            boundary.add(next_state)
                        intersection['transitions'].setdefault(((s1, s2), a), set()).add(next_state)
    return intersection

def build_graph(nfa: dict) -> dict:
    """
    Build a directed graph from the transitions of an NFA.
    """
    graph = { state: set() for state in nfa['states'] }
    for (state, symbol), next_states in nfa['transitions'].items():
        graph[state].update(next_states)
    return graph

def compute_coreachable_states(nfa: dict, graph: dict) -> set:
    """
    Compute the set of states from which an accepting state is reachable.
    """
    reverse_graph = { state: set() for state in nfa['states'] }
    for u in graph:
        for v in graph[u]:
            reverse_graph[v].add(u)
    co_reachable = set()
    queue = deque()
    for state in nfa['accepting_states']:
        if state in nfa['states']:
            co_reachable.add(state)
            queue.append(state)
    while queue:
        v = queue.popleft()
        for u in reverse_graph[v]:
            if u not in co_reachable:
                co_reachable.add(u)
                queue.append(u)
    return co_reachable

def has_cycle_in_subgraph(graph: dict, nodes: set) -> bool:
    """
    Check whether the subgraph induced by 'nodes' has a cycle using iterative DFS.
    """
    visited = set()
    
    for start in nodes:
        if start in visited:
            continue
            
        # Stack stores (node, iterator_over_neighbors)
        stack = [(start, iter([n for n in graph[start] if n in nodes]))]
        # Track nodes in current exploration path
        rec_stack = {start}
        visited.add(start)
        
        while stack:
            node, neighbors_iter = stack[-1]
            
            try:
                # Get next neighbor
                neighbor = next(neighbors_iter)
                
                if neighbor not in visited:
                    # Visit this neighbor next
                    visited.add(neighbor)
                    rec_stack.add(neighbor)
                    stack.append((neighbor, iter([n for n in graph[neighbor] if n in nodes])))
                elif neighbor in rec_stack:
                    # Cycle detected
                    return True
                    
            except StopIteration:
                # No more neighbors, backtrack
                rec_stack.remove(node)
                stack.pop()
    
    return False

def check_finite_acceptance(nfa: dict, spec_dfa: dict) -> bool:
    """
    Check whether the intersection L(N) ∩ L(spec_dfa) is finite.
    Returns True if finite, False if infinite.
    """
    # Convert the specification DFA into an NFA.
    spec_nfa = dfa_to_nfa(spec_dfa)
    # Compute the intersection.
    inter_nfa = nfa_intersection(nfa, spec_nfa)
    # Build the transition graph.
    graph = build_graph(inter_nfa)
    # Compute states that can reach an accepting state.
    co_reachable = compute_coreachable_states(inter_nfa, graph)
    # Check for cycles in the co-reachable subgraph.
    cycle_exists = has_cycle_in_subgraph(graph, co_reachable)
    # If a cycle exists, then the language is infinite.
    return not cycle_exists

# --------------------------
# Specification DFA (constant-size)
# --------------------------
SPEC_DFA = {
    'alphabet': set("AC0"),
    'states': {'q0', 'q_A', 'q_AC', 'q_0', 'q_0C', 'q_AC0', 'q_0CA'},
    'initial_state': 'q0',
    'accepting_states': {'q0', 'q_A', 'q_AC', 'q_0', 'q_0C'},
    'transitions': {
        # From the empty suffix state
        ('q0', 'A'): 'q_A',    # "" + A = "A"
        ('q0', 'C'): 'q0',     # "" + C = "C"
        ('q0', '0'): 'q_0',    # "" + 0 = "0"
        # From state q_A (matching "A")
        ('q_A', 'A'): 'q_A',
        ('q_A', 'C'): 'q_AC',
        ('q_A', '0'): 'q_0',
        # From state q_AC (matching "AC")
        ('q_AC', 'A'): 'q_A',
        ('q_AC', 'C'): 'q0',
        ('q_AC', '0'): 'q_AC0',
        # From state q_0 (matching "0")
        ('q_0', 'A'): 'q_A',
        ('q_0', 'C'): 'q_0C',
        ('q_0', '0'): 'q_0',
        # From state q_0C (matching "0C")
        ('q_0C', 'A'): 'q_0CA',
        ('q_0C', 'C'): 'q0',
        ('q_0C', '0'): 'q_0',
        # From state q_AC0 (forbidden "AC0")
        ('q_AC0', 'A'): 'q_A',
        ('q_AC0', 'C'): 'q_0C',
        ('q_AC0', '0'): 'q_0',
        # From state q_0CA (forbidden "0CA")
        ('q_0CA', 'A'): 'q_A',
        ('q_0CA', 'C'): 'q_AC',
        ('q_0CA', '0'): 'q_0',
    }
}

# --------------------------
# Helper function to generate random NFAs
# --------------------------
def generate_random_nfa(num_states: int, alphabet: str) -> dict:
    """
    Generates a random NFA with 'num_states' states over the given alphabet.
    - States are named "q0", "q1", ..., "q{num_states-1}".
    - A single random state is chosen as the initial state.
    - A random nonempty subset of states is chosen as accepting states.
    - For each state and symbol, with a given probability, add one or more transitions.
    """
    nfa = {
        'alphabet': set(alphabet),
        'states': { f"q{i}" for i in range(num_states) },
        'initial_states': set(),
        'accepting_states': set(),
        'transitions': dict()
    }
    
    all_states = list(nfa['states'])
    # Choose one random initial state.
    nfa['initial_states'].add(random.choice(all_states))
    # Choose a random nonempty set of accepting states.
    num_accepting = random.randint(1, num_states)
    nfa['accepting_states'] = set(random.sample(all_states, num_accepting))
    
    # For each state and symbol, with probability p, add transitions.
    transition_probability = 0.3
    for state in nfa['states']:
        for symbol in nfa['alphabet']:
            if random.random() < transition_probability:
                # Choose a random number of transitions (1 to 3)
                num_transitions = random.randint(1, 3)
                # Use random.choices to allow duplicates (which will become a set)
                next_states = set(random.choices(all_states, k=num_transitions))
                nfa['transitions'][(state, symbol)] = next_states
    return nfa

# --------------------------
# Helper functions for file management and results output
# --------------------------
def get_next_run_number():
    """Determine the next run number by checking existing directories."""
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        return 1
    
    existing_runs = [d for d in os.listdir(results_dir) 
                    if os.path.isdir(os.path.join(results_dir, d)) and d.startswith('run')]
    
    if not existing_runs:
        return 1
    
    run_numbers = [int(run.replace('run', '')) for run in existing_runs 
                  if run.replace('run', '').isdigit()]
    
    return max(run_numbers) + 1 if run_numbers else 1

def save_results_to_csv(all_results, test_sizes, run_number):
    """Save results to CSV files."""
    results_dir = os.path.join(os.path.dirname(__file__), 'results', f'run{run_number}')
    os.makedirs(results_dir, exist_ok=True)
    
    # Save individual trials data
    individual_file = os.path.join(results_dir, f'individual_times.csv')
    with open(individual_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['size', 'trial', 'finite', 'time'])
        for i, size_results in enumerate(all_results):
            for j, (size, finite, elapsed) in enumerate(size_results):
                writer.writerow([size, j+1, int(finite), elapsed])
    
    # Save summary statistics
    summary_file = os.path.join(results_dir, f'nfa_performance_data_run{run_number}.csv')
    with open(summary_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['size', 'mean_time', 'median_time', 'std_dev', 'ci_lower', 'ci_upper', 'finite_percent'])
        
        for i, size in enumerate(test_sizes):
            times = [r[2] for r in all_results[i]]
            finite_count = sum(1 for r in all_results[i] if r[1])
            mean_time = np.mean(times)
            std_dev = np.std(times)
            ci_lower = mean_time - 1.96 * std_dev / np.sqrt(len(times))
            ci_upper = mean_time + 1.96 * std_dev / np.sqrt(len(times))
            finite_percent = finite_count / len(all_results[i]) * 100
            
            writer.writerow([
                size,
                f"{mean_time:.6f}",
                f"{np.median(times):.6f}",
                f"{std_dev:.6f}",
                f"{ci_lower:.6f}",
                f"{ci_upper:.6f}",
                f"{finite_percent:.1f}"
            ])
    
    # Save summary statistics as text
    summary_text_file = os.path.join(results_dir, f'summary_statistics.txt')
    with open(summary_text_file, 'w') as f:
        f.write("Summary Statistics:\n")
        f.write("==================\n")
        
        for i, size in enumerate(test_sizes):
            times = [r[2] for r in all_results[i]]
            finite_count = sum(1 for r in all_results[i] if r[1])
            mean_time = np.mean(times)
            std_dev = np.std(times)
            ci_lower = mean_time - 1.96 * std_dev / np.sqrt(len(times))
            ci_upper = mean_time + 1.96 * std_dev / np.sqrt(len(times))
            
            f.write(f"Size {size}:\n")
            f.write(f"  Mean time: {mean_time:.6f} seconds\n")
            f.write(f"  Median time: {np.median(times):.6f} seconds\n")
            f.write(f"  Std dev: {std_dev:.6f} seconds\n")
            f.write(f"  95% CI: ({ci_lower:.6f}, {ci_upper:.6f})\n")
            f.write(f"  Finite results: {finite_count}/{len(all_results[i])} ({finite_count/len(all_results[i])*100:.1f}%)\n")
    
    return summary_file, results_dir

def analyze_and_save_plots(all_results, test_sizes, run_number):
    """Analyze results, generate plots, and save them."""
    results_dir = os.path.join(os.path.dirname(__file__), 'results', f'run{run_number}')
    os.makedirs(results_dir, exist_ok=True)
    
    means = []
    stdevs = []
    
    for size_results in all_results:
        times = [r[2] for r in size_results]
        means.append(np.mean(times))
        stdevs.append(np.std(times))
    
    # Plot 1: Average time vs size with error bars
    plt.figure(figsize=(12, 8))
    plt.errorbar(test_sizes, means, yerr=stdevs, fmt='o-', capsize=5)
    plt.xlabel('Number of states')
    plt.ylabel('Average time (seconds)')
    plt.title('NFA Finite Acceptance Check - Average Time with Standard Deviation')
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'Figure_1.png'))
    
    # Plot 2: Log-log plot to identify complexity class
    plt.figure(figsize=(12, 8))
    plt.loglog(test_sizes, means, 'o-', label='Measured times')
    
    # Calculate complexity using linear regression on log-log data
    log_sizes = np.log(test_sizes)
    log_times = np.log(means)
    slope, intercept = np.polyfit(log_sizes, log_times, 1)
    r_squared = np.corrcoef(log_sizes, log_times)[0, 1]**2
    
    # Generate fitted line for visualization
    fit_line = np.exp(intercept) * np.array(test_sizes)**slope
    plt.loglog(test_sizes, fit_line, 'r--', 
               label=f'Fitted line: O(n^{slope:.2f}), R² = {r_squared:.4f}')
    
    plt.xlabel('Number of states (log scale)')
    plt.ylabel('Time (log scale)')
    plt.title('Log-Log Plot to Identify Computational Complexity')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'Figure_2_big_o.png'))
    
    # Print and return complexity analysis
    if 0.8 <= slope <= 1.2:
        complexity = "approximately linear - O(n)"
    elif 1.8 <= slope <= 2.2:
        complexity = "approximately quadratic - O(n²)"
    elif 2.8 <= slope <= 3.2:
        complexity = "approximately cubic - O(n³)"
    else:
        complexity = f"O(n^{slope:.2f})"
    
    complexity_info = {
        "slope": slope,
        "r_squared": r_squared,
        "complexity": complexity
    }
    
    return complexity_info

# --------------------------
# Testing and Timing
# --------------------------
def run_tests(num_trials=30):  # Run each size multiple times
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
    
    # Save results to CSV
    summary_file, results_dir = save_results_to_csv(all_results, test_sizes, run_number)
    
    # Process, display statistics and save plots
    complexity_info = analyze_and_save_plots(all_results, test_sizes, run_number)
    
    # Print summary and complexity analysis to terminal
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
    
    # Print complexity analysis
    print("\nComplexity Analysis:")
    print("==================")
    print(f"Measured complexity: O(n^{complexity_info['slope']:.4f})")
    print(f"R-squared value: {complexity_info['r_squared']:.4f}")
    print(f"Algorithm appears to be {complexity_info['complexity']}")
    print(f"Coefficient of determination (R²): {complexity_info['r_squared']:.4f}")
    print(f"The R² value indicates how well the power law model fits the data (closer to 1.0 is better)")
    
    print(f"\nResults saved to: {results_dir}")
    
    return all_results

def run_unit_tests():
    # Unit test for dfa_to_nfa
    test_dfa = {
        'alphabet': set("ab"),
        'states': {'q0', 'q1'},
        'initial_state': 'q0',
        'accepting_states': {'q1'},
        'transitions': { ('q0','a'): 'q1', ('q1','b'): 'q0' }
    }
    nfa = dfa_to_nfa(test_dfa)
    assert nfa['alphabet'] == test_dfa['alphabet']
    assert nfa['states'] == test_dfa['states']
    assert nfa['initial_states'] == {'q0'}
    assert nfa['accepting_states'] == test_dfa['accepting_states']
    # Unit test for nfa_intersection with simple NFAs
    nfa1 = {
        'alphabet': set("ab"),
        'states': {'q0','q1'},
        'initial_states': {'q0'},
        'accepting_states': {'q1'},
        'transitions': { ('q0','a'): {'q1'} }
    }
    nfa2 = {
        'alphabet': set("ab"),
        'states': {'p0','p1'},
        'initial_states': {'p0'},
        'accepting_states': {'p1'},
        'transitions': { ('p0','a'): {'p1'} }
    }
    inter = nfa_intersection(nfa1, nfa2)
    assert (('q0','p0') in inter['initial_states'])
    assert inter['transitions'].get((('q0','p0'),'a')) == {('q1','p1')}
    # Unit test for build_graph and compute_coreachable_states on a two-state cycle
    simple_nfa = {
        'alphabet': set("a"),
        'states': {'s0','s1'},
        'initial_states': {'s0'},
        'accepting_states': {'s1'},
        'transitions': { ('s0','a'): {'s1'}, ('s1','a'): {'s0'} }
    }
    graph = build_graph(simple_nfa)
    co_reachable = compute_coreachable_states(simple_nfa, graph)
    assert 's0' in co_reachable and 's1' in co_reachable
    # Unit test for has_cycle_in_subgraph
    acyclic_graph = { 's0': {'s1'}, 's1': set() }
    cyclic_graph = { 's0': {'s1'}, 's1': {'s0'} }
    assert not has_cycle_in_subgraph(acyclic_graph, {'s0','s1'})
    assert has_cycle_in_subgraph(cyclic_graph, {'s0','s1'})
    # Unit test for check_finite_acceptance on a simple finite NFA (no transitions => finite language)
    finite_nfa = {
        'alphabet': set("a"),
        'states': {'s0'},
        'initial_states': {'s0'},
        'accepting_states': {'s0'},
        'transitions': {}
    }
    finite_result = check_finite_acceptance(finite_nfa, test_dfa)
    assert finite_result == True
    # Unit test for generate_random_nfa (basic structure test)
    rand_nfa = generate_random_nfa(10, "ab")
    assert 'alphabet' in rand_nfa and 'states' in rand_nfa and 'transitions' in rand_nfa
    print("All unit tests passed.")

if __name__ == '__main__':
    run_unit_tests()
    run_tests()
