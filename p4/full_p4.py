import random
import time
from collections import deque
from copy import deepcopy
import matplotlib.pyplot as plt  # added for graphing

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
    Check whether the subgraph induced by 'nodes' has a cycle using DFS.
    """
    visited = set()
    rec_stack = set()
    
    def dfs(v):
        visited.add(v)
        rec_stack.add(v)
        for neighbor in graph[v]:
            if neighbor not in nodes:
                continue
            if neighbor not in visited:
                if dfs(neighbor):
                    return True
            elif neighbor in rec_stack:
                return True
        rec_stack.remove(v)
        return False
    
    for v in nodes:
        if v not in visited:
            if dfs(v):
                return True
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
# Testing and Timing
# --------------------------
def run_tests():
    # Define various sizes (each at least a few hundred states)
    test_sizes = [200, 300, 400, 500, 600]
    results = []
    for size in test_sizes:
        # Generate a random NFA with the given size.
        nfa = generate_random_nfa(size, "AC0")
        start_time = time.perf_counter()
        # Run the finite acceptance check on the generated NFA.
        finite = check_finite_acceptance(nfa, SPEC_DFA)
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        results.append((size, finite, elapsed))
        print(f"NFA with {size} states: finite acceptance = {finite}, time taken = {elapsed:.6f} seconds")
    # Graph the results
    sizes = [r[0] for r in results]
    times = [r[2] for r in results]
    colors = ['green' if r[1] else 'red' for r in results]
    plt.figure()
    plt.scatter(sizes, times, c=colors)
    plt.xlabel('Number of states')
    plt.ylabel('Time taken (seconds)')
    plt.title('NFA Finite Acceptance Check Timing')
    plt.show()
    return results

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
