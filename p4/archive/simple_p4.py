from collections import deque
from copy import deepcopy

def dfa_to_nfa(dfa: dict) -> dict:
    """
    Convert a DFA (with keys: 'alphabet', 'states', 'initial_state', 
    'accepting_states', and 'transitions') into an equivalent NFA.
    
    In the resulting NFA:
      - 'initial_states' is a singleton set.
      - Each transition (state, symbol) -> next_state becomes 
        (state, symbol) -> {next_state}.
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
    Both automata are assumed to be over the same alphabet.
    
    The resulting NFA has:
      - States: pairs (s1, s2) where s1 is from nfa1 and s2 from nfa2.
      - Initial states: all pairs (i1, i2) with i1 in nfa1['initial_states'] 
        and i2 in nfa2['initial_states'].
      - Accepting states: pairs where both components are accepting.
      - Transitions: For each symbol a, if (s1, a) -> S1 and (s2, a) -> S2, 
        then ((s1, s2), a) -> { (t1, t2) for each t1 in S1 and t2 in S2 }.
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
    The graph is a dictionary mapping each state to the set of states 
    reachable in one transition (over any symbol).
    """
    graph = { state: set() for state in nfa['states'] }
    for (state, symbol), next_states in nfa['transitions'].items():
        graph[state].update(next_states)
    return graph

def compute_coreachable_states(nfa: dict, graph: dict) -> set:
    """
    Compute the set of states from which an accepting state is reachable.
    This is done by first building the reverse graph and then doing a BFS
    starting from the accepting states.
    """
    # Build reverse graph.
    reverse_graph = { state: set() for state in nfa['states'] }
    for u in graph:
        for v in graph[u]:
            reverse_graph[v].add(u)
    # BFS from accepting states.
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
    Check whether the subgraph induced by the set 'nodes' has a cycle.
    Uses DFS with a recursion stack to detect cycles.
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
    Given:
      - nfa: an NFA (in pysimpleautomata dictionary format) for the language L(N).
      - spec_dfa: a DFA (in dictionary format) that accepts exactly the strings
                  that do NOT end with the forbidden suffixes "AC0" or "0CA".
    
    This function checks whether L(N) ∩ L(spec_dfa) is finite.
    (That is, whether N accepts only finitely many strings that do NOT have the forbidden suffix.)
    
    Returns True if the intersection language is finite, False if it is infinite.
    """
    # Convert the specification DFA to an NFA.
    spec_nfa = dfa_to_nfa(spec_dfa)
    # Compute the intersection NFA: strings accepted by N and that do NOT have the forbidden suffix.
    inter_nfa = nfa_intersection(nfa, spec_nfa)
    
    # Build the transition graph from the intersection NFA.
    graph = build_graph(inter_nfa)
    
    # Compute the set of states that can reach an accepting state (co-reachable states).
    co_reachable = compute_coreachable_states(inter_nfa, graph)
    
    # Check for cycles in the subgraph induced by co-reachable states.
    cycle_exists = has_cycle_in_subgraph(graph, co_reachable)
    
    # If a cycle exists, the intersection language is infinite.
    return not cycle_exists

# ---------------------------
# Example usage:
# ---------------------------
if __name__ == '__main__':
    # Assume SPEC_DFA is defined exactly as given:
    SPEC_DFA = {
        'alphabet': set("AC0"),
        'states': {'q0', 'q_A', 'q_AC', 'q_0', 'q_0C', 'q_AC0', 'q_0CA'},
        'initial_state': 'q0',
        'accepting_states': {'q0', 'q_A', 'q_AC', 'q_0', 'q_0C'},
        'transitions': {
            # From the empty suffix state
            ('q0', 'A'): 'q_A',    # "" + A = "A"
            ('q0', 'C'): 'q0',     # "" + C = "C" (no match; remain in q0)
            ('q0', '0'): 'q_0',    # "" + 0 = "0"
    
            # From state q_A (matching "A")
            ('q_A', 'A'): 'q_A',   # "A" + A = "AA" → longest suffix "A"
            ('q_A', 'C'): 'q_AC',  # "A" + C = "AC"
            ('q_A', '0'): 'q_0',   # "A" + 0 = "A0" → longest suffix "0"
    
            # From state q_AC (matching "AC")
            ('q_AC', 'A'): 'q_A',   # "AC" + A = "ACA" → suffix "A"
            ('q_AC', 'C'): 'q0',    # "AC" + C = "ACC" → suffix "C" (no match)
            ('q_AC', '0'): 'q_AC0', # "AC" + 0 = "AC0" (forbidden)
    
            # From state q_0 (matching "0")
            ('q_0', 'A'): 'q_A',    # "0" + A = "0A" → suffix "A"
            ('q_0', 'C'): 'q_0C',   # "0" + C = "0C"
            ('q_0', '0'): 'q_0',    # "0" + 0 = "00" → suffix "0"
    
            # From state q_0C (matching "0C")
            ('q_0C', 'A'): 'q_0CA', # "0C" + A = "0CA" (forbidden)
            ('q_0C', 'C'): 'q0',    # "0C" + C = "0CC" → suffix "C" (no match)
            ('q_0C', '0'): 'q_0',   # "0C" + 0 = "0C0" → suffix "0"
    
            # From state q_AC0 (matching "AC0", forbidden)
            ('q_AC0', 'A'): 'q_A',  # "AC0" + A = "AC0A" → suffix "A"
            ('q_AC0', 'C'): 'q_0C', # "AC0" + C = "AC0C" → suffix "0C" (prefix of "0CA")
            ('q_AC0', '0'): 'q_0',  # "AC0" + 0 = "AC00" → suffix "0"
    
            # From state q_0CA (matching "0CA", forbidden)
            ('q_0CA', 'A'): 'q_A',  # "0CA" + A = "0CAA" → suffix "A"
            ('q_0CA', 'C'): 'q_AC', # "0CA" + C = "0CAC" → suffix "AC" (prefix of "AC0")
            ('q_0CA', '0'): 'q_0',  # "0CA" + 0 = "0CA0" → suffix "0"
        }
    }
    
    # For demonstration, suppose we have an NFA (this is just an example;
    # in practice, the NFA comes from your problem instance).
    # This example NFA accepts all strings over {A, C, 0}.
    example_nfa = {
        'alphabet': set("AC0"),
        'states': {'s0'},
        'initial_states': {'s0'},
        'accepting_states': {'s0'},
        'transitions': {
            ('s0', 'A'): {'s0'},
            ('s0', 'C'): {'s0'},
            ('s0', '0'): {'s0'},
        }
    }
    
    # Check whether the example NFA accepts only finitely many strings
    # that do not end with "AC0" or "0CA".
    finite = check_finite_acceptance(example_nfa, SPEC_DFA)
    if finite:
        print("The NFA accepts only finitely many strings without the forbidden suffix.")
    else:
        print("The NFA accepts infinitely many strings without the forbidden suffix.")
