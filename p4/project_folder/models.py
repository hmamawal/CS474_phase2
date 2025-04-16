"""
Models module for NFA-related data structures and model conversions.
"""

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

# Define the specification DFA
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