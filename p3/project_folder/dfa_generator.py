import random
from specs.suffix_spec import SPEC_DFA
from dfa_algorithms import check_suffix_inclusion

def generate_random_dfa(num_states, accept_prob=0.2):
    """
    Generates a completely random DFA over the alphabet {'0', 'A', 'C'}.
    Each state is marked as accepting with probability accept_prob.
    Transitions are chosen uniformly at random.
    """
    alphabet = {'0', 'A', 'C'}
    states = {f"s{i}" for i in range(num_states)}
    initial_state = "s0"
    accepting_states = {s for s in states if random.random() < accept_prob}
    transitions = {}
    state_list = list(states)
    for s in states:
        for symbol in alphabet:
            transitions[(s, symbol)] = random.choice(state_list)
    return {
        'alphabet': alphabet,
        'states': states,
        'initial_state': initial_state,
        'accepting_states': accepting_states,
        'transitions': transitions
    }

def generate_random_dfa_satisfying_spec(num_states, verbose=False):
    """
    Generates a random DFA D with 'num_states' states that satisfies the specification S.
    
    Args:
        num_states: Number of states in the generated DFA
        verbose: Whether to print debug information
    """
    alphabet = SPEC_DFA['alphabet']
    dfa_states = [f"s{i}" for i in range(num_states)]
    
    # Create a surjective mapping f: D.states -> SPEC_DFA.states.
    # First, start by mapping the initial state correctly
    shadows = {dfa_states[0]: SPEC_DFA['initial_state']}  # Force s0 to shadow q0
    
    # Then randomly assign the rest
    for s in dfa_states[1:]:
        shadows[s] = random.choice(list(SPEC_DFA['states']))
    
    # Force surjectivity: ensure every state in SPEC_DFA appears.
    for q in SPEC_DFA['states']:
        if q not in shadows.values():
            # Don't change s0's shadow
            candidates = [s for s in dfa_states[1:] if s not in shadows]
            if candidates:  # If there are still unassigned states
                chosen = random.choice(candidates)
                shadows[chosen] = q
            else:
                # Choose a random non-s0 state to reassign
                chosen = random.choice([s for s in dfa_states[1:]])
                shadows[chosen] = q
    
    # CRITICAL CHECK: Confirm initial state mapping
    if verbose:
        print(f"Initial state s0 shadows: {shadows['s0']}")
        
    if shadows['s0'] != SPEC_DFA['initial_state']:
        if verbose:
            print("WARNING: Initial state not mapping to initial state!")
    
    # Shadow mapping validation
    if verbose:
        shadow_counts = {q: sum(1 for s in shadows.values() if s == q) for q in SPEC_DFA['states']}
        print(f"Shadow distribution: {shadow_counts}")
    
    transitions = {}
    for s in dfa_states:
        for symbol in alphabet:
            # Determine the target shadow from SPEC_DFA.
            target_shadow = SPEC_DFA['transitions'][(shadows[s], symbol)]
            # Choose a candidate t such that f(t) = target_shadow.
            candidates = [t for t in dfa_states if shadows[t] == target_shadow]
            # (Candidates should be nonempty because f is surjective.)
            t_target = random.choice(candidates)
            transitions[(s, symbol)] = t_target

    initial_state = "s0"
    # Accepting states are those whose shadow is accepting in SPEC_DFA.
    accepting_states = {s for s in dfa_states if shadows[s] in SPEC_DFA['accepting_states']}
    
    # Verify the construction matches our intentions
    dfa = {
        'alphabet': alphabet,
        'states': set(dfa_states),
        'initial_state': initial_state,
        'accepting_states': accepting_states,
        'transitions': transitions
    }
    
    # Verify this DFA actually satisfies the spec
    result = check_suffix_inclusion(dfa)
    if verbose:
        print(f"Generated DFA satisfies spec? {result}")
        if not result:
            print("ERROR: Generated DFA does not satisfy the specification!")
    
    return dfa