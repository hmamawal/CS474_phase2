"""
Generator module for creating random NFAs.
"""
import random

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
        'states': {f"q{i}" for i in range(num_states)},
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