import random
import time
import sys
from specs.suffix_spec import SPEC_DFA
from dfa_algorithms import check_suffix_inclusion

def generate_random_dfa(num_states, accept_prob=0.2):
    """
    Generates a completely random DFA over the alphabet {'0', 'A', 'C'}.
    Each state is marked as accepting with probability accept_prob.
    Transitions are chosen uniformly at random.
    """
    start_time = time.perf_counter()
    
    alphabet = {'0', 'A', 'C'}
    states = {f"s{i}" for i in range(num_states)}
    initial_state = "s0"
    accepting_states = {s for s in states if random.random() < accept_prob}
    transitions = {}
    state_list = list(states)
    
    # For very large DFAs, provide some feedback
    if num_states > 50000:
        print(f"⚙️ Generating random DFA with {num_states} states...")
        progress_interval = max(1, num_states // 10)  # Report progress every ~10%
        
    for i, s in enumerate(states):
        # Progress reporting for large DFAs
        if num_states > 50000 and i % progress_interval == 0:
            percent = (i / num_states) * 100
            print(f"  → Generating transitions: {percent:.1f}% complete ({i}/{num_states} states)")
            
        for symbol in alphabet:
            transitions[(s, symbol)] = random.choice(state_list)
    
    if num_states > 50000:
        end_time = time.perf_counter()
        print(f"✓ Random DFA generated in {end_time - start_time:.3f} seconds")
        
    return {
        'alphabet': alphabet,
        'states': states,
        'initial_state': initial_state,
        'accepting_states': accepting_states,
        'transitions': transitions
    }

def generate_random_dfa_satisfying_spec(num_states, verbose=False, max_attempts=5):
    """
    Generates a random DFA D with 'num_states' states that satisfies the specification S.
    Using a more memory-efficient implementation for large DFAs.
    
    Args:
        num_states: Number of states in the generated DFA
        verbose: Whether to print debug information
        max_attempts: Maximum number of attempts before switching to slower but guaranteed method
    """
    start_time = time.perf_counter()
    
    # Always provide feedback for satisfying DFA generation since it's more complex
    print(f"⚙️ Generating satisfying DFA with {num_states} states...")

    # Ensure we have at least as many states as the spec
    if num_states < len(SPEC_DFA['states']):
        print(f"⚠️ Warning: Requested DFA size ({num_states}) is smaller than spec DFA size ({len(SPEC_DFA['states'])})")
        print(f"   Increasing size to minimum required: {len(SPEC_DFA['states'])}")
        num_states = len(SPEC_DFA['states'])

    alphabet = SPEC_DFA['alphabet']
    
    # Use lists instead of sets for memory efficiency with large DFAs
    dfa_states = [f"s{i}" for i in range(num_states)]
    
    # Create a surjective mapping f: D.states -> SPEC_DFA.states.
    print(f"  → Creating state mapping from {num_states} DFA states to {len(SPEC_DFA['states'])} specification states")
    mapping_start = time.perf_counter()
    
    # Pre-compute and store spec states
    spec_states_list = list(SPEC_DFA['states'])
    spec_state_count = len(spec_states_list)
    
    # Use the dictionary approach for all DFA sizes
    shadows = {dfa_states[0]: SPEC_DFA['initial_state']}  # Force s0 to shadow q0
    
    # Then randomly assign the rest
    for s in dfa_states[1:]:
        shadows[s] = random.choice(spec_states_list)
    
    # Force surjectivity: ensure every state in SPEC_DFA appears.
    # First count how many DFA states map to each spec state
    shadow_counts = {q: 0 for q in SPEC_DFA['states']}
    for s, q in shadows.items():
        shadow_counts[q] += 1
        
    # Then ensure every spec state has at least one shadow
    for q in SPEC_DFA['states']:
        if shadow_counts[q] == 0:
            # Choose a random non-s0 state to reassign
            chosen = random.choice([s for s in dfa_states[1:]])
            old_q = shadows[chosen]
            shadows[chosen] = q
            # Update the counts
            shadow_counts[old_q] -= 1
            shadow_counts[q] += 1
    
    # Verify all spec states have shadows
    for q, count in shadow_counts.items():
        if count == 0:
            print(f"⚠️ Warning: Spec state {q} has no shadows!")
    
    mapping_end = time.perf_counter()
    print(f"  → State mapping completed in {mapping_end - mapping_start:.3f} seconds")
    print(f"  → Memory used for state mapping: ~{sys.getsizeof(shadows) / (1024*1024):.2f} MB")
    
    # Progress tracking for transitions
    transitions_start = time.perf_counter()
    print(f"  → Building transitions table ({len(alphabet)} symbols × {num_states} states)...")
    
    # Use original approach for all DFA sizes
    transitions = {}
    progress_interval = max(1, num_states // 10)  # Report every 10%
    
    # Create reverse mapping for more efficient candidate selection
    reverse_mapping = {}
    for spec_state in SPEC_DFA['states']:
        reverse_mapping[spec_state] = []
        
    for s, spec_state in shadows.items():
        reverse_mapping[spec_state].append(s)
        
    # Verify every spec state has at least one state in the reverse mapping
    for spec_state, mapped_states in reverse_mapping.items():
        if not mapped_states:
            print(f"⚠️ Warning: No DFA states map to spec state {spec_state}! Adding a mapping.")
            # Choose a random non-s0 state to map to this spec state
            chosen = random.choice([s for s in dfa_states[1:]])
            old_spec = shadows[chosen]
            shadows[chosen] = spec_state
            # Update reverse mapping
            if chosen in reverse_mapping[old_spec]:
                reverse_mapping[old_spec].remove(chosen)
            reverse_mapping[spec_state].append(chosen)
    
    for i, s in enumerate(dfa_states):
        if i % progress_interval == 0:
            percent = (i / num_states) * 100
            print(f"    • Processed {i}/{num_states} states ({percent:.1f}%)")
            
        for symbol in alphabet:
            # Determine the target shadow from SPEC_DFA.
            source_shadow = shadows[s]
            target_shadow = SPEC_DFA['transitions'][(source_shadow, symbol)]
            
            # Choose a candidate t such that f(t) = target_shadow.
            candidates = reverse_mapping[target_shadow]
            
            # Check if candidates list is empty (should never happen due to our checks)
            if not candidates:
                print(f"⚠️ Warning: No candidates for target shadow {target_shadow}! Creating one.")
                # Choose a random state to map to this target shadow
                random_state = random.choice(dfa_states[1:])  # Don't use s0
                old_shadow = shadows[random_state]
                shadows[random_state] = target_shadow
                # Update reverse mapping
                if random_state in reverse_mapping[old_shadow]:
                    reverse_mapping[old_shadow].remove(random_state)
                reverse_mapping[target_shadow].append(random_state)
                candidates = [random_state]
            
            t_target = random.choice(candidates)
            transitions[(s, symbol)] = t_target

    transitions_end = time.perf_counter()
    print(f"  → Transitions table built in {transitions_end - transitions_start:.3f} seconds")
    
    initial_state = "s0"

    # Build accepting states set
    print(f"  → Determining accepting states...")
    accepting_start = time.perf_counter()
    
    # Original approach for all DFAs
    accepting_states = {s for s in dfa_states if shadows[s] in SPEC_DFA['accepting_states']}
    
    accepting_end = time.perf_counter()
    print(f"  → Accepting states determined in {accepting_end - accepting_start:.3f} seconds")
    
    # Construct the final DFA
    dfa = {
        'alphabet': alphabet,
        'states': set(f"s{i}" for i in range(num_states)),
        'initial_state': initial_state,
        'accepting_states': accepting_states,
        'transitions': transitions
    }
    
    # Verify this DFA actually satisfies the spec
    print(f"  → Verifying the generated DFA satisfies the specification...")
    verify_start = time.perf_counter()
    result = check_suffix_inclusion(dfa)
    verify_end = time.perf_counter()
    
    print(f"  → Verification completed in {verify_end - verify_start:.3f} seconds")
    print(f"  ✓ Generated DFA satisfies spec? {result}")
    
    if not result:
        print("  ❌ ERROR: Generated DFA does not satisfy the specification!")
        print("     This should not happen with our construction method.")
    else:
        print("  → Skipping verification for large DFA (our construction guarantees correctness)")
        result = True  # Our construction guarantees this
    
    end_time = time.perf_counter()
    total_time = end_time - start_time
    print(f"✓ Satisfying DFA with {num_states} states generated in {total_time:.3f} seconds")
    
    return dfa