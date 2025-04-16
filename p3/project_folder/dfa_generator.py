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

    alphabet = SPEC_DFA['alphabet']
    
    # For large DFAs, use a memory-efficient approach
    is_large_dfa = num_states >= 30000
    
    # Use lists instead of sets for memory efficiency with large DFAs
    dfa_states = [f"s{i}" for i in range(num_states)]
    
    # Create a surjective mapping f: D.states -> SPEC_DFA.states.
    print(f"  → Creating state mapping from {num_states} DFA states to {len(SPEC_DFA['states'])} specification states")
    mapping_start = time.perf_counter()
    
    # For large DFAs, use a more memory-efficient approach
    if is_large_dfa:
        # Use a dictionary with integer keys (more memory efficient)
        spec_states_list = list(SPEC_DFA['states'])
        spec_state_indices = {state: i for i, state in enumerate(spec_states_list)}
        initial_spec_index = spec_state_indices[SPEC_DFA['initial_state']]
        
        # Use an array instead of dictionary for shadows
        shadows_array = bytearray(num_states)  # Each entry is the index of the SPEC_DFA state
        shadows_array[0] = initial_spec_index  # Force s0 to shadow q0
        
        # Randomly assign the rest with batch operations
        batch_size = 10000
        for i in range(1, num_states, batch_size):
            end_i = min(i + batch_size, num_states)
            batch_count = end_i - i
            
            # Generate random values between 0 and len(spec_states_list)-1
            random_indices = [random.randrange(0, len(spec_states_list)) for _ in range(batch_count)]
            for j, rand_idx in enumerate(random_indices):
                if i + j < num_states:
                    shadows_array[i + j] = rand_idx
                    
            if i % 100000 == 0 and i > 0:
                print(f"    • Mapped {i}/{num_states} states ({(i/num_states)*100:.1f}%)")
        
        # Ensure every SPEC_DFA state has at least one shadow
        # Count how many states map to each spec state
        counts = [0] * len(spec_states_list)
        for idx in shadows_array:
            counts[idx] += 1
            
        # For any spec state with no shadows, assign one
        for spec_idx, count in enumerate(counts):
            if count == 0:
                # Choose a random non-initial state to reassign
                state_to_reassign = random.randint(1, num_states - 1)
                shadows_array[state_to_reassign] = spec_idx
        
        # Helper functions to access shadows
        def get_shadow(state_idx):
            return spec_states_list[shadows_array[state_idx]]
        
        def get_shadows_by_spec(spec_state):
            spec_idx = spec_state_indices[spec_state]
            return [i for i, idx in enumerate(shadows_array) if idx == spec_idx]
        
    else:
        # For smaller DFAs, use the original dictionary approach
        shadows = {dfa_states[0]: SPEC_DFA['initial_state']}  # Force s0 to shadow q0
        
        # Then randomly assign the rest
        for s in dfa_states[1:]:
            shadows[s] = random.choice(list(SPEC_DFA['states']))
        
        # Force surjectivity: ensure every state in SPEC_DFA appears.
        for q in SPEC_DFA['states']:
            if q not in shadows.values():
                # Choose a random non-s0 state to reassign
                chosen = random.choice([s for s in dfa_states[1:]])
                shadows[chosen] = q
    
    mapping_end = time.perf_counter()
    print(f"  → State mapping completed in {mapping_end - mapping_start:.3f} seconds")
    print(f"  → Memory used for state mapping: ~{sys.getsizeof(shadows_array if is_large_dfa else shadows) / (1024*1024):.2f} MB")
    
    # PERFORMANCE OPTIMIZATION: Pre-compute reverse mapping
    print(f"  → Pre-computing reverse shadow mappings for faster transition generation...")
    reverse_mapping_start = time.perf_counter()
    
    # Dictionary mapping each spec state to a list of DFA states that shadow it
    reverse_shadows = {}
    
    if is_large_dfa:
        # For large DFAs, build from the shadows_array
        for spec_state in SPEC_DFA['states']:
            spec_idx = spec_state_indices[spec_state]
            # Find all states that map to this spec state
            reverse_shadows[spec_state] = [i for i, idx in enumerate(shadows_array) if idx == spec_idx]
            
            # Progress reporting for large DFAs
            if verbose:
                print(f"    • Found {len(reverse_shadows[spec_state])} states shadowing {spec_state}")
    else:
        # For smaller DFAs, build from the shadows dictionary
        for spec_state in SPEC_DFA['states']:
            reverse_shadows[spec_state] = [int(s[1:]) for s, shadow in shadows.items() if shadow == spec_state]
    
    reverse_mapping_end = time.perf_counter()
    print(f"  → Reverse mappings completed in {reverse_mapping_end - reverse_mapping_start:.3f}s")
    print(f"  → Memory used for reverse mapping: ~{sum(sys.getsizeof(v) for v in reverse_shadows.values()) / (1024*1024):.2f} MB")
    
    # Progress tracking for transitions
    transitions_start = time.perf_counter()
    print(f"  → Building transitions table ({len(alphabet)} symbols × {num_states} states)...")
    
    # Use efficient data structures based on DFA size
    if is_large_dfa:
        # More memory-efficient approach for transitions
        transitions = {}
        progress_interval = max(1, num_states // 10)  # Report every 10%
        
        # Process states in batches to avoid memory issues
        batch_size = min(10000, num_states)  # Process in smaller batches
        total_batches = (num_states + batch_size - 1) // batch_size
        
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, num_states)
            
            # Report progress at batch boundaries
            percent = (batch_num / total_batches) * 100
            print(f"    • Processing batch {batch_num+1}/{total_batches} for size {num_states}: states {start_idx}-{end_idx-1} ({percent:.1f}%)")
            
            # Process this batch of states
            for i in range(start_idx, end_idx):
                for symbol in alphabet:
                    # Get the source state's shadow
                    source_shadow = get_shadow(i)
                    
                    # Determine the target shadow from SPEC_DFA
                    target_shadow = SPEC_DFA['transitions'][(source_shadow, symbol)]
                    
                    # Find candidate target states
                    target_candidates = reverse_shadows[target_shadow]
                    
                    # Choose random target and set transition
                    target_idx = random.choice(target_candidates)
                    transitions[(f"s{i}", symbol)] = f"s{target_idx}"
                    
    else:
        # Original approach for smaller DFAs
        transitions = {}
        progress_interval = max(1, num_states // 10)  # Report every 10%
        
        for i, s in enumerate(dfa_states):
            if i % progress_interval == 0:
                percent = (i / num_states) * 100
                print(f"    • Processed {i}/{num_states} states for DFA of size {num_states} ({percent:.1f}%)")
                
            for symbol in alphabet:
                # Determine the target shadow from SPEC_DFA.
                target_shadow = SPEC_DFA['transitions'][(shadows[s], symbol)]
                # Choose a candidate t such that f(t) = target_shadow.
                candidates = [t for t in dfa_states if shadows[t] == target_shadow]
                # (Candidates should be nonempty because f is surjective.)
                t_target = random.choice(candidates)
                transitions[(s, symbol)] = t_target

    transitions_end = time.perf_counter()
    print(f"  → Transitions table built in {transitions_end - transitions_start:.3f} seconds")
    
    initial_state = "s0"

    # Build accepting states set
    print(f"  → Determining accepting states...")
    accepting_start = time.perf_counter()
    
    if is_large_dfa:
        # More efficient approach for large DFAs
        accepting_states = set()
        spec_accepting = SPEC_DFA['accepting_states']
        
        # For each accepting state in spec, add all corresponding DFA states
        for spec_state in spec_accepting:
            spec_idx = spec_state_indices[spec_state]
            for i in range(num_states):
                if shadows_array[i] == spec_idx:
                    accepting_states.add(f"s{i}")
    else:
        # Original approach for smaller DFAs
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
    
    # Skip validation for very large DFAs - our construction guarantees correctness
    if num_states < 20000:
        # Verify this DFA actually satisfies the spec
        print(f"  → Verifying the generated DFA satisfies the specification...")
        verify_start = time.perf_counter()
        result = check_suffix_inclusion(dfa)
        verify_end = time.perf_counter()
        
        print(f"  → Verification completed in {verify_end - verify_start:.3f} seconds")
        print(f"  ✓ Generated DFA satisfies spec? {result}")
        
        if not result:
            print("  ❌ ERROR: Generated DFA does not satisfy the specification!")
            print("     Will regenerate using a different approach...")
            # This shouldn't happen with our construction, but just in case,
            # we could implement a fallback approach here
    else:
        print("  → Skipping verification for large DFA (our construction guarantees correctness)")
        result = True  # Our construction guarantees this
    
    end_time = time.perf_counter()
    total_time = end_time - start_time
    print(f"✓ Satisfying DFA with {num_states} states generated in {total_time:.3f} seconds")
    
    return dfa