"""
Algorithms module for NFA operations and analysis.
"""
from models import dfa_to_nfa, nfa_intersection
from graph import build_graph, compute_coreachable_states, has_cycle_in_subgraph

def check_finite_acceptance(nfa: dict, spec_dfa: dict) -> bool:
    """
    Check whether the intersection L(N) ∩ L(spec_dfa) is finite.
    Returns True if finite, False if infinite.
    
    This is the core algorithm for determining if an NFA's language 
    is finite when intersected with a specification DFA.
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