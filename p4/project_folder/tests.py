"""
Testing module for NFA operations.
"""
from models import dfa_to_nfa
from graph import build_graph, has_cycle_in_subgraph, compute_coreachable_states
from algorithms import check_finite_acceptance

def run_unit_tests():
    """
    Run unit tests for the various NFA operations.
    """
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
    from models import nfa_intersection
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
    
    # Test for generate_random_nfa (basic structure test)
    from generator import generate_random_nfa
    rand_nfa = generate_random_nfa(10, "ab")
    assert 'alphabet' in rand_nfa and 'states' in rand_nfa and 'transitions' in rand_nfa
    
    print("All unit tests passed.")