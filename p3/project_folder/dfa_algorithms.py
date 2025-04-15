from PySimpleAutomata.DFA import dfa_complementation, dfa_intersection, dfa_nonemptiness_check
from specs.suffix_spec import SPEC_DFA

def check_suffix_inclusion(D):
    """
    Returns True if every word accepted by DFA D ends with either "AC0" or "0CA",
    i.e. if L(D) ⊆ L(S). Otherwise returns False.
    
    It works by complementing SPEC_DFA to get S′, intersecting D with S′, and then
    checking for nonemptiness. An empty intersection indicates that D meets the spec.
    """
    S_complement = dfa_complementation(SPEC_DFA)
    product_dfa = dfa_intersection(D, S_complement)
    return not dfa_nonemptiness_check(product_dfa)