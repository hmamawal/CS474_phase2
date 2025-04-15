# The specification DFA (S) for suffix inclusion.
SPEC_DFA = {
    'alphabet': {'0', 'A', 'C'},
    'states': {'q0', 'q1', 'q2', 'q3', 'q4', 'q5'},
    'initial_state': 'q0',
    'accepting_states': {'q3'},
    'transitions': {
        ('q0', 'C'): 'q0',
        ('q0', 'A'): 'q1',
        ('q0', '0'): 'q4',

        ('q1', 'C'): 'q2',
        ('q1', 'A'): 'q1',
        ('q1', '0'): 'q4',

        ('q2', 'A'): 'q1',
        ('q2', '0'): 'q3',
        ('q2', 'C'): 'q0',

        ('q3', '0'): 'q4',
        ('q3', 'A'): 'q1',
        ('q3', 'C'): 'q0',

        ('q4', 'C'): 'q5',
        ('q4', '0'): 'q4',
        ('q4', 'A'): 'q1',

        ('q5', 'C'): 'q0',
        ('q5', 'A'): 'q3',
        ('q5', '0'): 'q4',
    }
}