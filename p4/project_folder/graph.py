"""
Graph module for graph operations and cycle detection.
"""
from collections import deque

def build_graph(nfa: dict) -> dict:
    """
    Build a directed graph from the transitions of an NFA.
    """
    graph = { state: set() for state in nfa['states'] }
    for (state, symbol), next_states in nfa['transitions'].items():
        graph[state].update(next_states)
    return graph

def compute_coreachable_states(nfa: dict, graph: dict) -> set:
    """
    Compute the set of states from which an accepting state is reachable.
    """
    reverse_graph = { state: set() for state in nfa['states'] }
    for u in graph:
        for v in graph[u]:
            reverse_graph[v].add(u)
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
    Check whether the subgraph induced by 'nodes' has a cycle using iterative DFS.
    """
    visited = set()
    
    for start in nodes:
        if start in visited:
            continue
            
        # Stack stores (node, iterator_over_neighbors)
        stack = [(start, iter([n for n in graph[start] if n in nodes]))]
        # Track nodes in current exploration path
        rec_stack = {start}
        visited.add(start)
        
        while stack:
            node, neighbors_iter = stack[-1]
            
            try:
                # Get next neighbor
                neighbor = next(neighbors_iter)
                
                if neighbor not in visited:
                    # Visit this neighbor next
                    visited.add(neighbor)
                    rec_stack.add(neighbor)
                    stack.append((neighbor, iter([n for n in graph[neighbor] if n in nodes])))
                elif neighbor in rec_stack:
                    # Cycle detected
                    return True
                    
            except StopIteration:
                # No more neighbors, backtrack
                rec_stack.remove(node)
                stack.pop()
    
    return False