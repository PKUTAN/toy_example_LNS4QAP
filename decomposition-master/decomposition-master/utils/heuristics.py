import networkx as nx


def greedy(graph):
    '''Run the greedy heuristics on the graph for MVC.'''

    selected_nodes = set()
    all_covered = False
    
    while not all_covered:
        all_covered = True
        for e in graph.edges:
            v1, v2 = e

            if v1 in selected_nodes or v2 in selected_nodes:
                continue

            selected_nodes.add(v1)
            selected_nodes.add(v2)
            all_covered = False

    return len(selected_nodes)
        
