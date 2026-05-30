# MACE evolved heuristic 01/10 for problem: maximal_independent_set
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Solves the Maximum Independent Set problem using a GRASP-inspired 
    multi-start approach with local search refinement.
    """
    start_time = time.time()
    graph = instance['graph']
    nodes = list(graph.nodes())
    
    best_mis = []
    
    # Heuristic strategy:
    # 1. Start with a greedy construction (Min-Degree)
    # 2. Refine using (2,1)-swaps
    # 3. If time allows, perform randomized restarts with local search
    
    def get_greedy_mis():
        # Min-degree greedy construction
        res_nodes = set(nodes)
        mis = []
        while res_nodes:
            # Pick node with minimum degree in the residual graph
            curr_node = min(res_nodes, key=lambda n: sum(1 for neighbor in graph.neighbors(n) if neighbor in res_nodes))
            mis.append(curr_node)
            # Remove node and neighbors
            res_nodes.remove(curr_node)
            for neighbor in graph.neighbors(curr_node):
                if neighbor in res_nodes:
                    res_nodes.remove(neighbor)
        return mis

    # Initial best from tools
    best_mis = tools['greedy_min_degree']()
    best_mis = tools['apply_swap_2_for_1'](best_mis)
    
    # Iterative improvement phase
    while time.time() - start_time < time_limit_s * 0.8:
        # Randomized construction
        current_set = []
        candidates = set(nodes)
        
        while candidates:
            # Pick from top-k min-degree nodes to introduce randomness
            subset = sorted(list(candidates), key=lambda n: len([nb for nb in graph.neighbors(n) if nb in candidates]))
            pick = random.choice(subset[:min(3, len(subset))])
            
            current_set.append(pick)
            candidates.remove(pick)
            for n in graph.neighbors(pick):
                if n in candidates:
                    candidates.remove(n)
        
        # Local search refinement
        refined = tools['apply_swap_2_for_1'](current_set)
        
        if len(refined) > len(best_mis):
            best_mis = refined
            
    # Final check against ILP if time permits
    if time.time() - start_time < time_limit_s * 0.95:
        # Use ILP to polish the result if the graph is small enough
        ilp_res = tools['ilp_max_independent_set'](time_limit_s=max(0.1, time_limit_s * 0.05))
        if ilp_res is not None and len(ilp_res) > len(best_mis):
            best_mis = ilp_res

    return {"mis_nodes": list(best_mis)}