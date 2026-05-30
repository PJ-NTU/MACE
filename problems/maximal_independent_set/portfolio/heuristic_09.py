# MACE evolved heuristic 09/10 for problem: maximal_independent_set
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Dispatcher-style solver for Maximum Independent Set.
    
    Heuristic: 
    - Small/Sparse graphs are handled effectively by ILP and aggressive local search.
    - Large/Dense graphs require more randomized constructive exploration and (2,1)-swaps.
    
    Modification: Enhanced the randomized construction by using a semi-greedy 
    probability-based selection (RCL) instead of purely picking from the min-degree set,
    allowing for greater exploration of the search space in the 'A-style' regime.
    """
    start_time = time.time()
    graph = instance['graph']
    n = tools['n_vertices']()
    m = tools['n_edges']()
    
    density = (2 * m) / (n * (n - 1)) if n > 1 else 0
    is_small_or_sparse = (n <= 200) or (density < 0.05)
    
    best_mis = tools['greedy_min_degree']()
    
    if is_small_or_sparse:
        if time.time() - start_time < time_limit_s * 0.8:
            ilp_time = min(time_limit_s * 0.5, max(1.0, time_limit_s * 0.3))
            ilp_res = tools['ilp_max_independent_set'](time_limit_s=ilp_time)
            if ilp_res is not None and len(ilp_res) > len(best_mis):
                best_mis = ilp_res
        
        best_mis = tools['apply_swap_2_for_1'](best_mis)
    else:
        best_mis = tools['apply_swap_2_for_1'](best_mis)
        
        while time.time() - start_time < time_limit_s * 0.85:
            current_set = set()
            candidates = set(graph.nodes())
            
            while candidates:
                # Calculate residual degrees
                degrees = {v: len([nb for nb in graph.neighbors(v) if nb in candidates]) for v in candidates}
                d_vals = list(degrees.values())
                min_d, max_d = min(d_vals), max(d_vals)
                
                # Semi-greedy: Pick from a Restricted Candidate List (RCL) 
                # including nodes with degree within a margin of the minimum
                margin = 0.2 * (max_d - min_d)
                rcl = [v for v in degrees if degrees[v] <= min_d + margin]
                
                choice = random.choice(rcl)
                current_set.add(choice)
                
                candidates.remove(choice)
                for nb in graph.neighbors(choice):
                    if nb in candidates:
                        candidates.remove(nb)
            
            refined = tools['apply_swap_2_for_1'](list(current_set))
            if len(refined) > len(best_mis):
                best_mis = refined
                
    if time.time() - start_time < time_limit_s * 0.95:
        best_mis = tools['apply_local_swap'](best_mis, t_limit=max(0.1, time_limit_s * 0.05))
        
    return {"mis_nodes": list(best_mis)}