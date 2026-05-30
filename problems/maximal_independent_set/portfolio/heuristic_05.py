# MACE evolved heuristic 05/10 for problem: maximal_independent_set
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Dispatcher-style solver for Maximum Independent Set.
    
    Heuristic: 
    - Small/Sparse graphs are handled effectively by ILP and aggressive local search (B-style).
    - Large/Dense graphs require more randomized constructive exploration and (2,1)-swaps (A-style).
    
    Decision Criteria:
    - n_vertices <= 200: Prioritize ILP (B-style).
    - n_vertices > 200: Prioritize multi-start GRASP with degree-rank stochasticity (A-style).
    """
    start_time = time.time()
    graph = instance['graph']
    n = tools['n_vertices']()
    m = tools['n_edges']()
    
    # Feature calculation: Density
    # Density = 2m / (n(n-1))
    density = (2 * m) / (n * (n - 1)) if n > 1 else 0
    
    # Regime Selection
    # If the graph is small or very sparse, ILP is likely to find the optimum.
    # If the graph is large or dense, randomized construction is more robust.
    is_small_or_sparse = (n <= 200) or (density < 0.05)
    
    best_mis = tools['greedy_min_degree']()
    
    if is_small_or_sparse:
        # B-style: Focus on ILP and simple refinement
        if time.time() - start_time < time_limit_s * 0.8:
            ilp_time = min(time_limit_s * 0.5, max(1.0, time_limit_s * 0.3))
            ilp_res = tools['ilp_max_independent_set'](time_limit_s=ilp_time)
            if ilp_res is not None and len(ilp_res) > len(best_mis):
                best_mis = ilp_res
        
        best_mis = tools['apply_swap_2_for_1'](best_mis)
    else:
        # A-style: Multi-start Randomized Greedy + Local Swap
        best_mis = tools['apply_swap_2_for_1'](best_mis)
        
        while time.time() - start_time < time_limit_s * 0.85:
            current_set = set()
            # Random node ordering for construction
            nodes = list(graph.nodes())
            random.shuffle(nodes)
            
            # Greedy construction with degree-based tie-breaking
            candidates = set(nodes)
            while candidates:
                # Find current min degree among candidates to guide selection
                degrees = {v: len([nb for nb in graph.neighbors(v) if nb in candidates]) for v in candidates}
                min_deg = min(degrees.values())
                choices = [v for v in degrees if degrees[v] == min_deg]
                
                choice = random.choice(choices)
                current_set.add(choice)
                
                # Remove choice and neighbors
                candidates.remove(choice)
                for nb in graph.neighbors(choice):
                    if nb in candidates:
                        candidates.remove(nb)
            
            refined = tools['apply_swap_2_for_1'](list(current_set))
            if len(refined) > len(best_mis):
                best_mis = refined
                
    # Final cleanup: ensure the result is as good as possible within time budget
    if time.time() - start_time < time_limit_s * 0.95:
        best_mis = tools['apply_local_swap'](best_mis, t_limit=max(0.1, time_limit_s * 0.05))
        
    return {"mis_nodes": list(best_mis)}