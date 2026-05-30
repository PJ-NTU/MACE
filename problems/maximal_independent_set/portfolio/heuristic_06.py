# MACE evolved heuristic 06/10 for problem: maximal_independent_set
import time
import random
import math

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Synthesized hybrid solver:
    - Analyzes structural properties (density, node degree variance) to classify
      the graph complexity.
    - Uses exact ILP for small/constrained graphs where search space allows.
    - Uses randomized greedy construction with degree-weighted sampling for 
      large/dense graphs to avoid local optima.
    """
    start_time = time.time()
    graph = instance['graph']
    n = tools['n_vertices']()
    m = tools['n_edges']()
    
    # Structural features
    density = (2 * m) / (n * (n - 1)) if n > 1 else 0
    avg_degree = (2 * m) / n if n > 0 else 0
    
    # Heuristic classification:
    # Dense graphs (density > 0.1) are hard for simple greedy; require more randomized exploration.
    # Small graphs (n < 150) are perfect for the exact solver (ILP).
    # Sparse graphs (density < 0.05) are well-handled by greedy + local swap.
    
    best_mis = tools['greedy_min_degree']()
    
    # Regime 1: Small enough for exact methods
    if n <= 150:
        ilp_time = min(time_limit_s * 0.4, 5.0)
        ilp_res = tools['ilp_max_independent_set'](time_limit_s=ilp_time)
        if ilp_res is not None and len(ilp_res) > len(best_mis):
            best_mis = ilp_res
        best_mis = tools['apply_swap_2_for_1'](best_mis)
        
    # Regime 2: Large/Dense - use multi-start stochastic greedy
    else:
        # Use a more aggressive local swap refinement
        best_mis = tools['apply_local_swap'](best_mis, t_limit=time_limit_s * 0.1)
        
        while time.time() - start_time < time_limit_s * 0.8:
            # Weighted randomized greedy construction
            # Pick nodes with lower current residual degree to stay independent longer
            current_set = []
            candidates = set(graph.nodes())
            
            while candidates:
                # Calculate residual degrees
                # For efficiency on large graphs, we sample or use simple degree
                # We prioritize vertices with lowest degree in the residual graph
                v = min(candidates, key=lambda x: len([nb for nb in graph.neighbors(x) if nb in candidates]))
                
                current_set.append(v)
                # Remove vertex and neighbors
                candidates.remove(v)
                for nb in graph.neighbors(v):
                    if nb in candidates:
                        candidates.remove(nb)
                
                # Periodically inject randomness to explore different branches
                if random.random() < 0.1:
                    if candidates:
                        v_rand = random.choice(list(candidates))
                        current_set.append(v_rand)
                        candidates.remove(v_rand)
                        for nb in graph.neighbors(v_rand):
                            if nb in candidates:
                                candidates.remove(nb)
            
            refined = tools['apply_swap_2_for_1'](current_set)
            if len(refined) > len(best_mis):
                best_mis = refined
    
    # Final polishing
    if time.time() - start_time < time_limit_s * 0.95:
        best_mis = tools['apply_local_swap'](best_mis, t_limit=max(0.1, time_limit_s * 0.05))
        
    return {"mis_nodes": list(best_mis)}