# MACE evolved heuristic 08/10 for problem: maximal_independent_set
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Synthesized heuristic for Maximum Independent Set:
    1. Warm start with greedy min-degree and (2,1)-swaps.
    2. Repeated construction phase using randomized greedy selection.
    3. Aggressive local search using (2,1)-swaps and time-bounded local swaps.
    4. Adaptive time management to ensure the best solution is found just before timeout.
    """
    start_time = time.time()
    graph = instance['graph']
    
    # 1. Initial high-quality seed
    best_mis = tools['greedy_min_degree']()
    best_mis = tools['apply_swap_2_for_1'](best_mis)
    
    # 2. Main Iterative Loop
    # We use a time-budgeted approach rather than a fixed iteration count.
    # We reserve time for a final intensive polish.
    while time.time() - start_time < time_limit_s * 0.85:
        # Randomized construction:
        # Instead of just picking min-degree, we pick from a small candidate pool 
        # of low-degree nodes to maintain a balance between quality and diversity.
        current_set = []
        candidates = set(graph.nodes())
        
        while candidates:
            # Calculate degrees in the residual graph
            degrees = {v: sum(1 for nb in graph.neighbors(v) if nb in candidates) for v in candidates}
            min_deg = min(degrees.values())
            
            # Pool of candidates with degree close to min
            pool = [v for v, d in degrees.items() if d <= min_deg + 1]
            pick = random.choice(pool)
            
            current_set.append(pick)
            candidates.remove(pick)
            for nb in graph.neighbors(pick):
                if nb in candidates:
                    candidates.remove(nb)
        
        # Immediate improvement
        refined = tools['apply_swap_2_for_1'](current_set)
        
        if len(refined) > len(best_mis):
            best_mis = refined
            
        # Occasional intensive local search on the current best
        if random.random() < 0.15:
            polished = tools['apply_local_swap'](best_mis, t_limit=min(0.2, (time_limit_s * 0.9 - (time.time() - start_time))))
            if len(polished) > len(best_mis):
                best_mis = polished

    # 3. Final Polish
    # Use remaining time for final refinement using the most robust tools available.
    remaining_time = time_limit_s - (time.time() - start_time)
    if remaining_time > 0.05:
        best_mis = tools['apply_local_swap'](best_mis, t_limit=min(remaining_time, 2.0))
        
    # Final sanity check against ILP if time was very generous, 
    # but prioritize local search stability for most graph sizes.
    if remaining_time > 2.0:
        ilp_res = tools['ilp_max_independent_set'](time_limit_s=min(remaining_time - 0.1, 5.0))
        if ilp_res is not None and len(ilp_res) > len(best_mis):
            best_mis = ilp_res

    return {"mis_nodes": list(best_mis)}