# MACE evolved heuristic 03/10 for problem: maximal_independent_set
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Solves the Maximum Independent Set problem using a GRASP-inspired 
    multi-start approach with local search refinement.
    
    Modified: Replaced static top-k selection with a probability-weighted 
    selection based on degree rank to improve diversity in the GRASP phase.
    """
    start_time = time.time()
    graph = instance['graph']
    nodes = list(graph.nodes())
    
    # Initial best from tools
    best_mis = tools['greedy_min_degree']()
    best_mis = tools['apply_swap_2_for_1'](best_mis)
    
    # Iterative improvement phase
    while time.time() - start_time < time_limit_s * 0.8:
        # Randomized construction
        current_set = []
        candidates = set(nodes)
        
        while candidates:
            # Sort by degree in residual graph
            sorted_candidates = sorted(list(candidates), key=lambda n: len([nb for nb in graph.neighbors(n) if nb in candidates]))
            
            # Weighted selection: prioritize low degree, but add stochasticity
            # We use a power-law-like distribution to pick from the top N candidates
            limit = min(len(sorted_candidates), 10)
            weights = [1.0 / (i + 1) for i in range(limit)]
            pick_idx = random.choices(range(limit), weights=weights)[0]
            pick = sorted_candidates[pick_idx]
            
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