# MACE evolved heuristic 04/10 for problem: maximal_independent_set
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Optimized MIS solver with time-budget management and prioritized heuristics.
    """
    start_time = time.time()
    margin = 0.5
    
    def time_left():
        return time.time() - start_time < time_limit_s - margin

    # 1. Warm start with fast greedy heuristics
    best_mis = tools['greedy_min_degree']()
    
    # Try max-degree removal if time permits
    if time_left():
        res2 = tools['greedy_max_degree_removal']()
        if len(res2) > len(best_mis):
            best_mis = res2

    # 2. Refinement pass using built-in (2,1)-swap
    # This is efficient and keeps the set valid
    if time_left():
        best_mis = tools['apply_swap_2_for_1'](best_mis)

    # 3. Iterative Local Search (ILS)
    # Perform limited restarts with perturbation
    while time_left():
        # Perturbation: drop a small random portion
        if len(best_mis) > 5:
            num_drop = max(1, len(best_mis) // 5)
            perturbed = list(best_mis)
            random.shuffle(perturbed)
            perturbed = perturbed[num_drop:]
            
            # Reconstruct greedily
            forbidden = tools['forbidden_by'](perturbed)
            candidates = [n for n in instance['graph'].nodes() if n not in forbidden]
            random.shuffle(candidates)
            for n in candidates:
                if n not in forbidden:
                    perturbed.append(n)
                    # Update forbidden set incrementally
                    forbidden.update([n] + tools['adjacency'](n))
            
            # Local improvement
            refined = tools['apply_swap_2_for_1'](perturbed)
            if len(refined) > len(best_mis):
                best_mis = refined
        else:
            break
            
    # 4. Final polish with ILP if we have significant time left
    # ILP is powerful but expensive; only run if we have a healthy buffer
    remaining = time_limit_s - (time.time() - start_time)
    if remaining > 2.0:
        # Use a slightly shorter limit to ensure we return before the ceiling
        ilp_res = tools['ilp_max_independent_set'](time_limit_s=remaining * 0.7)
        if ilp_res is not None and len(ilp_res) > len(best_mis):
            best_mis = ilp_res
            
    return {"mis_nodes": list(best_mis)}