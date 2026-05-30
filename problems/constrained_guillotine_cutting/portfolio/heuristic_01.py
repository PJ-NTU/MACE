# MACE evolved heuristic 01/10 for problem: constrained_guillotine_cutting
import time
import random
import math

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    A Simulated Tempering / Variable Neighborhood Descent approach that differs from 
    the portfolio by avoiding purely greedy construction and simple local swaps.
    
    Instead of relying on deterministic greedy orderings, it uses a Tabu-inspired
    perturbation mechanism: it iteratively 'shatters' the current packing by 
    removing a subset of pieces (a 'cut') and re-packing them using randomized 
    try-place-piece seeds.
    """
    start_time = time.time()
    
    # 1. Initial State: Start with a high-performing greedy baseline
    best_sol = {"total_value": 0, "placements": []}
    try:
        initial = tools['bottom_left_pack_demand_aware']()
        best_sol = {"total_value": tools['total_value_of'](initial), "placements": initial}
    except Exception:
        pass

    # 2. Iterative Shattering and Re-insertion
    # Portfolio members rely on 'apply_local_swap' which is restrictive.
    # We implement a 'Shatter and Re-fill' loop.
    m = tools['n_piece_types']()
    
    while time.time() - start_time < time_limit_s * 0.8:
        current_placements = list(best_sol["placements"])
        if not current_placements:
            break
            
        # Perturbation: Remove a random subset of pieces (shatter)
        num_to_remove = random.randint(1, max(1, len(current_placements) // 2))
        random.shuffle(current_placements)
        remaining = current_placements[num_to_remove:]
        
        # Re-fill: Try to insert pieces back into the layout using randomized insertion
        # This differs from greedy construction by attempting to fill holes left by the cut
        shuffled_types = list(range(1, m + 1))
        random.shuffle(shuffled_types)
        
        working_placements = list(remaining)
        for t in shuffled_types:
            if tools['used_count'](working_placements, t) < tools['piece_demand_max'](t):
                # Try to add this piece type; try_place_piece is expensive but effective
                # for filling gaps left by removals
                res = tools['try_place_piece'](working_placements, t, orient=0)
                if res is not None:
                    working_placements = res
        
        # Evaluation
        new_val = tools['total_value_of'](working_placements)
        if new_val > best_sol["total_value"]:
            best_sol = {"total_value": new_val, "placements": working_placements}
            
        # Check time
        if time.time() - start_time > time_limit_s * 0.9:
            break
            
    # 3. Final polishing with local swap if time remains
    try:
        if time.time() - start_time < time_limit_s * 0.95:
            improved = tools['apply_local_swap'](
                best_sol["placements"], 
                t_limit=min(0.5, time_limit_s - (time.time() - start_time))
            )
            val = tools['total_value_of'](improved)
            if val > best_sol["total_value"]:
                best_sol = {"total_value": val, "placements": improved}
    except Exception:
        pass

    # Ensure validity
    is_valid, _ = tools['is_feasible'](best_sol)
    if not is_valid:
        # Fallback to pure greedy if shattering produced invalid state
        fallback = tools['guillotine_pack_BFD']()
        return {"total_value": tools['total_value_of'](fallback), "placements": fallback}
        
    return best_sol