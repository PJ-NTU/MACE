# MACE evolved heuristic 07/10 for problem: constrained_guillotine_cutting
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Refined heuristic for Constrained Guillotine Cutting.
    
    Diagnosis of parent weakness:
    1. The 'shatter' mechanism is too destructive and lacks a guiding heuristic for re-insertion.
    2. It lacks a strategy to balance exploration (different initial packings) and exploitation.
    3. The reliance on random shuffles for re-insertion ignores the high-value pieces that 
       should be prioritized.
    
    Redesign:
    1. Use a multi-start strategy between density-based greedy and area-based greedy.
    2. Implement a focused local search that prioritizes replacing low-value pieces with 
       high-value pieces, rather than random removals.
    3. Use a time-budgeted approach that spends more time on refinement of the best found solution.
    """
    start_time = time.time()
    
    # 1. Generate strong initial candidates
    candidates = []
    
    # Strategy A: Value-density greedy
    try:
        sol_a = tools['bottom_left_pack_demand_aware']()
        candidates.append({"total_value": tools['total_value_of'](sol_a), "placements": sol_a})
    except:
        pass
        
    # Strategy B: Area-based BFD greedy
    try:
        sol_b = tools['guillotine_pack_BFD']()
        candidates.append({"total_value": tools['total_value_of'](sol_b), "placements": sol_b})
    except:
        pass
        
    if not candidates:
        return {"total_value": 0, "placements": []}
    
    best_sol = max(candidates, key=lambda x: x["total_value"])
    
    # 2. Iterative Improvement
    # Instead of random shattering, we use a Hill Climbing approach:
    # Try to swap out the piece with the lowest value in the current set 
    # for the highest value piece type not yet at max capacity.
    
    m = tools['n_piece_types']()
    piece_info = [
        {'id': i + 1, 'value': tools['piece_value'](i + 1)} 
        for i in range(m)
    ]
    # Sort by value descending
    piece_info.sort(key=lambda x: x['value'], reverse=True)
    
    while time.time() - start_time < time_limit_s * 0.7:
        current_placements = list(best_sol["placements"])
        if not current_placements:
            break
            
        # Select a victim to remove: the one with the lowest value
        # This is a heuristic to make room for more valuable pieces
        current_placements.sort(key=lambda p: tools['piece_value'](p[0]))
        victim = current_placements.pop(0)
        
        # Try to replace with a higher value piece
        improved = False
        for p_type in piece_info:
            if tools['used_count'](current_placements, p_type['id']) < tools['piece_demand_max'](p_type['id']):
                attempt = tools['try_place_piece'](current_placements, p_type['id'], orient=0)
                if attempt is not None:
                    new_val = tools['total_value_of'](attempt)
                    if new_val > best_sol["total_value"]:
                        best_sol = {"total_value": new_val, "placements": attempt}
                        improved = True
                        break
        
        if not improved:
            # If no improvement, stop early or reset
            break
            
    # 3. Final polish
    remaining_time = max(0.1, time_limit_s - (time.time() - start_time))
    try:
        improved = tools['apply_local_swap'](best_sol["placements"], t_limit=remaining_time)
        val = tools['total_value_of'](improved)
        if val > best_sol["total_value"]:
            best_sol = {"total_value": val, "placements": improved}
    except:
        pass

    # Final check
    is_valid, _ = tools['is_feasible'](best_sol)
    if not is_valid:
        # Fallback to the best initial greedy if local search failed
        return max(candidates, key=lambda x: x["total_value"])
        
    return best_sol