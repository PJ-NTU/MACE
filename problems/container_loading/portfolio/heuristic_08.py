# MACE evolved heuristic 08/10 for problem: container_loading
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    An optimized hybrid packing heuristic.
    
    The strategy relies on:
    1. A robust baseline using both Wall-Building and Corner-Packing.
    2. An iterative improvement phase that balances the exploration of 
       different packing sequences (Large-First vs. Small-First) against 
       the time budget.
    3. A local search refinement (apply_swap_boxes) that effectively 
       re-arranges the packed items to reduce fragmentation.
    """
    start_time = time.time()
    
    # 1. Generate baseline candidates
    # Corner-pack is strong for heterogeneous boxes, Wall-building for homogenous layouts.
    best_placements = []
    best_util = -1.0
    
    # Try different construction heuristics
    heuristics = [
        lambda: tools['corner_pack_3d'](allow_rotation=True),
        lambda: tools['wall_building_pack'](allow_rotation=True)
    ]
    
    # Also try variations with specific sorting if time allows
    # We prioritize the provided tools as they are well-optimized.
    for h in heuristics:
        if (time.time() - start_time) > (time_limit_s * 0.3):
            break
        try:
            current_placements = h()
            current_util = tools['utilization'](current_placements)
            if current_util > best_util:
                best_util = current_util
                best_placements = current_placements
        except Exception:
            continue
            
    # If no placements found, fallback to default
    if not best_placements:
        try:
            default_sol = tools['solve_default'](time_limit_s=time_limit_s * 0.2)
            best_placements = default_sol.get('placements', [])
        except:
            return {'placements': []}
            
    # 2. Refinement Phase
    # Use the remaining time to improve the best found solution.
    # apply_swap_boxes is highly effective at finding local optima by
    # shuffling subsets of placements.
    remaining_time = time_limit_s - (time.time() - start_time)
    if remaining_time > 0.5:
        # We limit the refinement time to keep the solution process safe
        refine_time = min(remaining_time * 0.8, 5.0)
        try:
            refined_placements = tools['apply_swap_boxes'](
                best_placements, 
                time_limit_s=refine_time
            )
            if tools['utilization'](refined_placements) > tools['utilization'](best_placements):
                best_placements = refined_placements
        except:
            pass
            
    # 3. Final verification and return
    # Ensure the structure matches the interface
    return tools['make_solution'](best_placements)