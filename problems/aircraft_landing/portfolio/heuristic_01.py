# MACE evolved heuristic 01/10 for problem: aircraft_landing
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Solve the aircraft landing problem using a multi-start greedy construction 
    followed by a local search (hill climbing).
    """
    start_time = time.time()
    num_planes = instance['num_planes']
    
    def get_best_feasible():
        best_sol = None
        best_penalty = float('inf')
        
        # Two constructive strategies
        constructors = [
            tools['greedy_target_time_construct'],
            tools['construct_by_appearance_order']
        ]
        
        for constructor in constructors:
            if time.time() - start_time > time_limit_s * 0.4:
                break
            
            try:
                candidate = constructor()
                if candidate:
                    feasible, _ = tools['validate_schedule'](candidate)
                    if feasible:
                        penalty = tools['penalty_of_schedule'](candidate)
                        if penalty < best_penalty:
                            best_penalty = penalty
                            best_sol = candidate
            except:
                continue
                
        return best_sol

    # 1. Initialization
    current_schedule = get_best_feasible()
    
    # If no feasible solution found by greedy, try a fallback
    if current_schedule is None:
        fallback = {}
        for i in range(1, num_planes + 1):
            p = instance['planes'][i-1]
            # Try to place at target, fallback to earliest
            fallback[i] = {"landing_time": p['target'], "runway": 1}
        current_schedule = fallback
        
        # Ensure we return a valid structure
        if not current_schedule:
            return {"schedule": {}}

    # 2. Local Search
    # Iteratively apply swaps to improve the penalty
    while time.time() - start_time < time_limit_s * 0.95:
        improved = tools['apply_swap_landings'](current_schedule, t_limit=0.1)
        
        # Check if the swap actually improved the cost
        if improved != current_schedule:
            # Validate feasibility before accepting
            feasible, _ = tools['validate_schedule'](improved)
            if feasible:
                new_penalty = tools['penalty_of_schedule'](improved)
                old_penalty = tools['penalty_of_schedule'](current_schedule)
                if new_penalty < old_penalty:
                    current_schedule = improved
                else:
                    break
            else:
                break
        else:
            break
            
    return {"schedule": current_schedule}