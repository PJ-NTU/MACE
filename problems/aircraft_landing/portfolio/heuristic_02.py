# MACE evolved heuristic 02/10 for problem: aircraft_landing
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Implements a GRASP-based approach:
    1. Construct multiple initial solutions using a randomized greedy approach.
    2. Improve the best found solution using local search (swap landings).
    """
    start_time = time.time()
    best_schedule = None
    best_penalty = float('inf')
    
    num_planes = instance['num_planes']
    num_runways = instance['num_runways']
    
    # Helper to calculate penalty for a given schedule
    def calculate_penalty(schedule):
        penalty = 0.0
        for i in range(1, num_planes + 1):
            target = instance['planes'][i-1]['target']
            landing = schedule[i]['landing_time']
            if landing < target:
                penalty += (target - landing) * instance['planes'][i-1]['penalty_early']
            elif landing > target:
                penalty += (landing - target) * instance['planes'][i-1]['penalty_late']
        return penalty

    # Randomized construction
    def construct_randomized():
        # Order planes by target time with some random jitter
        indices = list(range(1, num_planes + 1))
        indices.sort(key=lambda i: instance['planes'][i-1]['target'] + random.uniform(-10, 10))
        
        schedule = {}
        # Track runway availability
        runway_last_landing = {r: -float('inf') for r in range(1, num_runways + 1)}
        runway_last_id = {r: None for r in range(1, num_runways + 1)}
        
        for i in indices:
            best_runway = -1
            best_time = -1
            min_local_penalty = float('inf')
            
            # Try all runways
            possible_placements = []
            for r in range(1, num_runways + 1):
                earliest = instance['planes'][i-1]['earliest']
                target = instance['planes'][i-1]['target']
                latest = instance['planes'][i-1]['latest']
                
                # Separation constraint
                prev_id = runway_last_id[r]
                gap = instance['separation'][prev_id-1][i-1] if prev_id else 0
                ready_time = runway_last_landing[r] + gap
                
                start = max(earliest, ready_time)
                if start <= latest:
                    # Pick a time close to target
                    time_val = max(start, min(target, latest))
                    penalty = abs(time_val - target)
                    possible_placements.append((r, time_val, penalty))
            
            if not possible_placements:
                return None # Infeasible
            
            # Choose from best options (GRASP style)
            possible_placements.sort(key=lambda x: x[2])
            r, t, p = possible_placements[0]
            
            schedule[i] = {"landing_time": t, "runway": r}
            runway_last_landing[r] = t
            runway_last_id[r] = i
            
        return schedule

    # Main loop
    while time.time() - start_time < time_limit_s * 0.7:
        candidate = construct_randomized()
        if candidate:
            feasible, _ = tools['is_feasible'](candidate)
            if feasible:
                penalty = calculate_penalty(candidate)
                if penalty < best_penalty:
                    best_penalty = penalty
                    best_schedule = candidate
        
        if time.time() - start_time > time_limit_s * 0.8:
            break

    # Refine with local search if possible
    if best_schedule:
        # Use provided tool for local search
        improved = tools['apply_swap_landings'](best_schedule, time_limit_s * 0.1)
        if improved:
            best_schedule = improved
    else:
        # Fallback to simple greedy constructor if nothing found
        best_schedule = tools['greedy_target_time_construct']()

    return {"schedule": best_schedule}