# MACE evolved heuristic 08/10 for problem: aircraft_landing
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Improved solver:
    1. Uses MILP for small/medium instances.
    2. For larger instances, uses a Simulated Annealing approach on the 
       runway assignment and sequence, using the tools' re-packing logic.
    """
    start_time = time.time()
    time_limit_buffer = time_limit_s * 0.90
    
    num_planes = instance["num_planes"]
    num_runways = instance["num_runways"]

    # 1. Try MILP if it's within a reasonable size
    if num_planes <= 35:
        sol = tools.get('ilp_aircraft_landing', lambda time_limit_s: None)(
            time_limit_s=min(time_limit_s * 0.6, 15.0)
        )
        if sol is not None:
            return {"schedule": sol}

    # 2. Heuristic: Simulated Annealing on runway assignments
    # Initial state: Greedy construction
    best_schedule = None
    best_penalty = float('inf')

    # Try diverse seedings
    seeds = [tools['greedy_target_time_construct'](), tools['construct_by_appearance_order']()]
    for s in seeds:
        if s:
            feas, _ = tools['validate_schedule'](s)
            if feas:
                p = tools['penalty_of_schedule'](s)
                if p < best_penalty:
                    best_penalty = p
                    best_schedule = s

    # If no initial solution, try random assignments until one is found
    if best_schedule is None:
        for _ in range(500):
            if time.time() - start_time > time_limit_buffer * 0.2: break
            rand_assignments = {i: random.randint(1, num_runways) for i in range(1, num_planes + 1)}
            # We use apply_reassign_runway logic implicitly via a custom packer
            # Or just use tools if available. Here we use the provided tools.
            # Since we need a valid schedule, we attempt to use reassign_runway.
            pass # Fallback logic handled below

    # Current state
    curr_schedule = best_schedule
    curr_penalty = best_penalty

    # Simulated Annealing
    temp = 1000.0
    cooling_rate = 0.995
    
    while time.time() - start_time < time_limit_buffer:
        if curr_schedule is None:
            # Re-seed if stuck
            curr_schedule = tools['greedy_target_time_construct']()
            curr_penalty = tools['penalty_of_schedule'](curr_schedule) if curr_schedule else float('inf')
            continue

        # Mutation: Reassign a random plane to a random runway
        p_idx = random.randint(1, num_planes)
        r_idx = random.randint(1, num_runways)
        
        neighbor = tools['apply_reassign_runway'](curr_schedule, p_idx, r_idx)
        
        # Check if changed and feasible
        feas, _ = tools['validate_schedule'](neighbor)
        if feas:
            new_penalty = tools['penalty_of_schedule'](neighbor)
            delta = new_penalty - curr_penalty
            
            if delta < 0 or random.random() < math.exp(-delta / temp):
                curr_schedule = neighbor
                curr_penalty = new_penalty
                
                if curr_penalty < best_penalty:
                    best_penalty = curr_penalty
                    best_schedule = curr_schedule
        
        temp *= cooling_rate
        if temp < 0.1: temp = 0.1

    return {"schedule": best_schedule}