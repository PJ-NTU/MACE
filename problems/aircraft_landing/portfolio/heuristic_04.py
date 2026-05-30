# MACE evolved heuristic 04/10 for problem: aircraft_landing
import time
import random
import math
import numpy as np

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    A Tabu Search metaheuristic for the Aircraft Landing Problem.
    
    Departure from portfolio:
    1. Uses a forbidden list (Tabu list) to prevent cycling in local search, 
       unlike the portfolio's simple hill-climbing or greedy approaches.
    2. Uses a 'Move' based neighborhood exploration (runway reassignment) 
       rather than just global swap_landings.
    3. Explicitly prioritizes memory-based diversification to avoid the 
       stagnation of greedy restarts.
    """
    start_time = time.time()
    num_planes = instance['num_planes']
    num_runways = instance['num_runways']
    
    # 1. Initial State: Start from a decent greedy solution
    best_schedule = tools['greedy_target_time_construct']()
    if not best_schedule:
        best_schedule = tools['construct_by_appearance_order']()
    
    current_schedule = dict(best_schedule)
    best_penalty = tools['penalty_of_schedule'](current_schedule)
    
    # Tabu list: stores (plane_id, runway_id) as forbidden moves for a duration
    tabu_list = {}
    tabu_tenure = 5 
    
    def get_penalty(sched):
        feasible, _ = tools['validate_schedule'](sched)
        if not feasible:
            return float('inf')
        return tools['penalty_of_schedule'](sched)

    # 2. Main Search Loop
    iteration = 0
    while time.time() - start_time < time_limit_s * 0.9:
        iteration += 1
        best_move = None
        best_move_penalty = float('inf')
        
        # Explore neighborhood: Reassign a random plane to a different runway
        # This is distinct from the portfolio's focus on swap_landings
        for _ in range(min(num_planes, 15)):
            plane_id = random.randint(1, num_planes)
            target_runway = random.randint(1, num_runways)
            
            # Check Tabu status
            if tabu_list.get((plane_id, target_runway), 0) > iteration:
                continue
                
            candidate = tools['apply_reassign_runway'](current_schedule, plane_id, target_runway)
            
            # evaluate
            if candidate != current_schedule:
                p = get_penalty(candidate)
                if p < best_move_penalty:
                    best_move_penalty = p
                    best_move = (candidate, plane_id, target_runway)
        
        # Apply best non-tabu move
        if best_move:
            candidate, p_id, r_id = best_move
            current_schedule = candidate
            tabu_list[(p_id, r_id)] = iteration + tabu_tenure
            
            if best_move_penalty < best_penalty:
                best_penalty = best_move_penalty
                best_schedule = dict(candidate)
        else:
            # Diversification: Random jump if stuck
            p_id = random.randint(1, num_planes)
            r_id = random.randint(1, num_runways)
            current_schedule = tools['apply_reassign_runway'](current_schedule, p_id, r_id)
            
    return {"schedule": best_schedule}