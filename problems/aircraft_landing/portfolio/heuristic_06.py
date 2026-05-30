# MACE evolved heuristic 06/10 for problem: aircraft_landing
import time
import random
import numpy as np

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    A Tabu Search metaheuristic for the Aircraft Landing Problem.
    Modified: Replaces random neighborhood exploration with a weighted selection
    based on plane sensitivity (penalty_early + penalty_late), focusing 
    computational budget on planes that impact the objective most.
    """
    start_time = time.time()
    num_planes = instance['num_planes']
    num_runways = instance['num_runways']
    
    # Pre-calculate sensitivity scores to weight the neighborhood selection
    sensitivities = []
    for i in range(1, num_planes + 1):
        s = tools['plane_penalty_early'](i) + tools['plane_penalty_late'](i)
        sensitivities.append(s)
    sensitivities = np.array(sensitivities)
    # Convert to probability distribution
    probs = sensitivities / np.sum(sensitivities)
    
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
        
        # Explore neighborhood: Reassign a plane (weighted by sensitivity) to a different runway
        for _ in range(min(num_planes, 15)):
            plane_id = np.random.choice(range(1, num_planes + 1), p=probs)
            target_runway = random.randint(1, num_runways)
            
            # Check Tabu status
            if tabu_list.get((plane_id, target_runway), 0) > iteration:
                continue
                
            candidate = tools['apply_reassign_runway'](current_schedule, int(plane_id), target_runway)
            
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