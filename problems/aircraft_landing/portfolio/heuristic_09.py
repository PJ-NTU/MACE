# MACE evolved heuristic 09/10 for problem: aircraft_landing
import time
import random
import heapq

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    start_time = time.time()
    # Use a safe margin for termination
    deadline = start_time + time_limit_s * 0.85
    
    num_planes = instance['num_planes']
    num_runways = instance['num_runways']
    
    def get_penalty(s):
        feas, _ = tools['validate_schedule'](s)
        return tools['penalty_of_schedule'](s) if feas else float('inf')

    # Initial construction
    best_schedule = tools['greedy_target_time_construct']()
    if not best_schedule or not tools['validate_schedule'](best_schedule)[0]:
        best_schedule = tools['construct_by_appearance_order']()
    
    best_penalty = get_penalty(best_schedule)
    current_schedule = dict(best_schedule)

    # Iterative Improvement with time-limited budget
    iterations = 0
    max_iters = 200  # Cap iterations to prevent TLE
    
    while time.time() < deadline and iterations < max_iters:
        iterations += 1
        improved = False
        
        # 1. Try Swap (more global)
        if time.time() < deadline:
            neighbor = tools['apply_swap_landings'](current_schedule, t_limit=0.02)
            p_val = get_penalty(neighbor)
            if p_val < best_penalty:
                best_penalty = p_val
                best_schedule = dict(neighbor)
                current_schedule = dict(neighbor)
                improved = True
        
        # 2. Try Reassign (more targeted)
        if not improved and time.time() < deadline:
            p_idx = random.randint(1, num_planes)
            r_idx = random.randint(1, num_runways)
            neighbor = tools['apply_reassign_runway'](current_schedule, p_idx, r_idx)
            p_val = get_penalty(neighbor)
            if p_val < best_penalty:
                best_penalty = p_val
                best_schedule = dict(neighbor)
                current_schedule = dict(neighbor)
                improved = True
        
        # 3. Early exit if no improvement for a few iterations
        if not improved and iterations > 50:
            break
            
        # 4. Occasional random perturbation
        if not improved and iterations % 20 == 0:
            p_idx = random.randint(1, num_planes)
            r_idx = random.randint(1, num_runways)
            current_schedule = tools['apply_reassign_runway'](current_schedule, p_idx, r_idx)

    return {"schedule": best_schedule}