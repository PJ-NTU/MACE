# MACE evolved heuristic 06/10 for problem: open_shop_scheduling
import time
import random
import numpy as np

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    A 'Tabu-Search inspired' Variable Neighborhood Descent (VND) solver.
    
    1. Departure from Portfolio:
       - Most portfolio members use simple list-priority swaps or GA.
       - This uses a multi-neighborhood approach (VND) with a 'Tabu' mechanism 
         to prevent cycling, rather than just random restarts or SA.
       - Instead of just 'swapping', it uses 'insert' operations (moving an 
         operation to a different absolute position) which is more disruptive 
         than pairwise swaps.
       - Employs a 'greedy-descent with memory' rather than population-based 
         or pure stochastic hill-climbing.
    """
    start_time = time.time()
    n_jobs = instance['n_jobs']
    n_machines = instance['n_machines']
    
    # Core strategy: Start with LPT, then perform VND on the priority list.
    # Neighborhoods: 
    # N1: Shift-insert (move element to new pos)
    # N2: Block-swap (exchange two blocks)
    
    current_starts = tools['lpt_dense_construct']()
    best_starts = current_starts
    best_makespan = tools['simulate_makespan_from_starts'](best_starts)
    
    def get_priority_list(starts):
        ops = []
        for i in range(n_jobs):
            for j in range(n_machines):
                ops.append((starts[i][j], i, instance['machines'][i][j]))
        ops.sort()
        return [(job, m_id) for t, job, m_id in ops]

    current_priorities = get_priority_list(current_starts)
    tabu_list = {} # Store recent priority lists to avoid cycles
    
    def get_neighbor(priorities, neighborhood_type):
        new_p = list(priorities)
        if neighborhood_type == 0: # Shift Insert
            idx = random.randint(0, len(new_p) - 1)
            el = new_p.pop(idx)
            new_p.insert(random.randint(0, len(new_p)), el)
        else: # Block Swap
            b1 = random.randint(0, len(new_p) - 3)
            b2 = random.randint(b1 + 2, len(new_p) - 1)
            new_p[b1], new_p[b2] = new_p[b2], new_p[b1]
        return new_p

    neighborhood_index = 0
    while time.time() - start_time < time_limit_s * 0.9:
        # Generate neighbor in current neighborhood
        candidate_priorities = get_neighbor(current_priorities, neighborhood_index)
        
        # Tabu check
        if tuple(candidate_priorities) in tabu_list:
            neighborhood_index = (neighborhood_index + 1) % 2
            continue
            
        candidate_starts = tools['greedy_list_schedule'](priorities=candidate_priorities)
        c_makespan = tools['simulate_makespan_from_starts'](candidate_starts)
        
        if c_makespan < best_makespan:
            best_makespan = c_makespan
            best_starts = candidate_starts
            current_priorities = candidate_priorities
            tabu_list[tuple(candidate_priorities)] = True
            neighborhood_index = 0 # Reset neighborhood
        else:
            neighborhood_index = (neighborhood_index + 1) % 2
            
        # Periodic cleanup of tabu
        if len(tabu_list) > 100:
            tabu_list.clear()
            
    # Final polish with tool
    remaining = time_limit_s - (time.time() - start_time)
    if remaining > 0.1:
        try:
            refined = tools['apply_local_swap'](best_starts, time_limit_s=remaining * 0.9)
            return {"start_times": refined}
        except:
            return {"start_times": best_starts}
            
    return {"start_times": best_starts}