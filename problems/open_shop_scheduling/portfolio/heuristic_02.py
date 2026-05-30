# MACE evolved heuristic 02/10 for problem: open_shop_scheduling
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Solves the Open Shop Scheduling problem using a hybrid approach:
    1. Warm start using LPT-dense construction.
    2. Simulated Annealing on the priority list representation of the schedule.
    """
    start_time_budget = time.time()
    
    n_jobs = instance['n_jobs']
    n_machines = instance['n_machines']
    
    # Initial solution: LPT-dense construction
    best_start_times = tools['lpt_dense_construct']()
    best_makespan = tools['simulate_makespan_from_starts'](best_start_times)
    
    current_start_times = [row[:] for row in best_start_times]
    current_makespan = best_makespan
    
    # SA Parameters
    temp = 100.0
    cooling_rate = 0.995
    
    # Helper to convert start_times to a priority list representation
    # The priority list order determines the greedy sequence.
    # The error was caused by the logic of mapping (job, machine_index) 
    # where machine_index in the instance is 1-indexed (machine id).
    # We must construct a valid list of (job_id, machine_id).
    def get_priority_list(starts):
        ops = []
        for j in range(n_jobs):
            for m_idx in range(n_machines):
                # instance['machines'][j][m_idx] is the machine ID
                machine_id = instance['machines'][j][m_idx]
                ops.append((starts[j][m_idx], j, machine_id))
        ops.sort()
        return [(op[1], op[2]) for op in ops]

    current_priorities = get_priority_list(current_start_times)
    
    # Iterative refinement
    while time.time() - start_time_budget < time_limit_s * 0.9:
        # Neighbor: swap two random operations in the priority list
        new_priorities = list(current_priorities)
        idx1, idx2 = random.sample(range(len(new_priorities)), 2)
        new_priorities[idx1], new_priorities[idx2] = new_priorities[idx2], new_priorities[idx1]
        
        # Evaluate
        candidate_starts = tools['greedy_list_schedule'](priorities=new_priorities)
        candidate_makespan = tools['simulate_makespan_from_starts'](candidate_starts)
        
        # Acceptance criteria
        delta = candidate_makespan - current_makespan
        if delta < 0 or (temp > 0 and random.random() < (2.71828 ** (-delta / temp))):
            current_start_times = candidate_starts
            current_priorities = new_priorities
            current_makespan = candidate_makespan
            
            if current_makespan < best_makespan:
                best_makespan = current_makespan
                best_start_times = [row[:] for row in current_start_times]
        
        # Cool down
        temp *= cooling_rate
        if temp < 0.1:
            temp = 100.0 # Reset temperature for local search exploration
            
    return {"start_times": best_start_times}