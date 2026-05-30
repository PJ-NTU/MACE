# MACE evolved heuristic 01/10 for problem: open_shop_scheduling
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Solves the Open Shop Scheduling problem using a combination of a greedy 
    LPT-based construction and a randomized local search (Hill Climbing).
    """
    start_time = time.time()
    n_jobs = instance['n_jobs']
    n_machines = instance['n_machines']
    
    # Use the provided LPT construction as a strong initial feasible solution
    best_starts = tools['lpt_dense_construct']()
    best_makespan = tools['simulate_makespan_from_starts'](best_starts)
    
    # We will refine the schedule using a priority-list-based local search.
    # The priority list determines the order in which operations are scheduled
    # by the greedy list scheduler.
    
    def get_priority_list(start_times):
        """Extracts a priority list of (job, machine) based on start times."""
        ops = []
        for i in range(n_jobs):
            for j in range(n_machines):
                ops.append((start_times[i][j], i, instance['machines'][i][j]))
        ops.sort()
        return [(job, m_id) for t, job, m_id in ops]

    # Current priority list
    current_priorities = get_priority_list(best_starts)
    
    # Iterative improvement loop
    while time.time() - start_time < time_limit_s * 0.85:
        # Create a neighbor by swapping two random operations in the priority list
        if len(current_priorities) < 2:
            break
            
        new_priorities = list(current_priorities)
        idx1, idx2 = random.sample(range(len(new_priorities)), 2)
        new_priorities[idx1], new_priorities[idx2] = new_priorities[idx2], new_priorities[idx1]
        
        # Build new schedule from priority list
        candidate_starts = tools['greedy_list_schedule'](priorities=new_priorities)
        candidate_makespan = tools['simulate_makespan_from_starts'](candidate_starts)
        
        # Keep if better
        if candidate_makespan < best_makespan:
            best_makespan = candidate_makespan
            best_starts = candidate_starts
            current_priorities = new_priorities
        
        # Periodic check for time budget
        if random.random() < 0.01: # Check time every 100 iterations on average
            if time.time() - start_time > time_limit_s * 0.95:
                break
                
    # Final cleanup: apply the provided local swap tool if time permits
    # as it is highly optimized for this specific problem structure.
    try:
        refined_starts = tools['apply_local_swap'](best_starts, time_limit_s=max(0.1, time_limit_s * 0.05))
        best_starts = refined_starts
    except:
        pass

    return {"start_times": best_starts}