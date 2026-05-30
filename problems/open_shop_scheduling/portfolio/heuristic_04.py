# MACE evolved heuristic 04/10 for problem: open_shop_scheduling
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Optimized Open Shop Scheduler:
    1. Uses ILP for a high-quality baseline if time permits.
    2. Employs a 'Destroy and Repair' metaheuristic using segment shuffling 
       in the priority list space to escape local optima effectively.
    3. Concludes with a focused local search refinement using the provided 
       apply_local_swap tool, which is better at fine-tuning than stochastic SA.
    """
    start_time_budget = time.time()
    n_jobs = instance['n_jobs']
    n_machines = instance['n_machines']

    # 1. Initialization
    # Prioritize ILP for small/medium instances; fallback to LPT construction.
    best_start_times = None
    if n_jobs * n_machines <= 100:
        best_start_times = tools['ilp_open_shop'](time_limit_s=min(5.0, time_limit_s * 0.2))
    
    if best_start_times is None:
        best_start_times = tools['lpt_dense_construct']()
        
    best_makespan = tools['simulate_makespan_from_starts'](best_start_times)
    
    # Helper to derive priority list from a feasible schedule
    def get_priority_list(starts):
        ops = []
        for j in range(n_jobs):
            for m_idx in range(n_machines):
                m_id = instance['machines'][j][m_idx]
                ops.append((starts[j][m_idx], j, m_id))
        ops.sort()
        return [(op[1], op[2]) for op in ops]

    current_priorities = get_priority_list(best_start_times)
    
    # 2. Iterated Local Search (Destroy and Repair)
    # Block shuffling preserves structural coherence better than random swaps.
    while time.time() - start_time_budget < time_limit_s * 0.75:
        new_priorities = list(current_priorities)
        block_size = random.randint(2, max(3, n_jobs))
        start_idx = random.randint(0, len(new_priorities) - block_size)
        
        block = new_priorities[start_idx : start_idx + block_size]
        del new_priorities[start_idx : start_idx + block_size]
        insert_pos = random.randint(0, len(new_priorities))
        new_priorities[insert_pos:insert_pos] = block
        
        candidate_starts = tools['greedy_list_schedule'](priorities=new_priorities)
        candidate_makespan = tools['simulate_makespan_from_starts'](candidate_starts)
        
        if candidate_makespan < best_makespan:
            best_makespan = candidate_makespan
            best_start_times = candidate_starts
            current_priorities = new_priorities
            
    # 3. Final Polish
    # Devote the remaining time to first-improvement local search, 
    # which is computationally expensive but highly effective for convergence.
    remaining = time_limit_s - (time.time() - start_time_budget)
    if remaining > 0.2:
        try:
            refined_starts = tools['apply_local_swap'](best_start_times, time_limit_s=remaining * 0.9)
            refined_makespan = tools['simulate_makespan_from_starts'](refined_starts)
            if refined_makespan < best_makespan:
                best_start_times = refined_starts
        except Exception:
            pass # Fallback to best_start_times if refinement fails
            
    return {"start_times": best_start_times}