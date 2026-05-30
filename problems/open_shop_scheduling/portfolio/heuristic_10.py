# MACE evolved heuristic 10/10 for problem: open_shop_scheduling
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Hybrid heuristic for Open Shop Scheduling:
    - Uses ILP for small instances.
    - Uses a multi-start Iterated Local Search (ILS) with adaptive perturbation.
    - Combines greedy construction, block-shuffling (for structural changes),
      and targeted swaps (for local refinement).
    """
    start_time = time.time()
    n_jobs = instance['n_jobs']
    n_machines = instance['n_machines']
    
    # Initialization
    best_starts = None
    best_makespan = float('inf')

    # Initial Construction
    if n_jobs * n_machines <= 64:
        best_starts = tools['ilp_open_shop'](time_limit_s=min(5.0, time_limit_s * 0.2))
    
    if best_starts is None:
        best_starts = tools['lpt_dense_construct']()
        
    best_makespan = tools['simulate_makespan_from_starts'](best_starts)

    def to_priority_list(starts):
        ops = []
        for j in range(n_jobs):
            for m_idx in range(n_machines):
                m_id = instance['machines'][j][m_idx]
                ops.append((starts[j][m_idx], j, m_id))
        ops.sort()
        return [(op[1], op[2]) for op in ops]

    current_priorities = to_priority_list(best_starts)
    
    # Adaptive ILS Loop
    # We alternate between block shuffles (exploration) and neighborhood swaps (exploitation)
    iteration = 0
    while time.time() - start_time < time_limit_s * 0.8:
        new_priorities = list(current_priorities)
        
        # Every 10 iterations do a structural block shuffle, otherwise a swap
        if iteration % 10 == 0:
            block_size = random.randint(2, max(3, n_jobs))
            start_idx = random.randint(0, len(new_priorities) - block_size)
            block = new_priorities[start_idx : start_idx + block_size]
            del new_priorities[start_idx : start_idx + block_size]
            insert_pos = random.randint(0, len(new_priorities))
            new_priorities[insert_pos:insert_pos] = block
        else:
            idx1, idx2 = random.sample(range(len(new_priorities)), 2)
            new_priorities[idx1], new_priorities[idx2] = new_priorities[idx2], new_priorities[idx1]
            
        candidate_starts = tools['greedy_list_schedule'](priorities=new_priorities)
        candidate_makespan = tools['simulate_makespan_from_starts'](candidate_starts)
        
        if candidate_makespan < best_makespan:
            best_makespan = candidate_makespan
            best_starts = candidate_starts
            current_priorities = new_priorities
        
        iteration += 1
            
    # Final intensification
    remaining = time_limit_s - (time.time() - start_time)
    if remaining > 0.1:
        try:
            refined_starts = tools['apply_local_swap'](best_starts, time_limit_s=min(remaining, 5.0))
            refined_makespan = tools['simulate_makespan_from_starts'](refined_starts)
            if refined_makespan is not None and refined_makespan < best_makespan:
                best_starts = refined_starts
        except Exception:
            pass
            
    return {"start_times": best_starts}