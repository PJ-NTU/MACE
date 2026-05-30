# MACE evolved heuristic 08/10 for problem: open_shop_scheduling
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Optimized heuristic for Open Shop Scheduling.
    
    Strategy:
    1. Warm start: Prioritize ILP for small instances and LPT-dense for larger ones.
    2. Adaptive perturbation: Perform a randomized neighborhood search using 
       greedy list scheduling, focusing on priority-list perturbations.
    3. Final intensification: Use the highly optimized 'apply_local_swap' 
       on the best-found sequence.
    """
    start_time = time.time()
    n_jobs = instance['n_jobs']
    n_machines = instance['n_machines']
    
    # Track best solution
    best_start_times = None
    best_makespan = float('inf')

    def get_makespan(starts):
        val = tools['simulate_makespan_from_starts'](starts)
        return val if val is not None else float('inf')

    # 1. Initialization
    # Use ILP for small instances, fallback to LPT dense
    if n_jobs * n_machines <= 64:
        best_start_times = tools['ilp_open_shop'](time_limit_s=min(5.0, time_limit_s * 0.3))
    
    if best_start_times is None:
        best_start_times = tools['lpt_dense_construct']()
    
    best_makespan = get_makespan(best_start_times)

    # 2. Iterated Local Search (ILS) with Randomized Priority List Perturbation
    # We maintain a priority list representation to explore the space efficiently
    def get_priority_list(starts):
        ops = []
        for j in range(n_jobs):
            for m_idx in range(n_machines):
                m_id = instance['machines'][j][m_idx]
                ops.append((starts[j][m_idx], j, m_id))
        ops.sort()
        return [(op[1], op[2]) for op in ops]

    current_priorities = get_priority_list(best_start_times)
    
    # Allocate time for ILS loop
    ils_end = start_time + (time_limit_s * 0.7)
    
    while time.time() < ils_end:
        # Perturb: Randomly swap a small range or shift a block
        new_priorities = list(current_priorities)
        if random.random() < 0.5:
            # Block shuffle
            b_sz = random.randint(2, max(3, n_jobs))
            start_idx = random.randint(0, len(new_priorities) - b_sz)
            block = new_priorities[start_idx : start_idx + b_sz]
            del new_priorities[start_idx : start_idx + b_sz]
            pos = random.randint(0, len(new_priorities))
            new_priorities[pos:pos] = block
        else:
            # Pair swap
            idx1, idx2 = random.sample(range(len(new_priorities)), 2)
            new_priorities[idx1], new_priorities[idx2] = new_priorities[idx2], new_priorities[idx1]
            
        cand_starts = tools['greedy_list_schedule'](priorities=new_priorities)
        cand_makespan = get_makespan(cand_starts)
        
        if cand_makespan < best_makespan:
            best_makespan = cand_makespan
            best_start_times = cand_starts
            current_priorities = new_priorities
            
    # 3. Final Intensification
    # Use the optimized local swap tool on the best result found
    remaining = time_limit_s - (time.time() - start_time)
    if remaining > 0.1:
        try:
            refined_starts = tools['apply_local_swap'](best_start_times, time_limit_s=min(remaining, 5.0))
            if get_makespan(refined_starts) < best_makespan:
                best_start_times = refined_starts
        except Exception:
            pass
            
    return {"start_times": best_start_times}