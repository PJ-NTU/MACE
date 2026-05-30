# MACE evolved heuristic 05/10 for problem: open_shop_scheduling
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Dispatcher-style heuristic for Open Shop Scheduling.
    
    Hypothesis: 
    - Small instances (n_jobs * n_machines <= 64) benefit significantly from 
      the exact ILP solver provided in tools, as the search space is small 
      enough to reach optimality.
    - Larger/Dense instances benefit from the randomized priority-list 
      perturbation (ILS) strategy, which explores the sequence space more 
      effectively than simple random swaps when the search space is too large 
      for exact methods.
    """
    start_time = time.time()
    n_jobs = instance['n_jobs']
    n_machines = instance['n_machines']
    total_ops = n_jobs * n_machines
    
    # Heuristic: Dispatch to ILP-based strategy for small instances,
    # or block-shuffle ILS for larger ones.
    is_small = total_ops <= 64

    def get_priority_list(starts):
        ops = []
        for j in range(n_jobs):
            for m_idx in range(n_machines):
                m_id = instance['machines'][j][m_idx]
                ops.append((starts[j][m_idx], j, m_id))
        ops.sort()
        return [(op[1], op[2]) for op in ops]

    # Strategy 1: ILP-heavy (Best for small/medium instances)
    if is_small:
        best_starts = tools['ilp_open_shop'](time_limit_s=min(10.0, time_limit_s * 0.4))
        if best_starts is None:
            best_starts = tools['lpt_dense_construct']()
    else:
        # Strategy 2: ILS with Block Shuffling (Best for large instances)
        best_starts = tools['lpt_dense_construct']()
        
    best_makespan = tools['simulate_makespan_from_starts'](best_starts)
    current_priorities = get_priority_list(best_starts)
    
    # Iterate with budget monitoring
    while time.time() - start_time < time_limit_s * 0.85:
        new_priorities = list(current_priorities)
        
        # Perturbation: Block shuffle
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
            best_starts = candidate_starts
            current_priorities = new_priorities
            
    # Final refinement: use optimized local swap tool
    remaining = time_limit_s - (time.time() - start_time)
    if remaining > 0.1:
        try:
            refined_starts = tools['apply_local_swap'](best_starts, time_limit_s=remaining * 0.9)
            if tools['simulate_makespan_from_starts'](refined_starts) < best_makespan:
                best_starts = refined_starts
        except Exception:
            pass
            
    return {"start_times": best_starts}