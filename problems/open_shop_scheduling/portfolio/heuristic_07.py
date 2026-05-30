# MACE evolved heuristic 07/10 for problem: open_shop_scheduling
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Improved heuristic for Open Shop Scheduling:
    Replaces the simple block shuffle with a weighted-probability mutation 
    strategy that favors disrupting operations with longer processing times 
    in the current schedule to better escape local optima.
    """
    start_time = time.time()
    n_jobs = instance['n_jobs']
    n_machines = instance['n_machines']
    total_ops = n_jobs * n_machines
    
    is_small = total_ops <= 64

    def get_priority_list(starts):
        ops = []
        for j in range(n_jobs):
            for m_idx in range(n_machines):
                m_id = instance['machines'][j][m_idx]
                ops.append((starts[j][m_idx], j, m_id))
        ops.sort()
        return [(op[1], op[2]) for op in ops]

    # Strategy: ILP-heavy for small, LPT-dense for large
    if is_small:
        best_starts = tools['ilp_open_shop'](time_limit_s=min(10.0, time_limit_s * 0.4))
        if best_starts is None:
            best_starts = tools['lpt_dense_construct']()
    else:
        best_starts = tools['lpt_dense_construct']()
        
    best_makespan = tools['simulate_makespan_from_starts'](best_starts)
    current_priorities = get_priority_list(best_starts)
    
    # Calculate weights based on processing times for weighted mutation
    # Higher processing time ops are more likely to be moved
    proc_times = [instance['times'][j][m_idx] for j in range(n_jobs) for m_idx in range(n_machines)]
    avg_pt = sum(proc_times) / len(proc_times)
    weights = [max(1.0, instance['times'][j][m_idx] / avg_pt) for j in range(n_jobs) for m_idx in range(n_machines)]
    
    while time.time() - start_time < time_limit_s * 0.85:
        new_priorities = list(current_priorities)
        
        # Weighted mutation: choose an index to move based on op processing time impact
        idx = random.choices(range(len(new_priorities)), weights=weights, k=1)[0]
        op = new_priorities.pop(idx)
        insert_pos = random.randint(0, len(new_priorities))
        new_priorities.insert(insert_pos, op)
        
        candidate_starts = tools['greedy_list_schedule'](priorities=new_priorities)
        candidate_makespan = tools['simulate_makespan_from_starts'](candidate_starts)
        
        if candidate_makespan < best_makespan:
            best_makespan = candidate_makespan
            best_starts = candidate_starts
            current_priorities = new_priorities
            
    # Final refinement
    remaining = time_limit_s - (time.time() - start_time)
    if remaining > 0.1:
        try:
            refined_starts = tools['apply_local_swap'](best_starts, time_limit_s=remaining * 0.9)
            if tools['simulate_makespan_from_starts'](refined_starts) < best_makespan:
                best_starts = refined_starts
        except Exception:
            pass
            
    return {"start_times": best_starts}