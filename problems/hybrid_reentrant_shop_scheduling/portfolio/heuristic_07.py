# MACE evolved heuristic 07/10 for problem: hybrid_reentrant_shop_scheduling
import time
import random
import heapq

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    A robust HRSS solver combining priority-rule construction, 
    first-improvement local search, and time-aware simulated annealing.
    """
    start_time = time.time()
    n_jobs = instance['n_jobs']
    n_machines = instance['n_machines']

    # 1. Deterministic Batch Assignment
    # Per problem constraints and eval_func, machines are assigned via list scheduling.
    batch_assignment = [0] * n_jobs
    machine_heap = [(0, i + 1) for i in range(n_machines)]
    init_time = instance['init_time']
    for j in range(n_jobs):
        finish_time, m_id = heapq.heappop(machine_heap)
        batch_assignment[j] = m_id
        heapq.heappush(machine_heap, (finish_time + init_time, m_id))

    # 2. Construction: Generate initial population with diverse priority rules
    rules = ['natural', 'spt_setup', 'lpt_setup', 'spt_main', 'lpt_main', 'spt_total', 'lpt_total', 'erd']
    best_perm = list(range(1, n_jobs + 1))
    best_makespan = float('inf')

    for rule in rules:
        try:
            perm = tools['list_scheduling_priority'](rule)
            sim = tools['simulate_schedule'](perm)
            if sim['makespan'] < best_makespan:
                best_makespan = sim['makespan']
                best_perm = list(perm)
        except Exception:
            continue

    # 3. Refinement: Local Search + Simulated Annealing Hybrid
    # Use 'apply_local_swap' for efficient first-improvement, then explore
    # using a temperature-based restart/perturbation strategy.
    
    current_perm = list(best_perm)
    
    def get_makespan(perm):
        return tools['simulate_schedule'](perm)['makespan']

    # Perform thorough search using the provided tool
    current_perm = tools['apply_local_swap'](
        current_perm, 
        time_limit_s=time_limit_s * 0.5, 
        neighbourhood='insert'
    )
    best_makespan = get_makespan(current_perm)

    # Simulated Annealing / Perturbation phase
    # Allows escaping local optima by accepting worse moves with decreasing probability
    temp = 100.0
    cooling_rate = 0.995
    while time.time() - start_time < time_limit_s * 0.95:
        # Perturb: perform a small swap
        i, j = random.sample(range(n_jobs), 2)
        new_perm = list(current_perm)
        new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
        
        try:
            new_makespan = get_makespan(new_perm)
            delta = new_makespan - best_makespan
            
            if delta < 0 or (temp > 0 and random.random() < (math.exp(-delta / temp))):
                current_perm = new_perm
                if new_makespan < best_makespan:
                    best_makespan = new_makespan
                    best_perm = list(new_perm)
            
            temp *= cooling_rate
        except Exception:
            continue

    return {
        "permutation": best_perm,
        "batch_assignment": batch_assignment
    }