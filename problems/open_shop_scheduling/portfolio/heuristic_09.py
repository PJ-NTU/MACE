# MACE evolved heuristic 09/10 for problem: open_shop_scheduling
import time
import random
import numpy as np

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Dispatcher-style solver for Open Shop Scheduling.
    
    Dispatch Logic:
    - If the problem is small (n_jobs * n_machines <= 50), the search space is 
      manageable for local search intensification. We use a hybrid approach 
      combining ILP (for backbone) and local swapping.
    - If the problem is large (n_jobs * n_machines > 50), the search space 
      requires robust exploration. We use a population-based greedy list-scheduling 
      approach to maintain diversity and optimize the makespan.
    """
    start_time = time.time()
    n_jobs = instance['n_jobs']
    n_machines = instance['n_machines']
    total_ops = n_jobs * n_machines
    
    # Heuristic: Small instances benefit from exact solver + local search,
    # while large instances benefit from population-based metaheuristics.
    if total_ops <= 50:
        return _solve_small(instance, tools, time_limit_s)
    else:
        return _solve_large(instance, tools, time_limit_s)

def _solve_small(instance, tools, time_limit_s):
    start_time = time.time()
    # Attempt ILP for optimal foundation if time allows
    best_starts = tools['ilp_open_shop'](time_limit_s=min(5.0, time_limit_s * 0.3))
    
    if best_starts is None:
        best_starts = tools['lpt_dense_construct']()
        
    # Refine with local swap
    remaining = time_limit_s - (time.time() - start_time)
    if remaining > 0.1:
        best_starts = tools['apply_local_swap'](best_starts, time_limit_s=min(remaining, 10.0))
        
    return {"start_times": best_starts}

def _solve_large(instance, tools, time_limit_s):
    start_time = time.time()
    n_jobs = instance['n_jobs']
    n_machines = instance['n_machines']
    
    # Initialize population with diverse greedy strategies
    population = []
    # 1. LPT-based
    population.append(tools['lpt_dense_construct']())
    # 2. Random priority list scheduling
    for _ in range(4):
        ops = [(j, instance['machines'][j][m]) for j in range(n_jobs) for m in range(n_machines)]
        random.shuffle(ops)
        population.append(tools['greedy_list_schedule'](priorities=ops))
        
    def get_flat_priority(starts):
        ops = []
        for j in range(n_jobs):
            for m_idx in range(n_machines):
                ops.append((starts[j][m_idx], j, instance['machines'][j][m_idx]))
        ops.sort()
        return [(op[1], op[2]) for op in ops]

    best_starts = min(population, key=lambda s: tools['simulate_makespan_from_starts'](s))
    best_makespan = tools['simulate_makespan_from_starts'](best_starts)
    
    # Evolutionary loop
    while time.time() - start_time < time_limit_s * 0.9:
        p1, p2 = random.sample(population, 2)
        pri1, pri2 = get_flat_priority(p1), get_flat_priority(p2)
        
        # Crossover
        cut = random.randint(1, len(pri1) - 1)
        offspring_pri = pri1[:cut]
        seen = set(offspring_pri)
        for op in pri2:
            if op not in seen: 
                offspring_pri.append(op)
        
        # Mutation
        if random.random() < 0.4:
            idx = random.randint(0, len(offspring_pri) - 1)
            el = offspring_pri.pop(idx)
            offspring_pri.insert(random.randint(0, len(offspring_pri)), el)
            
        candidate = tools['greedy_list_schedule'](priorities=offspring_pri)
        c_makespan = tools['simulate_makespan_from_starts'](candidate)
        
        # Replacement strategy
        population.sort(key=lambda s: tools['simulate_makespan_from_starts'](s))
        if c_makespan < tools['simulate_makespan_from_starts'](population[-1]):
            population[-1] = candidate
            
        if c_makespan < best_makespan:
            best_makespan = c_makespan
            best_starts = candidate

    return {"start_times": best_starts}