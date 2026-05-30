# MACE evolved heuristic 05/10 for problem: flow_shop_scheduling
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Dispatcher-style solver for Flow Shop Scheduling.
    
    Heuristic Selection Logic:
    - Small instances (n <= 20): Use ILP if available, or aggressive insertion search.
    - Medium-Large instances (n > 20):
        - High m (m >= 10): Likely bottleneck-constrained; prefer robust, 
          multi-start insertion search (Parent A style).
        - Low m (m < 10): Likely sequence-constrained; prefer block-perturbation 
          ILS (Parent B style) to explore the search space more effectively.
    """
    start_time = time.time()
    n = instance['n']
    m = instance['m']
    
    # 1. Exact solver for small scale
    if n <= 15:
        res = tools['ilp_flow_shop'](time_limit_s=time_limit_s * 0.8)
        if res:
            return tools['make_solution'](res)
    
    # 2. Heuristic dispatch
    # Parent A logic: NEH + Iterative Insertion Search (Good for general cases)
    # Parent B logic: NEH + Block Perturbation ILS (Better at escaping local optima)
    
    if m >= 10:
        # Parent A Strategy
        current_perm = tools['neh_construct']()
        refinement_limit = time_limit_s * 0.95
        best_perm = list(current_perm)
        best_makespan = tools['simulate_makespan'](best_perm)
        
        while time.time() - start_time < refinement_limit:
            remaining = refinement_limit - (time.time() - start_time)
            if remaining < 0.05: break
            
            improved_perm = tools['apply_insertion_search'](
                list(current_perm), 
                time_limit_s=remaining, 
                first_improvement=True
            )
            
            if tools['simulate_makespan'](improved_perm) < best_makespan:
                best_perm = list(improved_perm)
                best_makespan = tools['simulate_makespan'](best_perm)
                current_perm = list(best_perm)
            else:
                # Random swap to diversify
                current_perm = list(best_perm)
                idx1, idx2 = random.sample(range(n), 2)
                current_perm[idx1], current_perm[idx2] = current_perm[idx2], current_perm[idx1]
        return tools['make_solution'](best_perm)
        
    else:
        # Parent B Strategy (Block Perturbation)
        current_perm = tools['neh_construct']()
        current_perm = tools['apply_insertion_search'](current_perm, time_limit_s=time_limit_s * 0.2)
        
        while time.time() - start_time < time_limit_s * 0.9:
            perturbed = list(current_perm)
            if n > 3:
                idx = random.randint(0, n - 3)
                block = [perturbed.pop(idx), perturbed.pop(idx)]
                pos = random.randint(0, n - 2)
                perturbed.insert(pos, block[0])
                perturbed.insert(pos + 1, block[1])
            
            remaining = time_limit_s - (time.time() - start_time)
            if remaining < 0.05: break
            
            candidate = tools['apply_insertion_search'](perturbed, time_limit_s=remaining, first_improvement=True)
            if tools['simulate_makespan'](candidate) < tools['simulate_makespan'](current_perm):
                current_perm = candidate
                
        return tools['make_solution'](current_perm)