# MACE evolved heuristic 10/10 for problem: flow_shop_scheduling
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Improved solver with a more aggressive diversification strategy for the
    block-perturbation phase (Parent B) to better escape local optima.
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
    if m >= 10:
        # Parent A Strategy: NEH + Iterative Insertion Search
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
                current_perm = list(best_perm)
                idx1, idx2 = random.sample(range(n), 2)
                current_perm[idx1], current_perm[idx2] = current_perm[idx2], current_perm[idx1]
        return tools['make_solution'](best_perm)
        
    else:
        # Parent B Strategy (Modified Block Perturbation)
        current_perm = tools['neh_construct']()
        current_perm = tools['apply_insertion_search'](current_perm, time_limit_s=time_limit_s * 0.2)
        
        while time.time() - start_time < time_limit_s * 0.9:
            perturbed = list(current_perm)
            # INCREASED MUTATION: Move a random sub-block of size up to 30% of n
            if n > 3:
                block_size = random.randint(2, max(2, int(n * 0.3)))
                start_idx = random.randint(0, n - block_size)
                block = [perturbed.pop(start_idx + i) for i in range(block_size)]
                pos = random.randint(0, n - block_size)
                for i in range(block_size):
                    perturbed.insert(pos + i, block[i])
            
            remaining = time_limit_s - (time.time() - start_time)
            if remaining < 0.05: break
            
            candidate = tools['apply_insertion_search'](perturbed, time_limit_s=remaining, first_improvement=True)
            if tools['simulate_makespan'](candidate) < tools['simulate_makespan'](current_perm):
                current_perm = candidate
                
        return tools['make_solution'](current_perm)