# MACE evolved heuristic 06/10 for problem: flow_shop_scheduling
import time
import random
import numpy as np

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Dispatcher Heuristic:
    - If n is small (n <= 20), use the ILP solver as it is guaranteed/near optimal.
    - If n is large and m is small (m <= 5), the problem space is narrow; use 
      the robust NEH + Insertion Search (Parent A) as it performs well on 
      narrow job-shop constraints.
    - If n is large and m is large (m > 5), the problem space is vast; use 
      a Tabu-Search inspired approach with randomized restarts to explore 
      the larger permutation space efficiently (Parent B strategy).
    """
    n = instance['n']
    m = instance['m']
    start_time = time.time()
    
    # 1. Specialized solver for small instances
    if n <= 20:
        res = tools['ilp_flow_shop'](time_limit_s=time_limit_s * 0.9)
        if res is not None:
            return tools['make_solution'](res)
    
    # 2. Heuristic for "narrow" instances (few machines, high job count)
    # NEH + Insertion Search is excellent for deeper machine pipelines
    if m <= 5:
        current_perm = tools['neh_construct']()
        best_perm = tools['apply_insertion_search'](
            current_perm, 
            time_limit_s=time_limit_s * 0.9, 
            first_improvement=True
        )
        return tools['make_solution'](best_perm)
    
    # 3. Heuristic for "wide" instances (many machines)
    # Tabu/Swap-based search is often faster at exploring large search spaces
    else:
        # Initial construction using NEH for a strong baseline
        current_perm = tools['neh_construct']()
        best_perm = list(current_perm)
        best_makespan = tools['simulate_makespan'](best_perm)
        
        tabu_list = {}
        
        while time.time() - start_time < time_limit_s * 0.9:
            # Sample neighborhood to keep O(N^2) complexity manageable
            sample_size = min(n * (n - 1) // 2, 60)
            best_neighbor = None
            best_neighbor_makespan = float('inf')
            move_to_apply = None
            
            for _ in range(sample_size):
                i, j = random.sample(range(n), 2)
                if i > j: i, j = j, i
                
                if tabu_list.get((i, j), 0) > time.time():
                    continue
                    
                neighbor = list(current_perm)
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                
                makespan = tools['simulate_makespan'](neighbor)
                
                if makespan < best_neighbor_makespan:
                    best_neighbor = neighbor
                    best_neighbor_makespan = makespan
                    move_to_apply = (i, j)
            
            if best_neighbor:
                current_perm = best_neighbor
                tabu_list[move_to_apply] = time.time() + (time_limit_s * 0.02)
                if best_neighbor_makespan < best_makespan:
                    best_makespan = best_neighbor_makespan
                    best_perm = list(best_neighbor)
            else:
                # Diversification
                idx1, idx2 = random.sample(range(n), 2)
                current_perm[idx1], current_perm[idx2] = current_perm[idx2], current_perm[idx1]
                
        return tools['make_solution'](best_perm)