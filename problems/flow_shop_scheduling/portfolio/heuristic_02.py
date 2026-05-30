# MACE evolved heuristic 02/10 for problem: flow_shop_scheduling
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Solves the flow shop scheduling problem using the NEH heuristic 
    followed by an Iterated Local Search (ILS) refinement.
    """
    start_time = time.time()
    
    # 1. Generate high-quality initial solution using NEH
    best_perm = tools['neh_construct']()
    best_makespan = tools['simulate_makespan'](best_perm)
    
    current_perm = list(best_perm)
    
    # 2. Iterated Local Search (ILS) / Variable Neighborhood Descent
    # We use insertion search as the primary improvement operator.
    # While budget allows, we perform local searches and occasional perturbation.
    
    def get_remaining():
        return time_limit_s - (time.time() - start_time)

    # First pass: Exhaustive insertion search
    improved = True
    while improved and get_remaining() > 0.1:
        prev_makespan = best_makespan
        current_perm = tools['apply_insertion_search'](
            current_perm, 
            time_limit_s=min(get_remaining(), 2.0), 
            first_improvement=True
        )
        current_makespan = tools['simulate_makespan'](current_perm)
        
        if current_makespan < best_makespan:
            best_makespan = current_makespan
            best_perm = list(current_perm)
        
        if best_makespan >= prev_makespan:
            improved = False

    # 3. Perturbation and further refinement if time permits
    # Randomly swap jobs to escape local optima
    while get_remaining() > 0.5:
        # Perturbation: swap 2 random elements
        n = instance['n']
        if n < 2:
            break
            
        p = list(best_perm)
        idx1, idx2 = random.sample(range(n), 2)
        p[idx1], p[idx2] = p[idx2], p[idx1]
        
        # Local search from perturbed state
        p = tools['apply_insertion_search'](
            p, 
            time_limit_s=min(get_remaining(), 1.0), 
            first_improvement=True
        )
        
        new_makespan = tools['simulate_makespan'](p)
        if new_makespan < best_makespan:
            best_makespan = new_makespan
            best_perm = list(p)
        else:
            # Simulated annealing-like acceptance could go here, 
            # but sticking to strict improvement for robustness.
            pass

    return tools['make_solution'](best_perm)