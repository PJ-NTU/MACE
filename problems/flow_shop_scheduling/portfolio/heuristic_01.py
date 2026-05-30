# MACE evolved heuristic 01/10 for problem: flow_shop_scheduling
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Solves the flow shop scheduling problem using the NEH heuristic 
    followed by an Iterated Local Search (ILS) with insertion-based 
    local search to refine the solution within the time limit.
    """
    start_time = time.time()
    
    # Use the gold-standard NEH construction as the initial solution
    # The tool provides a 1-indexed permutation
    current_perm = tools['neh_construct']()
    
    # We allocate a portion of the time for local search refinement
    # Leaving a buffer for the remaining time
    refinement_limit = time_limit_s * 0.9
    
    def get_makespan(perm):
        return tools['simulate_makespan'](perm)

    best_perm = list(current_perm)
    best_makespan = get_makespan(best_perm)
    
    # Iterated Local Search
    # 1. Local search refinement (Insertion Search)
    # 2. Perturbation (Random swap) to escape local optima
    
    while time.time() - start_time < refinement_limit:
        # Perform Insertion Search (the most effective neighborhood for flow shop)
        improved_perm = tools['apply_insertion_search'](
            list(current_perm), 
            time_limit_s=max(0.1, (refinement_limit - (time.time() - start_time)) / 2)
        )
        
        improved_makespan = get_makespan(improved_perm)
        
        if improved_makespan < best_makespan:
            best_perm = list(improved_perm)
            best_makespan = improved_makespan
            current_perm = list(improved_perm)
        else:
            # Perturbation: swap 2 random elements to escape local optimum
            current_perm = list(best_perm)
            n = len(current_perm)
            if n > 1:
                idx1, idx2 = random.sample(range(n), 2)
                current_perm[idx1], current_perm[idx2] = current_perm[idx2], current_perm[idx1]
        
        # Check time before next iteration
        if time.time() - start_time > refinement_limit:
            break
            
    return tools['make_solution'](best_perm)