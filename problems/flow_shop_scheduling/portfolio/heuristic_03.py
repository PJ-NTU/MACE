# MACE evolved heuristic 03/10 for problem: flow_shop_scheduling
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Solves the Flow Shop Scheduling problem using the NEH constructive heuristic
    followed by an Iterated Local Search (ILS) with Insertion (Or-opt-1) moves.
    """
    start_time = time.time()
    n = instance['n']
    
    # 1. Warm start using NEH
    current_perm = tools['neh_construct']()
    
    # 2. Refine using Insertion Search if time permits
    # Insertion search is the most effective local search for flow shop.
    remaining_time = time_limit_s - (time.time() - start_time)
    if remaining_time > 0.1:
        current_perm = tools['apply_insertion_search'](
            current_perm, 
            time_limit_s=remaining_time, 
            first_improvement=True
        )
    
    # 3. Iterated Local Search (Perturbation + Local Search)
    # If we have significant time left, try to escape local optima by shaking
    # the sequence and re-applying insertion search.
    while time.time() - start_time < time_limit_s * 0.9:
        # Perturbation: Perform a random block shift (destroy/repair)
        perturbed_perm = list(current_perm)
        # Move a sub-sequence of length 2 to a random position
        idx = random.randint(0, n - 2)
        block = [perturbed_perm.pop(idx), perturbed_perm.pop(idx)]
        pos = random.randint(0, n - 2)
        perturbed_perm.insert(pos, block[0])
        perturbed_perm.insert(pos + 1, block[1])
        
        # Local search polish
        remaining_time = time_limit_s - (time.time() - start_time)
        if remaining_time < 0.05:
            break
            
        candidate = tools['apply_insertion_search'](
            perturbed_perm, 
            time_limit_s=remaining_time, 
            first_improvement=True
        )
        
        # Compare makespans
        if tools['simulate_makespan'](candidate) < tools['simulate_makespan'](current_perm):
            current_perm = candidate
        else:
            # Simulated Annealing style acceptance could go here, but simple 
            # hill climbing is more robust given the tight time constraints.
            pass

    return tools['make_solution'](current_perm)