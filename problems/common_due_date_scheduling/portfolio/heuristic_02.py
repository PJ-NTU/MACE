# MACE evolved heuristic 02/10 for problem: common_due_date_scheduling
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Solves the Common Due Date scheduling problem using a V-shape construction
    followed by iterated local search (ILS) with insertion and swap moves.
    """
    start_time = time.time()
    n = tools['n_jobs']()
    
    # 1. Generate strong initial solution using V-shape construction
    current_schedule = tools['v_shape_construct']()
    best_penalty = tools['compute_total_penalty'](current_schedule)
    best_schedule = list(current_schedule)

    # 2. Iterated Local Search Loop
    # Time buffer for safety
    while time.time() - start_time < time_limit_s * 0.85:
        # Perform intensification (local search)
        # Apply Or-opt-1 (insertion) to refine
        improved_schedule = tools['apply_insertion_search'](
            current_schedule, 
            time_limit_s=(time_limit_s - (time.time() - start_time)) * 0.5
        )
        
        # Apply 2-opt swap to refine further
        improved_schedule = tools['apply_swap_2opt'](
            improved_schedule, 
            time_limit_s=(time_limit_s - (time.time() - start_time)) * 0.5
        )
        
        current_penalty = tools['compute_total_penalty'](improved_schedule)
        
        if current_penalty < best_penalty:
            best_penalty = current_penalty
            best_schedule = list(improved_schedule)
            current_schedule = list(improved_schedule)
        else:
            # Perturbation: Random swap to escape local optima
            current_schedule = list(best_schedule)
            idx1, idx2 = random.sample(range(n), 2)
            current_schedule[idx1], current_schedule[idx2] = current_schedule[idx2], current_schedule[idx1]
            
        # Check if we should stop
        if time.time() - start_time > time_limit_s * 0.95:
            break

    return {'schedule': best_schedule}