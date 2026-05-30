# MACE evolved heuristic 01/10 for problem: generalised_assignment_problem
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    An optimized heuristic for GAP leveraging ILP for high-quality starts and
    stochastic local search (Hill Climbing with random restarts) to maximize
    the objective within the time limit.
    """
    start_time = time.time()
    n = tools['n_tasks']()
    m = tools['n_agents']()
    
    # Use ILP as the primary engine if time allows, as it is globally optimal
    # and provides a strong baseline for any subsequent local search.
    ilp_result = tools['ilp_gap'](time_limit_s=min(time_limit_s * 0.4, 5.0))
    
    if ilp_result is not None and tools['is_feasible_assignment'](ilp_result)[0]:
        best_assignment = ilp_result
    else:
        # Fallback to greedy construction if ILP fails or is unavailable
        best_assignment = tools['greedy_min_cost']()
        is_f, _ = tools['is_feasible_assignment'](best_assignment)
        if not is_f:
            # If greedy is infeasible, try resource ratio heuristic
            best_assignment = tools['greedy_min_resource_ratio']()
            is_f, _ = tools['is_feasible_assignment'](best_assignment)
            if not is_f:
                # Random initialization as last resort
                best_assignment = [random.randint(1, m) for _ in range(n)]

    best_score = tools['objective']({'assignments': best_assignment})

    # Stochastic Local Search (Hill Climbing with frequent resets)
    # Modified: Increased swap probability to 0.5 to better escape capacity-constrained local optima
    while time.time() - start_time < time_limit_s * 0.95:
        current = list(best_assignment)
        improved = False
        
        for _ in range(100):
            if time.time() - start_time > time_limit_s * 0.95:
                break
            
            # Stochastic move selection
            if random.random() < 0.5:
                # Swap move: powerful for escaping capacity-constrained local optima
                t1, t2 = random.sample(range(n), 2)
                candidate = tools['apply_swap_assignments'](current, t1, t2)
            else:
                # Reassignment move: standard neighborhood
                task = random.randint(0, n - 1)
                new_agent = random.randint(1, m)
                candidate = tools['apply_reassign'](current, task, new_agent)
            
            if candidate is not None:
                score = tools['objective']({'assignments': candidate})
                # Minimization/Maximization handled by objective tool (LOWER IS BETTER)
                if score < best_score:
                    best_score = score
                    best_assignment = candidate
                    current = candidate
                    improved = True
        
        if not improved:
            break
            
    return {'assignments': best_assignment}