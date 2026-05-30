# MACE evolved heuristic 05/10 for problem: p_median_capacitated
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Optimized heuristic for Capacitated P-Median Problem.
    1. Initial population: Mix of greedy seeds and random samples.
    2. Local Search: Hill-climbing with swap operator.
    3. Emergency fallback: ILP solver.
    """
    start_time = time.time()
    n = tools['n_customers']()
    p = tools['p']()

    best_solution = None
    best_obj = float('inf')

    def get_sol_obj(open_set, assignment):
        sol = tools['to_solution'](list(open_set), assignment)
        is_feas, _ = tools['is_feasible'](sol)
        if is_feas:
            return sol['objective'], sol
        return float('inf'), None

    # 1. Initial Seeding Strategies
    seeds = []
    # Greedy seeding
    try:
        seeds.append(set(tools['greedy_p_picks_by_distance']()))
    except:
        pass
    # Random seeds
    for _ in range(5):
        seeds.append(set(random.sample(range(n), p)))

    # 2. Iterative Improvement
    for open_set in seeds:
        if time.time() - start_time > time_limit_s * 0.5:
            break
        
        assignment = tools['assignment_by_nearest_feasible'](list(open_set))
        if -1 in assignment:
            continue
            
        current_obj, current_sol = get_sol_obj(open_set, assignment)
        if current_obj < best_obj:
            best_obj = current_obj
            best_solution = current_sol

        # Local Swap Search
        improved = True
        while improved and (time.time() - start_time < time_limit_s * 0.7):
            improved = False
            open_list = list(open_set)
            closed_list = [i for i in range(n) if i not in open_set]
            random.shuffle(open_list)
            
            for o in open_list:
                for c in closed_list:
                    if time.time() - start_time > time_limit_s * 0.7:
                        break
                    
                    new_set = (open_set - {o}) | {c}
                    new_assignment = tools['assignment_by_nearest_feasible'](list(new_set))
                    
                    if -1 not in new_assignment:
                        new_obj, new_sol = get_sol_obj(new_set, new_assignment)
                        if new_obj < current_obj:
                            current_obj = new_obj
                            current_sol = new_sol
                            open_set = new_set
                            assignment = new_assignment
                            improved = True
                            if current_obj < best_obj:
                                best_obj = current_obj
                                best_solution = current_sol
                            break
                if improved: break

    # 3. Final Fallback to ILP if no solution found or to refine
    if best_solution is None or (time.time() - start_time < time_limit_s * 0.95):
        remaining_time = max(0.5, time_limit_s - (time.time() - start_time))
        ilp_sol = tools['ilp_cpm'](time_limit_s=remaining_time)
        if ilp_sol:
            # Check if ILP is better than current best
            if ilp_sol['objective'] < best_obj:
                return ilp_sol
                
    return best_solution