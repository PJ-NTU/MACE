# MACE evolved heuristic 08/10 for problem: p_median_capacitated
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Dispatcher-style solver for Capacitated P-Median Problem.
    
    Hypothesis:
    - Small to Medium instances (n < 200) are best handled by ILP or aggressive 
      Local Search (A-style) because the search space is small enough to optimize 
      globally.
    - Large instances (n >= 200) or instances where greedy seeds fail to find 
      feasible solutions early require Tabu Search (B-style) to navigate the 
      constrained landscape without getting trapped in local optima.
    """
    start_time = time.time()
    n = tools['n_customers']()
    p = tools['p']()
    
    # Feature extraction for dispatching
    is_large = n >= 200
    
    def get_sol_obj(open_set):
        assignment = tools['assignment_by_nearest_feasible'](list(open_set))
        if -1 in assignment:
            return float('inf'), None
        sol = tools['to_solution'](list(open_set), assignment)
        return sol['objective'], sol

    # Strategy A: Hill-Climbing with Swap (Good for smaller instances)
    def run_strategy_a(time_budget):
        best_obj, best_sol = float('inf'), None
        seeds = [set(tools['greedy_p_picks_by_distance']())]
        for _ in range(3):
            seeds.append(set(random.sample(range(n), p)))
            
        for open_set in seeds:
            if time.time() - start_time > time_budget: break
            curr_obj, curr_sol = get_sol_obj(open_set)
            if curr_obj < best_obj:
                best_obj, best_sol = curr_obj, curr_sol
                
            improved = True
            while improved and time.time() - start_time < time_budget:
                improved = False
                open_list, closed_list = list(open_set), [i for i in range(n) if i not in open_set]
                for o in open_list:
                    for c in closed_list:
                        new_set = (open_set - {o}) | {c}
                        obj, sol = get_sol_obj(new_set)
                        if obj < curr_obj:
                            curr_obj, curr_sol = obj, sol
                            open_set = new_set
                            improved = True
                            if curr_obj < best_obj:
                                best_obj, best_sol = curr_obj, curr_sol
                            break
                    if improved: break
        return best_sol

    # Strategy B: Tabu Search (Good for large/complex instances)
    def run_strategy_b(time_budget):
        current_open = set(random.sample(range(n), p))
        best_obj, best_sol = float('inf'), None
        tabu_list = {}
        
        while time.time() - start_time < time_budget:
            open_list, closed_list = list(current_open), [i for i in range(n) if i not in current_open]
            best_move = None
            sample_o = random.sample(open_list, min(len(open_list), 4))
            sample_c = random.sample(closed_list, min(len(closed_list), 4))
            
            for o in sample_o:
                for c in sample_c:
                    if tabu_list.get(o, 0) > time.time(): continue
                    new_set = (current_open - {o}) | {c}
                    obj, sol = get_sol_obj(new_set)
                    if obj < best_obj:
                        best_obj, best_sol = obj, sol
                        best_move = new_set
            
            if best_move:
                current_open = best_move
                tabu_list[list(current_open - best_move)[0] if best_move else 0] = time.time() + 0.5
        return best_sol

    # Dispatcher
    if not is_large:
        res = run_strategy_a(time_limit_s * 0.6)
    else:
        res = run_strategy_b(time_limit_s * 0.6)
        
    # Final cleanup: ILP if still not optimal or infeasible
    if res is None or time.time() - start_time < time_limit_s * 0.95:
        ilp_sol = tools['ilp_cpm'](time_limit_s=max(0.2, time_limit_s - (time.time() - start_time)))
        if ilp_sol:
            return ilp_sol
            
    return res if res else tools['to_solution'](list(range(p)), tools['assignment_by_nearest_feasible'](list(range(p))))