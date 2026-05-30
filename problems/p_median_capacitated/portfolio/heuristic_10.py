# MACE evolved heuristic 10/10 for problem: p_median_capacitated
import time
import random
import math

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    A dispatcher-style heuristic for the Capacitated P-Median Problem.
    
    Dispatch Logic:
    - Small instances (n <= 60): Prioritize ILP (Parent A approach).
    - Dense/Large instances (n > 60): Prioritize Iterative Local Search (Parent B approach).
    - The dispatch criteria relies on the observation that ILP is optimal for small n,
      whereas the swap-based local search heuristic is more robust to the search space
      explosion seen in larger CPM instances.
    """
    start_time = time.time()
    n = tools['n_customers']()
    p = tools['p']()
    
    # Heuristic Dispatcher
    # Use ILP for small instances where the global optimum is likely reachable.
    # Use randomized greedy/swap-based search for larger instances.
    if n <= 60:
        # Parent A-style: Heavy focus on ILP
        res = tools['ilp_cpm'](time_limit_s=max(1.0, time_limit_s * 0.7))
        if res:
            return res
        # Fallback to greedy if ILP fails
        greedy_m = tools['greedy_p_picks_by_distance']()
        greedy_a = tools['assignment_by_nearest_feasible'](greedy_m)
        if -1 not in greedy_a:
            return tools['to_solution'](greedy_m, greedy_a)
        return {}
    else:
        # Parent B-style: Iterative Local Search
        best_sol = None
        best_obj = float('inf')
        
        # Initial seeding: Start with greedy to get a baseline
        seeds = [tools['greedy_p_picks_by_distance']()]
        
        while time.time() - start_time < time_limit_s * 0.9:
            if seeds:
                current_medians = seeds.pop(0)
            else:
                current_medians = random.sample(range(n), p)
                
            assignments = tools['assignment_by_nearest_feasible'](current_medians)
            
            # Simple repair logic
            if -1 in assignments:
                unused = list(set(range(n)) - set(current_medians))
                if unused:
                    current_medians[random.randrange(p)] = random.choice(unused)
                    assignments = tools['assignment_by_nearest_feasible'](current_medians)
            
            if -1 in assignments:
                continue
                
            # Perform swap-based local search
            remaining = time_limit_s - (time.time() - start_time)
            if remaining < 0.1:
                break
                
            new_medians, new_assignments = tools['apply_swap_open_close'](
                current_medians, t_limit=min(remaining * 0.5, 1.5)
            )
            
            candidate = tools['to_solution'](new_medians, new_assignments)
            is_f, _ = tools['is_feasible'](candidate)
            
            if is_f:
                obj = candidate['objective']
                if obj < best_obj:
                    best_obj = obj
                    best_sol = candidate
                    
        # Final fallback
        if best_sol is None:
            greedy_m = tools['greedy_p_picks_by_distance']()
            greedy_a = tools['assignment_by_nearest_feasible'](greedy_m)
            if -1 not in greedy_a:
                return tools['to_solution'](greedy_m, greedy_a)
            
        return best_sol if best_sol else {}