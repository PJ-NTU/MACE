# MACE evolved heuristic 07/10 for problem: p_median_uncapacitated
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Synthesized heuristic for UPM:
    - Small instances (n <= 100): Use intensive Lin-Kernighan (LK) search.
    - Large instances (n > 100): Use greedy construction followed by 
      a hybrid of LK and aggressive, time-budgeted swap-1-for-1 local search.
      
    Hypothesis: 
    Parent A's multi-start LK is superior for smaller, highly constrained 
    search spaces where global optima are reachable via multiple restarts. 
    Parent B's shake-and-refine approach is better for large instances 
    where the search space is too vast to restart, favoring local basin 
    exploration around a strong greedy baseline.
    """
    start_time = time.time()
    n = instance['n']
    p = instance['p']

    def get_remaining():
        return max(0.01, time_limit_s - (time.time() - start_time))

    # 1. Initialization
    try:
        current_medians = tools['greedy_add_one_until_p']()
    except Exception:
        current_medians = random.sample(range(1, n + 1), p)

    # 2. Strategy Dispatch
    # Small instances: favor exhaustive multi-start LK (Parent A logic)
    if n <= 100:
        best_medians = current_medians
        best_cost = tools['cost_given_open'](best_medians)
        
        iteration = 0
        while get_remaining() > 0.1:
            if iteration > 0:
                current_medians = random.sample(range(1, n + 1), p)
            
            try:
                optimized = tools['apply_interchange_LK'](
                    current_medians, 
                    time_limit_s=min(get_remaining(), 2.0)
                )
                cost = tools['cost_given_open'](optimized)
                if cost < best_cost:
                    best_cost = cost
                    best_medians = optimized
            except Exception:
                pass
            iteration += 1
            if iteration > 5: break
        current_medians = best_medians

    # Large instances: favor deep LK followed by shake-and-refine (Parent B logic)
    else:
        try:
            current_medians = tools['apply_interchange_LK'](
                current_medians,
                time_limit_s=get_remaining() * 0.6
            )
        except Exception:
            pass

        while get_remaining() > 0.5:
            new_candidate = list(current_medians)
            idx_to_remove = random.randrange(p)
            new_candidate.pop(idx_to_remove)
            
            pool = [i for i in range(1, n + 1) if i not in set(new_candidate)]
            if not pool: break
            new_candidate.append(random.choice(pool))
            
            try:
                refined = tools['apply_swap_one_for_one'](
                    new_candidate,
                    time_limit_s=min(0.5, get_remaining()),
                    first_improvement=True
                )
                if tools['objective']({"medians": refined}) < tools['objective']({"medians": current_medians}):
                    current_medians = refined
            except Exception:
                break

    # Final validation ensure feasibility
    final_sol = {"medians": list(current_medians)}
    is_valid, _ = tools['is_feasible'](final_sol)
    if not is_valid:
        # Fallback to greedy if something went wrong
        return {"medians": tools['greedy_add_one_until_p']()}
        
    return final_sol