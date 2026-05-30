# MACE evolved heuristic 05/10 for problem: p_median_uncapacitated
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Improved heuristic for the Uncapacitated p-median problem.
    
    Strategy:
    1. Start with the strong greedy construction.
    2. Perform a best-improvement local search (Teitz-Bart) using 
       apply_interchange_LK.
    3. Spend the remaining time budget on a multi-start random perturbation 
       with a biased selection: prioritize replacing medians that contribute 
       the most to the total cost.
    """
    start_time = time.time()
    n = instance['n']
    p = instance['p']
    dist = instance['dist']
    
    def get_time_left():
        return max(0.0, time_limit_s - (time.time() - start_time))

    # 1. Primary Construction
    best_medians = tools['greedy_add_one_until_p']()
    best_cost = tools['objective']({"medians": best_medians})

    # 2. Primary Refinement (Lin-Kernighan flavor)
    time_for_lk = get_time_left() * 0.7
    if time_for_lk > 0.5:
        refined = tools['apply_interchange_LK'](
            open_set=best_medians, 
            time_limit_s=time_for_lk
        )
        current_cost = tools['objective']({"medians": refined})
        if current_cost < best_cost:
            best_medians = refined
            best_cost = current_cost

    # 3. Escaping local optima: Biased Perturbation
    # Calculate contribution of each median to the total cost to identify "weak" medians
    while get_time_left() > 0.5:
        # Calculate individual median contribution
        contributions = {m: 0.0 for m in best_medians}
        for i in range(n):
            # Find nearest and second nearest to assign cost
            dists_to_medians = sorted([(dist[i][m - 1], m) for m in best_medians])
            contributions[dists_to_medians[0][1]] += dists_to_medians[0][0]
        
        # Sort medians by contribution (descending) and remove one of the worst
        sorted_medians = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
        idx_to_remove = sorted_medians[0][0]
        
        candidate = [m for m in best_medians if m != idx_to_remove]
        
        # Add 1 random node not in current set
        current_set = set(candidate)
        candidates = [i for i in range(1, n + 1) if i not in current_set]
        if not candidates:
            break
        candidate.append(random.choice(candidates))
        
        # Refine the perturbed candidate
        try:
            refined = tools['apply_swap_one_for_one'](
                open_set=candidate,
                time_limit_s=min(0.5, get_time_left()),
                first_improvement=True
            )
            c_cost = tools['objective']({"medians": refined})
            if c_cost < best_cost:
                best_medians = refined
                best_cost = c_cost
        except:
            pass

    return {"medians": best_medians}