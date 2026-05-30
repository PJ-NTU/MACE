# MACE evolved heuristic 08/10 for problem: p_median_uncapacitated
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Redesigned solver for the Uncapacitated P-Median Problem.
    
    Modification: Enhanced the construction step by using a 'Semi-Greedy' 
    (GRASP-like) approach instead of purely random initialization. By choosing 
    from the top-k candidates during construction, we maintain better structural 
    integrity of the initial solutions, leading to faster convergence in 
    the Lin-Kernighan refinement phase.
    """
    start_time = time.time()
    n = instance['n']
    p = instance['p']
    dist = instance['dist']
    
    best_overall_medians = None
    best_overall_cost = float('inf')
    
    num_restarts = 0
    while time.time() - start_time < time_limit_s * 0.85:
        num_restarts += 1
        
        if num_restarts == 1:
            current_medians = tools['greedy_add_one_until_p']()
        else:
            # Semi-Greedy construction: 
            # Start with a random choice, then greedily pick the next median
            # from a restricted candidate list (RCL) of the best performing nodes.
            current_medians = []
            candidates = list(range(1, n + 1))
            random.shuffle(candidates)
            current_medians.append(candidates.pop())
            
            while len(current_medians) < p:
                best_c = None
                best_c_score = float('inf')
                # Evaluate a sample of nodes to find a high-quality addition
                sample = random.sample(candidates, min(len(candidates), max(1, n // 10)))
                for c in sample:
                    temp_medians = current_medians + [c]
                    score = tools['cost_given_open'](temp_medians)
                    if score < best_c_score:
                        best_c_score = score
                        best_c = c
                current_medians.append(best_c)
                candidates.remove(best_c)
            
        remaining_time = time_limit_s - (time.time() - start_time)
        if remaining_time < 0.1:
            break
            
        optimized_medians = tools['apply_interchange_LK'](
            current_medians, 
            time_limit_s=remaining_time
        )
        
        current_cost = tools['cost_given_open'](optimized_medians)
        
        if current_cost < best_overall_cost:
            best_overall_cost = current_cost
            best_overall_medians = optimized_medians
            
        if n < 50:
            break

    if best_overall_medians is None:
        best_overall_medians = tools['greedy_add_one_until_p']()

    return {"medians": list(best_overall_medians)}