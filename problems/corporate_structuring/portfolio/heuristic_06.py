# MACE evolved heuristic 06/10 for problem: corporate_structuring
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Solve the corporate structuring problem using a multi-start greedy construction 
    followed by stochastic local search, with an improved warm-start strategy.
    """
    start_time = time.time()
    target = instance['target']
    pos_profit_countries = tools['positive_profit_countries']()
    
    # 1. Improved Warm Start: Instead of just one greedy pass, try multiple 
    # random permutations to explore the search space before local search.
    best_structure = None
    best_score = -float('inf')
    
    construction_budget = time_limit_s * 0.2
    
    # Try a few different permutations to find a better starting point
    while time.time() - start_time < construction_budget:
        perm = list(pos_profit_countries)
        if target in perm:
            perm.remove(target)
        random.shuffle(perm)
        
        current_structure = tools['greedy_attach'](order=perm, time_limit_s=construction_budget / 3)
        current_score = tools['tree_score'](current_structure)
        
        if current_score > best_score:
            best_score = current_score
            best_structure = current_structure
            
        if len(pos_profit_countries) < 5: # Small instances, don't over-loop
            break
            
    if best_structure is None:
        best_structure = tools['flat_tree']()
    
    # 2. Refinement: Use reparent_local_search to improve the structure.
    remaining_time = time_limit_s - (time.time() - start_time)
    if remaining_time > 0.1:
        refined_structure = tools['reparent_local_search'](
            structure=best_structure, 
            time_limit_s=remaining_time, 
            first_improvement=True
        )
        if refined_structure:
            best_structure = refined_structure
            
    # 3. Final check: Ensure the output format is correct.
    for country in pos_profit_countries:
        if country == target:
            if country not in best_structure:
                best_structure[country] = 0
            continue
        if country not in best_structure:
            best_structure[country] = target
            
    return {"structure": best_structure}