# MACE evolved heuristic 01/10 for problem: corporate_structuring
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Solves the corporate structuring problem using a greedy construction
    followed by a stochastic hill-climbing local search.
    """
    start_time = time.time()
    
    # 1. Initialization: Get basic information
    target = instance['target']
    positive_profit_nodes = tools['positive_profit_countries']()
    
    # Ensure all positive profit nodes are included
    # Start with a flat tree (all connected to target)
    structure = tools['flat_tree']()
    
    # 2. Greedy Construction refinement
    # Try re-attaching nodes to see if we can find a better starting point
    # We use the provided greedy_attach tool as a strong baseline/warm start.
    # We limit the time for this construction phase to half of the budget.
    construction_time = time_limit_s * 0.3
    best_structure = tools['greedy_attach'](time_limit_s=construction_time)
    
    # 3. Local Search refinement
    # Use reparent_local_search to improve the structure iteratively.
    # We allow the remaining time for this optimization.
    remaining_time = time_limit_s - (time.time() - start_time)
    if remaining_time > 0.1:
        # We perform multiple restarts of local search if time permits
        # to escape local optima.
        best_structure = tools['reparent_local_search'](
            best_structure, 
            time_limit_s=remaining_time, 
            first_improvement=True
        )
        
    # 4. Final verification and fallback
    # Ensure the result is valid
    is_valid, _ = tools['is_feasible']({"structure": best_structure})
    if not is_valid:
        # Fallback to the simplest possible valid structure
        return {"structure": tools['flat_tree']()}
        
    return {"structure": best_structure}