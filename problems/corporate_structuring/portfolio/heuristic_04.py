# MACE evolved heuristic 04/10 for problem: corporate_structuring
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Solves the corporate structuring problem using a greedy construction
    followed by a stochastic hill-climbing local search, using a randomized
    order for the greedy construction to explore different tree topologies.
    """
    start_time = time.time()
    
    # 1. Initialization
    positive_profit_nodes = tools['positive_profit_countries']()
    
    # 2. Randomized Greedy Construction
    # Instead of just the default order, we shuffle the nodes to explore
    # different construction paths, which helps avoid getting stuck in 
    # poor local optima early on.
    nodes_to_attach = list(positive_profit_nodes)
    if instance['target'] in nodes_to_attach:
        nodes_to_attach.remove(instance['target'])
    random.shuffle(nodes_to_attach)
    
    construction_time = time_limit_s * 0.4
    best_structure = tools['greedy_attach'](order=nodes_to_attach, time_limit_s=construction_time)
    
    # 3. Local Search refinement
    # Use reparent_local_search to improve the structure iteratively.
    remaining_time = time_limit_s - (time.time() - start_time)
    if remaining_time > 0.1:
        # Perform local search until the budget is nearly exhausted
        best_structure = tools['reparent_local_search'](
            best_structure, 
            time_limit_s=remaining_time, 
            first_improvement=True
        )
        
    # 4. Final verification and fallback
    is_valid, _ = tools['is_feasible']({"structure": best_structure})
    if not is_valid:
        # Fallback to the simplest possible valid structure
        return {"structure": tools['flat_tree']()}
        
    return {"structure": best_structure}