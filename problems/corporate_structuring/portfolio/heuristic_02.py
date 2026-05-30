# MACE evolved heuristic 02/10 for problem: corporate_structuring
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Solves the corporate structuring problem using a greedy initialization 
    followed by a randomized local search (reparenting) to optimize the tree structure.
    """
    start_time = time.time()
    
    # 1. Warm start: Use a greedy construction approach
    # We prioritize countries with the largest profits and try to find the best parent
    # for them among the already-placed nodes, including the target root.
    structure = tools['greedy_attach'](time_limit_s=time_limit_s * 0.3)
    
    if not structure:
        # Fallback to a flat tree if greedy fails
        structure = tools['flat_tree']()
        
    # 2. Local Search: Iterative improvement
    # We perform hill-climbing by attempting to move nodes to different parents.
    # Given the tree structure, we use the provided reparent_local_search tool.
    # We allocate the remaining time budget to this refinement process.
    remaining_time = time_limit_s - (time.time() - start_time)
    if remaining_time > 0.1:
        refined_structure = tools['reparent_local_search'](
            structure, 
            time_limit_s=remaining_time * 0.8, 
            first_improvement=True
        )
        if refined_structure:
            structure = refined_structure
            
    # 3. Final check: Ensure the output is valid
    # The structure must be a dict where keys are children and values are parents.
    # All nodes with positive profit must be included.
    positive_nodes = set(tools['positive_profit_countries']())
    
    # Add missing positive profit countries to the flat tree if any were missed
    # (though greedy_attach should handle this).
    for node in positive_nodes:
        if node != instance['target'] and node not in structure:
            structure[node] = instance['target']
            
    # Clean up structure: Ensure target is parent 0
    structure[instance['target']] = 0
    
    return {"structure": structure}