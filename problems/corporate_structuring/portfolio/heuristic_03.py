# MACE evolved heuristic 03/10 for problem: corporate_structuring
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Solves the corporate structuring problem using a combination of a 
    greedy initialization and a randomized hill-climbing local search.
    """
    start_time = time.time()
    
    # 1. Warm start: Use the provided greedy_attach to get a decent baseline.
    # greedy_attach returns a dict with key "structure"
    greedy_res = tools['greedy_attach'](time_limit_s=time_limit_s * 0.2)
    # The tool returns the structure directly, but the problem expects {"structure": ...}
    # Note: If greedy_attach returns the structure dict directly, we wrap it.
    if isinstance(greedy_res, dict) and "structure" in greedy_res:
        best_structure = greedy_res["structure"]
    else:
        best_structure = greedy_res
        
    best_score = tools['tree_score'](best_structure)
    
    # 2. Local Search: Iterative improvement via random reparenting.
    target = instance['target']
    
    iteration = 0
    while time.time() - start_time < time_limit_s * 0.95:
        iteration += 1
        
        # Get current nodes in structure (excluding root/target)
        current_nodes = [n for n in best_structure.keys() if n != target]
        if not current_nodes:
            break
            
        # Create a copy to mutate
        current_structure = best_structure.copy()
        
        # Pick a random node (not the root) and attempt to move it to a new parent
        node_to_move = random.choice(current_nodes)
        
        # Potential parents are the target or any other node in the tree
        potential_parents = [target] + [n for n in current_nodes if n != node_to_move]
        new_parent = random.choice(potential_parents)
        
        old_parent = current_structure[node_to_move]
        if old_parent == new_parent:
            continue
            
        current_structure[node_to_move] = new_parent
        
        # Evaluate
        new_score = tools['tree_score'](current_structure)
        
        if new_score > best_score:
            best_score = new_score
            best_structure = current_structure
        
        # Occasionally perform a deeper local search if we aren't seeing progress
        if iteration % 50 == 0:
            improved = tools['reparent_local_search'](
                best_structure, 
                time_limit_s=max(0.1, (time_limit_s - (time.time() - start_time)) * 0.1)
            )
            score_improved = tools['tree_score'](improved)
            if score_improved > best_score:
                best_score = score_improved
                best_structure = improved

    # Final sanity check: ensure the structure is valid according to the tools
    final_sol = {"structure": best_structure}
    is_valid, _ = tools['is_feasible'](final_sol)
    if not is_valid:
        return {"structure": tools['flat_tree']()}
        
    return final_sol