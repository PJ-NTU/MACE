# MACE evolved heuristic 07/10 for problem: corporate_structuring
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    A 'Variable Neighborhood Search' (VNS) solver.
    
    Fixed: The KeyError: 1 occurred because the loop tried to access/modify 
    nodes in the structure dictionary that might not have been present in 
    the initial flat_tree (if a country had <= 0 profit). 
    """
    start_time = time.time()
    target = instance['target']
    
    # Start from a flat structure (contains only positive profit countries)
    structure = tools['flat_tree']()
    best_score = tools['tree_score'](structure)
    
    # Get all positive profit countries (required to be in the tree)
    positive_nodes = tools['positive_profit_countries']()
    # The structure keys only contain positive profit countries excluding the target
    # because the target has parent 0 (not in structure dict per eval_func rules).
    nodes = [n for n in positive_nodes if n != target]

    # VNS Loop
    while time.time() - start_time < time_limit_s * 0.9:
        if not nodes: 
            break
        
        # Shake
        k = random.randint(1, 3)
        trial_structure = structure.copy()
        
        try:
            if k == 1:
                # Standard reparent
                node = random.choice(nodes)
                new_parent = random.choice([target] + [n for n in nodes if n != node])
                trial_structure[node] = new_parent
                
            elif k == 2:
                # Swap operation
                if len(nodes) >= 2:
                    n1, n2 = random.sample(nodes, 2)
                    trial_structure[n1], trial_structure[n2] = trial_structure[n2], trial_structure[n1]
                    
            elif k == 3:
                # Subtree graft: move node and its children (children implicitly follow)
                node = random.choice(nodes)
                new_parent = random.choice([target] + [n for n in nodes if n != node])
                trial_structure[node] = new_parent
            
            # Evaluate
            is_valid, _ = tools['is_feasible']({"structure": trial_structure})
            if is_valid:
                score = tools['tree_score'](trial_structure)
                if score > best_score:
                    structure = trial_structure
                    best_score = score
                else:
                    # Quick greedy improvement
                    node = random.choice(nodes)
                    for p in random.sample([target] + [n for n in nodes if n != node], min(5, len(nodes))):
                        temp = trial_structure.copy()
                        temp[node] = p
                        if tools['is_feasible']({"structure": temp})[0]:
                            s = tools['tree_score'](temp)
                            if s > best_score:
                                structure = temp
                                best_score = s
                                break
        except (KeyError, ValueError, IndexError):
            continue

    return {"structure": structure}