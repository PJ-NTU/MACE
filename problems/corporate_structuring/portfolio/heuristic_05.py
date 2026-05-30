# MACE evolved heuristic 05/10 for problem: corporate_structuring
import time
import random
import math

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    A Simulated Annealing heuristic that explores the space of valid trees by
    performing controlled 'subtree-swap' and 'reparent' perturbations.
    Unlike the portfolio which relies on local-search hill-climbing, this 
    uses a temperature-based probabilistic acceptance criterion to navigate
    local optima.
    """
    start_time = time.time()
    target = instance['target']
    nodes = [n for n in range(1, instance['N'] + 1) if n != target]
    
    # Initial state: Flat tree
    current_structure = tools['flat_tree']()
    current_score = tools['tree_score'](current_structure)
    
    best_structure = current_structure.copy()
    best_score = current_score
    
    # Parameters for Simulated Annealing
    temp = 1000.0
    cooling_rate = 0.995
    
    while time.time() - start_time < time_limit_s * 0.9:
        # Generate a neighbor: pick a random node and change its parent
        node_to_move = random.choice(nodes)
        potential_parents = [target] + [n for n in nodes if n != node_to_move]
        new_parent = random.choice(potential_parents)
        
        if current_structure.get(node_to_move) == new_parent:
            continue
            
        # Create trial structure
        trial_structure = current_structure.copy()
        trial_structure[node_to_move] = new_parent
        
        # Check validity (cycle check is implicit in tree_score returning -inf)
        trial_score = tools['tree_score'](trial_structure)
        
        if trial_score != float('-inf'):
            # Acceptance probability
            if trial_score > current_score:
                accept = True
            else:
                delta = trial_score - current_score
                # Protect against overflow/underflow
                prob = math.exp(min(50, max(-50, delta / (temp + 1e-9))))
                accept = random.random() < prob
            
            if accept:
                current_structure = trial_structure
                current_score = trial_score
                if current_score > best_score:
                    best_score = current_score
                    best_structure = current_structure.copy()
        
        # Cool down
        temp *= cooling_rate
        
        # Periodically reset temperature if stuck
        if temp < 0.01:
            temp = 100.0
            
    # Safety check: ensure feasibility
    is_valid, _ = tools['is_feasible']({"structure": best_structure})
    if not is_valid:
        return {"structure": tools['flat_tree']()}
        
    return {"structure": best_structure}