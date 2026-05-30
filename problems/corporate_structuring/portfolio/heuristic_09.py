# MACE evolved heuristic 09/10 for problem: corporate_structuring
import time
import random
import numpy as np

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    A 'Simulated Annealing on Tree Spanning Trees' heuristic with a 
    'Node-Centric' perturbation strategy.
    
    Unlike the portfolio, which heavily relies on 'greedy_attach' and 
    'reparent_local_search' (deterministic/first-improvement hill-climbers),
    this heuristic uses a global temperature-based search that allows 
    non-improving moves. It explicitly avoids the greedy construction 
    template, opting for a random Pruefer-sequence-inspired tree 
    initialization to explore structurally diverse regions of the 
    feasible space early on.
    """
    start_time = time.time()
    target = instance['target']
    N = instance['N']
    positive_nodes = [n for n in tools['positive_profit_countries']() if n != target]
    
    def get_random_tree():
        # Generate a random tree structure using a randomized parent assignment
        # ensuring all positive profit nodes are connected.
        nodes = list(positive_nodes)
        random.shuffle(nodes)
        structure = {target: 0}
        connected = [target]
        for node in nodes:
            parent = random.choice(connected)
            structure[node] = parent
            connected.append(node)
        return structure

    # Initialize: Start with a random tree rather than greedy/flat
    current_structure = get_random_tree()
    best_structure = current_structure.copy()
    best_score = tools['tree_score'](best_structure)
    current_score = best_score
    
    # Annealing schedule
    temp = 100.0
    cooling_rate = 0.999
    
    # Main loop: Perform random structural perturbations
    while time.time() - start_time < time_limit_s * 0.95:
        # Perturbation: Randomly pick a node and re-attach to a random node
        # This is a 'global' move compared to local hill-climbing
        node_to_move = random.choice(positive_nodes)
        potential_parents = [target] + [n for n in positive_nodes if n != node_to_move]
        new_parent = random.choice(potential_parents)
        
        if current_structure.get(node_to_move) == new_parent:
            continue
            
        trial_structure = current_structure.copy()
        trial_structure[node_to_move] = new_parent
        
        # Check validity implicitly via tree_score
        trial_score = tools['tree_score'](trial_structure)
        
        if trial_score != float('-inf'):
            # Metropolis criterion: Accept non-improving moves with probability
            if trial_score > current_score:
                accept = True
            else:
                delta = trial_score - current_score
                # Use a normalized temperature
                prob = np.exp(delta / (temp + 1e-9))
                accept = random.random() < prob
            
            if accept:
                current_structure = trial_structure
                current_score = trial_score
                if current_score > best_score:
                    best_score = current_score
                    best_structure = current_structure.copy()
        
        # Cool down
        temp *= cooling_rate
        if temp < 0.001:
            temp = 10.0 # Re-heat to escape local optima
            
    # Final cleanup to ensure compliance
    is_valid, _ = tools['is_feasible']({"structure": best_structure})
    if not is_valid:
        return {"structure": tools['flat_tree']()}
        
    return {"structure": best_structure}