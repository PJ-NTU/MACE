# MACE evolved heuristic 08/10 for problem: corporate_structuring
import time
import random
import math

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    A 'Tabu Search' heuristic with a focus on 'Non-greedy' neighborhood exploration.
    """
    start_time = time.time()
    target = instance['target']
    positive_nodes = [n for n in tools['positive_profit_countries']() if n != target]
    
    # Initialize with a random valid tree to ensure diversity from the start
    def get_random_tree():
        nodes = list(positive_nodes)
        random.shuffle(nodes)
        struct = {target: 0}
        available = [target]
        for node in nodes:
            parent = random.choice(available)
            struct[node] = parent
            available.append(node)
        return struct

    current_structure = get_random_tree()
    best_structure = current_structure.copy()
    best_score = tools['tree_score'](current_structure)
    
    # Tabu list: stores (child, parent) moves
    tabu_list = {}
    tabu_tenure = max(5, len(positive_nodes) // 4)
    
    # Search parameters
    iteration = 0
    stagnation_counter = 0
    
    while time.time() - start_time < time_limit_s * 0.9:
        iteration += 1
        
        # Select a candidate move: pick a node and move to a random parent
        node_to_move = random.choice(positive_nodes)
        potential_parents = [target] + [n for n in positive_nodes if n != node_to_move]
        new_parent = random.choice(potential_parents)
        
        # Check tabu status
        if tabu_list.get((node_to_move, new_parent), 0) > iteration:
            continue
            
        trial_structure = current_structure.copy()
        trial_structure[node_to_move] = new_parent
        
        trial_score = tools['tree_score'](trial_structure)
        
        # If invalid (cycle), skip
        if trial_score == float('-inf'):
            continue
            
        current_score = tools['tree_score'](current_structure)
        
        # Accept move logic: Tabu Search acceptance
        if trial_score > current_score:
            current_structure = trial_structure
            tabu_list[(node_to_move, new_parent)] = iteration + tabu_tenure
            stagnation_counter = 0
        else:
            # Probabilistic acceptance of inferior moves
            diff = trial_score - current_score
            if random.random() < math.exp(diff / (100.0 / (1 + iteration * 0.01))):
                current_structure = trial_structure
                stagnation_counter += 1
        
        if tools['tree_score'](current_structure) > best_score:
            best_score = tools['tree_score'](current_structure)
            best_structure = current_structure.copy()
            
        # Long-term memory: if stuck, restart from a random tree
        if stagnation_counter > 100:
            current_structure = get_random_tree()
            stagnation_counter = 0
            
    # Final check
    is_valid, _ = tools['is_feasible']({"structure": best_structure})
    if not is_valid:
        return {"structure": tools['flat_tree']()}
        
    return {"structure": best_structure}