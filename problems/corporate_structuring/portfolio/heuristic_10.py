# MACE evolved heuristic 10/10 for problem: corporate_structuring
import time
import random
import math

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Dispatcher-style solver:
    - If N is small, use Simulated Annealing (B-style) to explore the 
      state space more thoroughly.
    - If N is large, use Greedy Construction + Hill Climbing (A-style)
      to maintain computational efficiency and avoid stagnation.
    """
    start_time = time.time()
    N = instance['N']
    target = instance['target']
    
    # Feature Engineering: 
    # Small N allows for global search through SA.
    # Larger N makes the state space too sparse for SA, 
    # so we rely on local improvement heuristics.
    if N <= 25:
        # Simulated Annealing Strategy
        nodes = [n for n in range(1, N + 1) if n != target]
        current_structure = tools['flat_tree']()
        current_score = tools['tree_score'](current_structure)
        
        best_structure = current_structure.copy()
        best_score = current_score
        
        temp = 1000.0
        cooling_rate = 0.99
        
        while time.time() - start_time < time_limit_s * 0.9:
            node_to_move = random.choice(nodes)
            potential_parents = [target] + [n for n in range(1, N + 1) if n != node_to_move]
            new_parent = random.choice(potential_parents)
            
            if current_structure.get(node_to_move) == new_parent:
                continue
                
            trial_structure = current_structure.copy()
            trial_structure[node_to_move] = new_parent
            trial_score = tools['tree_score'](trial_structure)
            
            if trial_score != float('-inf'):
                if trial_score > current_score or (random.random() < math.exp((trial_score - current_score) / (temp + 1e-9))):
                    current_structure = trial_structure
                    current_score = trial_score
                    if current_score > best_score:
                        best_score = current_score
                        best_structure = current_structure.copy()
            
            temp *= cooling_rate
            if temp < 0.01: temp = 100.0
            
    else:
        # Greedy + Local Search Strategy
        # Use a significant portion of time for construction
        construction_time = time_limit_s * 0.2
        best_structure = tools['greedy_attach'](time_limit_s=construction_time)
        
        remaining_time = time_limit_s - (time.time() - start_time)
        if remaining_time > 0.1:
            best_structure = tools['reparent_local_search'](
                best_structure, 
                time_limit_s=remaining_time, 
                first_improvement=True
            )
            
    # Final safety check
    is_valid, _ = tools['is_feasible']({"structure": best_structure})
    if not is_valid:
        return {"structure": tools['flat_tree']()}
        
    return {"structure": best_structure}