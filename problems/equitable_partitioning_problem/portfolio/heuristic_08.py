# MACE evolved heuristic 08/10 for problem: equitable_partitioning_problem
import time
import random
import numpy as np

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Dispatcher-style solver for equitable partitioning.
    
    Hypothesis: 
    - Small instances (n < 200) or high-density attribute matrices benefit from 
      the ILP solver (exact or near-exact), as the problem space is small enough 
      to explore exhaustively.
    - Large instances (n >= 200) benefit from a randomized GRASP approach (Parent A), 
      as the local search space is too large for ILP to converge and stochastic 
      restarts help avoid local minima.
    """
    start_time = time.time()
    data = instance['data']
    n_individuals = len(data)
    n_attributes = len(data[0])
    
    # Feature calculation: Density of the attribute matrix
    # Higher density implies more constraints overlap, making ILP more difficult
    # to solve globally, whereas sparsity might be easier for ILP.
    density = np.mean(data)
    
    def get_remaining():
        return max(0.1, time_limit_s - (time.time() - start_time) - 0.2)

    # Dispatch Logic
    # If the problem is small enough for ILP, use it. 
    # Otherwise, use the robust GRASP/Local Search approach.
    if n_individuals <= 150:
        # ILP is likely to find the optimal or near-optimal solution quickly.
        try:
            ilp_result = tools['ilp_partition'](time_limit_s=get_remaining())
            if ilp_result is not None:
                return {'assignment': ilp_result}
        except Exception:
            pass
    
    # Fallback to GRASP-inspired refinement (Parent A style)
    # We use multiple random starts if time permits, otherwise one strong start.
    best_assignment = tools['greedy_balanced_split'](seed=42)
    best_score = tools['total_imbalance'](best_assignment)
    
    # Attempt to improve with multiple short local searches if time allows
    num_restarts = 3 if n_individuals < 500 else 1
    
    for i in range(num_restarts):
        if get_remaining() < 1.0:
            break
            
        current_seed = 42 + i
        candidate = tools['greedy_balanced_split'](seed=current_seed)
        
        refined = tools['apply_swap_items_across_groups'](
            partition=candidate,
            time_limit_s=get_remaining() / (num_restarts - i),
            seed=current_seed
        )
        
        score = tools['total_imbalance'](refined)
        if score < best_score:
            best_score = score
            best_assignment = refined
            
    return {'assignment': best_assignment}