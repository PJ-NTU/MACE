# MACE evolved heuristic 09/10 for problem: equitable_partitioning_problem
import time
import random
import numpy as np

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    An optimized hybrid heuristic for the equitable partitioning problem.
    
    Strategy:
    1. Warm-start with a high-quality constructive heuristic (greedy_balanced_split).
    2. Use a multi-start local search strategy:
        - Apply aggressive local search (swap/move) to refine the baseline.
        - If time permits, perturb the best solution (randomly reassign a subset of items)
          and re-optimize to escape local optima.
    3. Respect time budget strictly.
    """
    start_time = time.time()
    
    def get_remaining():
        return max(0.05, time_limit_s - (time.time() - start_time) - 0.1)

    # 1. Initial Construction
    # greedy_balanced_split provides a strong baseline compared to random or round-robin.
    best_assignment = tools['greedy_balanced_split'](seed=42)
    best_score = tools['total_imbalance'](best_assignment)
    
    # 2. Refinement Loop
    # We use apply_swap_items_across_groups as it is a highly optimized first-improvement
    # local search provided by the environment.
    
    # First pass: refine the greedy starting point
    best_assignment = tools['apply_swap_items_across_groups'](
        partition=best_assignment,
        time_limit_s=get_remaining() * 0.6,
        seed=101
    )
    best_score = tools['total_imbalance'](best_assignment)
    
    # 3. Perturbation & Restarts
    # If time remains, perform random perturbations to explore new regions of the solution space.
    n_individuals = len(instance['data'])
    
    while get_remaining() > 0.5:
        # Create a new candidate by partially shuffling the current best
        # Perturb ~10% of the assignments
        candidate = list(best_assignment)
        num_to_perturb = max(1, n_individuals // 10)
        for _ in range(num_to_perturb):
            idx = random.randrange(n_individuals)
            candidate[idx] = random.randint(1, 8)
            
        # Ensure the perturbed partition is valid (not empty groups)
        # tools['apply_swap_items_across_groups'] handles the repair of empty groups.
        try:
            refined = tools['apply_swap_items_across_groups'](
                partition=candidate,
                time_limit_s=get_remaining(),
                seed=random.randint(0, 10000)
            )
            score = tools['total_imbalance'](refined)
            
            if score < best_score:
                best_score = score
                best_assignment = refined
        except Exception:
            # If refinement fails, revert to previous best
            break
            
    return {'assignment': best_assignment}