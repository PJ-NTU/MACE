# MACE evolved heuristic 10/10 for problem: euclidean_steiner_problem
import time
import math
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    A robust Steiner point optimization heuristic using a multi-phase approach:
    1. Deterministic Fermat-point seeding for high-quality initial topology.
    2. Adaptive local search (Coordinate Descent) with multi-scale refinement.
    3. Stochastic hill-climbing to escape local optima by perturbing existing 
       Steiner configurations, balanced by a strict feasibility-aware budget.
    """
    start_time = time.time()
    points = instance.get("points", [])
    if len(points) < 3:
        return tools['make_solution'](steiner_points=[])

    # Phase 1: Seed with Fermat points
    # This captures the optimal local geometry for 3-terminal nodes.
    best_steiner = tools['add_fermat_points_for_mst_triples'](min_improvement=1e-12)
    best_len = tools['mst_length'](best_steiner)

    # Phase 2: Multi-Scale Coordinate Descent
    # Refine the initial points using increasingly granular steps.
    scales = [0.05, 0.01, 0.005, 0.001]
    
    # Phase 3: Randomized Hill Climbing
    # Periodically perturb the Steiner set to explore neighboring configurations,
    # followed by refinement.
    
    current_steiner = list(best_steiner)
    
    # Calculate bounding box for adaptive perturbation
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    diag = math.sqrt((max(xs) - min(xs))**2 + (max(ys) - min(ys))**2)
    
    iteration = 0
    while time.time() - start_time < time_limit_s * 0.85:
        # Every few iterations, apply a random perturbation to potentially
        # shift a Steiner point out of a suboptimal local basin.
        if iteration > 0 and len(current_steiner) > 0:
            idx = random.randrange(len(current_steiner))
            px, py = current_steiner[idx]
            # Small random jitter
            current_steiner[idx] = (px + random.gauss(0, diag * 0.02), 
                                    py + random.gauss(0, diag * 0.02))
        
        # Refine current state
        step = scales[min(iteration, len(scales) - 1)]
        refined = tools['local_relocate_steiner'](
            steiner_points=current_steiner,
            time_limit_s=max(0.02, (time_limit_s * 0.95 - (time.time() - start_time)) / 5.0),
            step=step
        )
        
        new_len = tools['mst_length'](refined)
        if new_len < best_len:
            best_len = new_len
            best_steiner = list(refined)
            current_steiner = list(refined)
        else:
            # Backtrack if perturbation didn't help
            current_steiner = list(best_steiner)
            
        iteration += 1
        
        # Safety break if no points can improve
        if not current_steiner:
            break

    return tools['make_solution'](steiner_points=best_steiner)