# MACE evolved heuristic 03/10 for problem: euclidean_steiner_problem
import time
import math
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Dispatcher heuristic for the Euclidean Steiner Problem.
    
    Strategy Analysis:
    - Parent A (Coarse-to-fine descent) excels on dense, large-scale point sets 
      where global structure is complex and requires multi-scale refinement.
    - Parent B (Stochastic local search) excels on sparse or highly irregular 
      point sets where escaping local optima via perturbation is critical.
      
    Decision logic:
    - Use the density (mean distance between terminals) vs. the number of terminals
      to classify the instance. Larger, denser instances favor Parent A's structured
      descent, while sparse/small instances favor Parent B's exploratory approach.
    """
    start_time = time.time()
    points = instance.get("points", [])
    n = len(points)
    
    if n < 3:
        return {"steiner_points": []}

    # Calculate density metric
    # Average distance between points as a proxy for density
    def get_avg_dist():
        if n < 2: return 0
        dists = []
        for i in range(min(n, 20)):
            for j in range(i + 1, min(n, 20)):
                dists.append(tools['distance'](points[i], points[j]))
        return sum(dists) / len(dists) if dists else 1.0

    avg_d = get_avg_dist()
    density_score = n / (avg_d + 1e-6)
    
    # Threshold for regime:
    # If density_score is high, use multi-scale descent (A)
    # If density_score is low, use stochastic perturbation (B)
    if density_score > 50:
        # Parent A implementation
        steiner_points = tools['add_fermat_points_for_mst_triples'](min_improvement=1e-10)
        scales = [0.2, 0.1, 0.05, 0.02, 0.005]
        for i, step in enumerate(scales):
            if time.time() - start_time > time_limit_s * 0.9:
                break
            steiner_points = tools['local_relocate_steiner'](
                steiner_points=steiner_points,
                time_limit_s=max(0.01, (time_limit_s * 0.9 - (time.time() - start_time)) / (len(scales) - i)),
                step=step
            )
        return tools['make_solution'](steiner_points)
    else:
        # Parent B implementation
        try:
            best_steiner = tools['add_fermat_points_for_mst_triples'](min_improvement=1e-8)
        except:
            best_steiner = []
        best_mst = tools['mst_length'](best_steiner)
        step = 0.05
        while time.time() - start_time < time_limit_s * 0.85:
            refined = tools['local_relocate_steiner'](best_steiner, time_limit_s=0.2, step=step)
            current_mst = tools['mst_length'](refined)
            if current_mst < best_mst - 1e-9:
                best_mst, best_steiner, step = current_mst, list(refined), min(0.1, step * 1.1)
            elif best_steiner:
                idx = random.randrange(len(best_steiner))
                p = best_steiner[idx]
                best_steiner[idx] = (p[0] + random.uniform(-step, step), p[1] + random.uniform(-step, step))
                if tools['mst_length'](best_steiner) < best_mst:
                    best_mst = tools['mst_length'](best_steiner)
                else:
                    best_steiner[idx] = p
                step *= 0.95
            if step < 1e-5: step = 0.05
        
        sol = tools['make_solution'](best_steiner)
        is_valid, _ = tools['is_feasible'](sol)
        return sol if is_valid else {"steiner_points": []}