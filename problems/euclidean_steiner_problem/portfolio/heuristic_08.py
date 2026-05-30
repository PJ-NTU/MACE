# MACE evolved heuristic 08/10 for problem: euclidean_steiner_problem
import time
import math
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Dispatcher heuristic for the Euclidean Steiner Problem.
    
    Strategy Analysis:
    - Large-scale instances (N > 50) have a large solution space where 
      constructive Fermat seeding followed by deterministic coordinate 
      descent (Parent A) is more stable and prevents premature convergence.
    - Small/Sparse instances (N <= 50) benefit from the stochastic exploration 
      of Simulated Annealing (Parent B), which finds non-obvious Steiner 
      topologies that coordinate descent might miss.
    """
    start_time = time.time()
    points = instance.get("points", [])
    n = len(points)
    
    if n < 3:
        return {"steiner_points": []}

    # Hypothesis: N=50 is the inflection point where the landscape complexity 
    # transitions from needing stochastic escape (B) to structured refinement (A).
    if n > 50:
        # Parent A: Multi-scale Descent
        # Focuses on high-quality local refinement for complex point sets.
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
        # Parent B: Simulated Annealing
        # Focuses on global exploration for smaller sets where local optima are traps.
        initial_steiner = tools['add_fermat_points_for_mst_triples'](min_improvement=1e-9)
        current_steiner = list(initial_steiner)
        current_len = tools['mst_length'](current_steiner)
        best_steiner = list(current_steiner)
        best_len = current_len
        
        min_x = min(p[0] for p in points)
        max_x = max(p[0] for p in points)
        min_y = min(p[1] for p in points)
        max_y = max(p[1] for p in points)
        bbox_size = math.sqrt((max_x - min_x)**2 + (max_y - min_y)**2)
        
        T = 0.05 * current_len
        cooling_rate = 0.9997
        
        while time.time() - start_time < time_limit_s * 0.8:
            idx = random.randrange(len(current_steiner)) if current_steiner else -1
            op = random.random()
            new_steiner = list(current_steiner)
            
            if op < 0.1 and len(new_steiner) < n:
                new_steiner.append((random.uniform(min_x, max_x), random.uniform(min_y, max_y)))
            elif op < 0.3 and idx != -1:
                new_steiner.pop(idx)
            elif idx != -1:
                p = new_steiner[idx]
                scale = bbox_size * 0.02 * (T / (0.05 * current_len + 1e-9) + 0.1)
                new_steiner[idx] = (p[0] + random.gauss(0, scale), p[1] + random.gauss(0, scale))
            else:
                continue

            new_len = tools['mst_length'](new_steiner)
            delta = new_len - current_len
            if delta < 0 or (T > 1e-9 and math.exp(-delta / T) > random.random()):
                current_steiner = new_steiner
                current_len = new_len
                if current_len < best_len:
                    best_len = current_len
                    best_steiner = list(current_steiner)
            T *= cooling_rate
            
        polished = tools['local_relocate_steiner'](
            best_steiner, 
            time_limit_s=max(0.05, time_limit_s - (time.time() - start_time)),
            step=0.01
        )
        return tools['make_solution'](steiner_points=polished)