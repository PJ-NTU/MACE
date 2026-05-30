# MACE evolved heuristic 08/10 for problem: graph_colouring
import time
import random
import math

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Dispatcher-based graph coloring heuristic.
    
    Strategy:
    - Small/Sparse graphs: Benefit from high-quality constructive initialization
      (DSATUR) followed by aggressive local refinement (recolor_to_minimize_colors).
    - Large/Dense graphs: Benefit from a Tabu-search based min-conflicts framework 
      that explores infeasible space to escape local optima in highly constrained environments.
    """
    start_time = time.time()
    n = tools['n_vertices']()
    m = tools['n_edges']()
    
    # Calculate density and structural complexity
    # Density: 2*m / (n*(n-1))
    # Threshold 0.15 is chosen as a heuristic boundary for constraint tightness
    density = (2 * m) / (n * (n - 1)) if n > 1 else 0
    is_dense = density > 0.15 or (n > 100 and m / n > 5)

    def run_sparse_refinement():
        coloring = tools['dsatur_color']()
        while time.time() - start_time < time_limit_s - 0.2:
            current_colors = len(tools['colors_used'](coloring))
            refined = tools['recolor_to_minimize_colors'](coloring, max_passes=100)
            if len(tools['colors_used'](refined)) < current_colors:
                coloring = refined
            else:
                # Perturbation to escape local optima
                v = random.randint(1, n)
                all_used = list(tools['colors_used'](coloring))
                random.shuffle(all_used)
                for c in all_used:
                    new_coloring = tools['apply_recolor_vertex'](coloring, v, c)
                    if new_coloring:
                        coloring = new_coloring
                        break
                break
        return coloring

    def run_dense_tabu():
        best_coloring = tools['dsatur_color']()
        best_k = len(tools['colors_used'](best_coloring))
        tabu_list = {}
        tabu_tenure = max(5, int(n * 0.1))
        iteration = 0
        
        while time.time() - start_time < time_limit_s - 0.2:
            if best_k <= 2:
                break
            
            target_k = best_k - 1
            working_coloring = best_coloring.copy()
            # Force reduction
            colors = sorted(list(tools['colors_used'](working_coloring)))
            color_to_remove = colors[-1]
            for v in working_coloring:
                if working_coloring[v] == color_to_remove:
                    working_coloring[v] = random.randint(1, target_k)
            
            # Local search to repair
            for _ in range(500):
                conflicts = tools['color_conflicts'](working_coloring)
                if not conflicts:
                    best_k = target_k
                    best_coloring = working_coloring.copy()
                    break
                
                u, v = random.choice(conflicts)
                target_v = u if random.random() < 0.5 else v
                
                best_c = working_coloring[target_v]
                min_c_conflicts = float('inf')
                
                indices = list(range(1, target_k + 1))
                random.shuffle(indices)
                for c in indices:
                    if tabu_list.get((target_v, c), 0) > iteration:
                        continue
                    c_conflicts = sum(1 for neighbor in tools['adjacency'](target_v) if working_coloring[neighbor] == c)
                    if c_conflicts < min_c_conflicts:
                        min_c_conflicts = c_conflicts
                        best_c = c
                
                working_coloring[target_v] = best_c
                tabu_list[(target_v, best_c)] = iteration + tabu_tenure
                iteration += 1
                if time.time() - start_time > time_limit_s - 0.2:
                    break
        return best_coloring

    if not is_dense:
        return run_sparse_refinement()
    else:
        return run_dense_tabu()