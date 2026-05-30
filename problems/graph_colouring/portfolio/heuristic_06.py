# MACE evolved heuristic 06/10 for problem: graph_colouring
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Dispatcher-style solver for Graph Colouring.
    
    Hypothesis: 
    - DSATUR + Recolor (Parent A) is highly effective for sparse to medium-density 
      graphs where the chromatic number is relatively low and local search 
      plateaus are easier to escape via color class elimination.
    - Tabu-Search (Parent B) excels on dense graphs or hard instances where 
      the search space is extremely constrained and requires traversing 
      temporarily infeasible states to find a valid k-coloring.
    
    Decision:
    Density = 2 * |E| / (n * (n - 1))
    If density > 0.35, use Tabu-Search (B).
    Otherwise, use DSATUR + Recolor (A).
    """
    start_time = time.time()
    n = tools['n_vertices']()
    m = tools['n_edges']()
    
    # Calculate density safely
    if n > 1:
        density = (2 * m) / (n * (n - 1))
    else:
        density = 0
        
    # Strategy A: DSATUR + Recolor (Best for sparse graphs)
    def run_strategy_a():
        best_solution = tools['dsatur_color']()
        while time.time() - start_time < time_limit_s * 0.9:
            prev_solution = best_solution.copy()
            best_solution = tools['recolor_to_minimize_colors'](best_solution, max_passes=50)
            if len(tools['colors_used'](best_solution)) >= len(tools['colors_used'](prev_solution)):
                break
        return best_solution

    # Strategy B: Tabu-Search (Best for dense graphs)
    def run_strategy_b():
        current_coloring = tools['dsatur_color']()
        best_coloring = current_coloring.copy()
        best_k = len(tools['colors_used'](best_coloring))
        tabu_list = {}
        tabu_tenure = 10
        iteration = 0
        
        while time.time() - start_time < time_limit_s - 0.5:
            if best_k > 2:
                target_k = best_k - 1
                working_coloring = best_coloring.copy()
                # Force reduction to target_k
                for v in working_coloring:
                    if working_coloring[v] > target_k:
                        working_coloring[v] = random.randint(1, target_k)
                
                # Repair phase
                for _ in range(300):
                    conflicts = tools['color_conflicts'](working_coloring)
                    if not conflicts:
                        best_k = target_k
                        best_coloring = working_coloring.copy()
                        break
                    
                    u, v = random.choice(conflicts)
                    target_v = u if random.random() < 0.5 else v
                    
                    # Min-conflicts with Tabu
                    best_c = working_coloring[target_v]
                    min_c_conflicts = float('inf')
                    potential_colors = list(range(1, target_k + 1))
                    random.shuffle(potential_colors)
                    
                    for c in potential_colors:
                        if tabu_list.get((target_v, c), 0) > iteration:
                            continue
                        c_conflicts = sum(1 for neighbor in tools['adjacency'](target_v) 
                                          if working_coloring[neighbor] == c)
                        if c_conflicts < min_c_conflicts:
                            min_c_conflicts = c_conflicts
                            best_c = c
                    
                    working_coloring[target_v] = best_c
                    tabu_list[(target_v, best_c)] = iteration + tabu_tenure
                    iteration += 1
            else:
                break
        return best_coloring

    # Dispatch
    if density > 0.35:
        return run_strategy_b()
    else:
        return run_strategy_a()