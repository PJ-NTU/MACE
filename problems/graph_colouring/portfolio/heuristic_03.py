# MACE evolved heuristic 03/10 for problem: graph_colouring
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Dispatcher-style solver for Graph Coloring.
    
    Hypothesis:
    - Sparse graphs (low density) or small graphs benefit from the refined, 
      constructive search of Parent A (DSATUR + recoloring), which is highly
      efficient at finding near-optimal colorings quickly.
    - Dense graphs (high edge density) benefit from the Tabu-Search/Min-conflicts 
      approach of Parent B, which can break out of local optima in highly 
      constrained search spaces where greedy construction fails.
    """
    start_time = time.time()
    n = tools['n_vertices']()
    m = tools['n_edges']()
    
    # Density calculation: m / (n * (n - 1) / 2)
    # Simplified to m / n^2 for efficiency
    density = (2 * m) / (n * (n - 1)) if n > 1 else 0
    
    # Strategy A: Constructive Refinement (DSATUR + Recolor)
    def strategy_a():
        best_coloring = tools['dsatur_color']()
        best_coloring = tools['recolor_to_minimize_colors'](best_coloring, max_passes=100)
        best_score = len(tools['colors_used'](best_coloring))
        
        while time.time() - start_time < time_limit_s * 0.9:
            nodes = list(range(1, n + 1))
            nodes.sort(key=lambda x: tools['degree'](x), reverse=True)
            # Add small stochastic noise to ordering
            for _ in range(max(1, n // 10)):
                i, j = random.sample(range(n), 2)
                nodes[i], nodes[j] = nodes[j], nodes[i]
            
            current_coloring = tools['greedy_color'](order=nodes)
            current_coloring = tools['recolor_to_minimize_colors'](current_coloring, max_passes=30)
            
            if len(tools['colors_used'](current_coloring)) < best_score:
                best_coloring = current_coloring
                best_score = len(tools['colors_used'](current_coloring))
        return best_coloring

    # Strategy B: Local Search (Tabu / Min-Conflicts)
    def strategy_b():
        current_coloring = tools['dsatur_color']()
        best_k = len(tools['colors_used'](current_coloring))
        best_coloring = current_coloring.copy()
        tabu_list = {}
        tabu_tenure = 10
        iteration = 0
        
        while time.time() - start_time < time_limit_s * 0.9:
            if best_k <= 2: break
            
            target_k = best_k - 1
            working_coloring = best_coloring.copy()
            colors = sorted(list(tools['colors_used'](working_coloring)))
            color_to_remove = colors[-1]
            for v in working_coloring:
                if working_coloring[v] == color_to_remove:
                    working_coloring[v] = random.randint(1, target_k)
            
            for _ in range(300):
                conflicts = tools['color_conflicts'](working_coloring)
                if not conflicts:
                    best_k = target_k
                    best_coloring = working_coloring.copy()
                    break
                
                u, v = random.choice(conflicts)
                target_v = u if random.random() < 0.5 else v
                
                best_c = working_coloring[target_v]
                min_c_conflicts = float('inf')
                
                potential_colors = list(range(1, target_k + 1))
                random.shuffle(potential_colors)
                
                for c in potential_colors:
                    if tabu_list.get((target_v, c), 0) > iteration: continue
                    c_conflicts = sum(1 for neighbor in tools['adjacency'](target_v) 
                                      if working_coloring[neighbor] == c)
                    if c_conflicts < min_c_conflicts:
                        min_c_conflicts = c_conflicts
                        best_c = c
                
                working_coloring[target_v] = best_c
                tabu_list[(target_v, best_c)] = iteration + tabu_tenure
                iteration += 1
        return best_coloring

    # Dispatch logic
    if density < 0.3 or n < 50:
        return strategy_a()
    else:
        return strategy_b()