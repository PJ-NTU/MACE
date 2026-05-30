# MACE evolved heuristic 04/10 for problem: graph_colouring
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Dispatcher-style solver for graph coloring.
    
    Hypothesis: 
    - Parent A (20 passes) is more robust for smaller, denser graphs where 
      local search needs to switch strategies quickly to avoid getting stuck.
    - Parent B (100 passes) is superior for larger or sparser graphs where 
      the color-elimination heuristic has more room to maneuver and benefits 
      from deeper, more exhaustive local passes.
    """
    start_time = time.time()
    
    n = tools['n_vertices']()
    m = tools['n_edges']()
    # Density: edges / max_possible_edges (n*(n-1)/2)
    density = (2 * m) / (n * (n - 1)) if n > 1 else 0
    
    # Dispatch Logic:
    # If the graph is relatively large (n > 100) or sparse (density < 0.1),
    # use the more intensive B-style (100 passes).
    # Otherwise, use the more agile A-style (20 passes).
    is_large_or_sparse = (n > 100) or (density < 0.1)
    max_passes = 100 if is_large_or_sparse else 20
    
    best_coloring = tools['dsatur_color']()
    
    def get_remaining_time():
        return time_limit_s - (time.time() - start_time)

    # Iterative refinement loop
    while get_remaining_time() > 0.5:
        current_colors = len(tools['colors_used'](best_coloring))
        
        # Apply the chosen refinement intensity
        refined = tools['recolor_to_minimize_colors'](best_coloring, max_passes=max_passes)
        
        new_colors = len(tools['colors_used'](refined))
        
        if new_colors < current_colors:
            best_coloring = refined
        else:
            # Perturbation phase: attempt to jump out of local optima
            # Pick a vertex with a high degree to maximize impact of change
            v = random.randint(1, n)
            neighbors = tools['adjacency'](v)
            neighbor_colors = {best_coloring[n] for n in neighbors}
            all_used = tools['colors_used'](best_coloring)
            possible_colors = [c for c in all_used if c not in neighbor_colors]
            
            if possible_colors:
                new_c = random.choice(possible_colors)
                new_coloring = tools['apply_recolor_vertex'](best_coloring, v, new_c)
                if new_coloring:
                    best_coloring = new_coloring
            else:
                # If no improvement possible after perturbation, terminate early
                break
                
    return best_coloring