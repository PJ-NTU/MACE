# MACE evolved heuristic 01/10 for problem: graph_colouring
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Solve graph coloring using a combination of DSATUR, local search refinement,
    and iterative recoloring to minimize the number of colors.
    """
    start_time = time.time()
    
    # 1. Generate a strong initial solution using DSATUR
    # DSATUR is efficient and usually provides a near-optimal coloring.
    best_coloring = tools['dsatur_color']()
    
    # 2. Attempt to reduce the number of colors using recolor_to_minimize_colors
    # This tool is specifically designed to eliminate color classes.
    # We allocate time for this based on the remaining budget.
    def get_remaining_time():
        return time_limit_s - (time.time() - start_time)

    # Perform iterative refinement
    # We use a loop to apply the optimization until time is nearly up.
    # recolor_to_minimize_colors is quite efficient (O(passes * n_edges)).
    while get_remaining_time() > 0.5:
        # Save current best
        current_colors = len(tools['colors_used'](best_coloring))
        
        # Try to refine
        refined = tools['recolor_to_minimize_colors'](best_coloring, max_passes=20)
        
        new_colors = len(tools['colors_used'](refined))
        
        if new_colors < current_colors:
            best_coloring = refined
        else:
            # If no improvement, try a randomized small perturbation 
            # (Tabu-like move) to jump out of local optima.
            # Pick a random vertex and attempt to change its color to a 
            # color not currently in its neighborhood.
            v = random.randint(1, tools['n_vertices']())
            neighbors = tools['adjacency'](v)
            neighbor_colors = {best_coloring[n] for n in neighbors}
            
            # Find a color that is currently used but not in neighbors
            all_used = tools['colors_used'](best_coloring)
            possible_colors = [c for c in all_used if c not in neighbor_colors]
            
            if possible_colors:
                new_c = random.choice(possible_colors)
                new_coloring = tools['apply_recolor_vertex'](best_coloring, v, new_c)
                if new_coloring:
                    best_coloring = new_coloring
            else:
                # If we cannot improve, break to return the current best
                break
                
    return best_coloring