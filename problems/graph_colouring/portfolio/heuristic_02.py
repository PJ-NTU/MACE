# MACE evolved heuristic 02/10 for problem: graph_colouring
import time
import random
import heapq

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    A Tabu-Search inspired approach for Graph Coloring.
    
    The portfolio members primarily rely on constructive heuristics (DSATUR) 
    followed by local optimization (recoloring). This heuristic departs by 
    using a state-space search that allows temporary invalid states (conflicts) 
    to traverse the search space effectively, using a Tabu list to prevent 
    cycling, which is a common blind spot in purely constructive/local-refinement 
    approaches.
    """
    start_time = time.time()
    n = tools['n_vertices']()
    
    # Start with a valid coloring to have a baseline
    current_coloring = tools['dsatur_color']()
    current_score = len(tools['colors_used'](current_coloring))
    
    # Try to reduce k by forcing a removal of one color class
    best_k = current_score
    best_coloring = current_coloring.copy()
    
    # Tabu Search parameters
    tabu_list = {}
    tabu_tenure = 10
    iteration = 0
    
    # We attempt to reduce the number of colors by forcing a reduction and then
    # repairing the resulting conflicts using a min-conflicts-like heuristic.
    while time.time() - start_time < time_limit_s - 0.5:
        # If we have a valid solution with k colors, try to find one with k-1
        if best_k > 2:
            target_k = best_k - 1
            # Force reduction: randomly reassign vertices of one color to a smaller color
            # This creates an invalid state (conflicts)
            working_coloring = best_coloring.copy()
            colors = sorted(list(tools['colors_used'](working_coloring)))
            color_to_remove = colors[-1]
            for v in working_coloring:
                if working_coloring[v] == color_to_remove:
                    working_coloring[v] = random.randint(1, target_k)
            
            # Local search to repair conflicts (Min-conflicts)
            for repair_step in range(500):
                conflicts = tools['color_conflicts'](working_coloring)
                if not conflicts:
                    best_k = target_k
                    best_coloring = working_coloring.copy()
                    break
                
                # Pick a random conflicting edge
                u, v = random.choice(conflicts)
                # Pick one of the two endpoints to move
                target_v = u if random.random() < 0.5 else v
                
                # Find color with minimum conflicts
                best_c = working_coloring[target_v]
                min_c_conflicts = float('inf')
                
                # Try all colors 1..target_k
                potential_colors = list(range(1, target_k + 1))
                random.shuffle(potential_colors)
                
                for c in potential_colors:
                    if (target_v, c) in tabu_list and tabu_list[(target_v, c)] > iteration:
                        continue
                    
                    c_conflicts = 0
                    for neighbor in tools['adjacency'](target_v):
                        if working_coloring[neighbor] == c:
                            c_conflicts += 1
                    
                    if c_conflicts < min_c_conflicts:
                        min_c_conflicts = c_conflicts
                        best_c = c
                
                working_coloring[target_v] = best_c
                tabu_list[(target_v, best_c)] = iteration + tabu_tenure
                iteration += 1
                
                if time.time() - start_time > time_limit_s - 0.5:
                    break
        else:
            break
            
    return best_coloring