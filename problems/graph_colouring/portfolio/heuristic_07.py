# MACE evolved heuristic 07/10 for problem: graph_colouring
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    A robust graph coloring solver utilizing a hybrid of DSATUR construction,
    iterative color reduction, and a Tabu-search local search mechanism.
    
    Strategy:
    1. Start with a high-quality DSATUR coloring.
    2. Repeatedly attempt to reduce the chromatic number using local recoloring.
    3. If improvement stalls, use a min-conflicts Tabu search to explore the
       feasibility landscape of (k-1) colorings.
    """
    start_time = time.time()
    
    n = tools['n_vertices']()
    
    # Initial constructive heuristic
    best_coloring = tools['dsatur_color']()
    best_k = len(tools['colors_used'](best_coloring))
    
    def get_time_left():
        return time_limit_s - (time.time() - start_time)

    # 1. Iterative refinement phase
    while get_time_left() > 0.2:
        prev_k = best_k
        best_coloring = tools['recolor_to_minimize_colors'](best_coloring, max_passes=100)
        best_k = len(tools['colors_used'](best_coloring))
        
        if best_k >= prev_k:
            break

    # 2. Tabu Search phase for further reduction
    # We attempt to find a valid coloring using best_k - 1 colors.
    tabu_list = {}
    tabu_tenure = 7
    iteration = 0
    
    while get_time_left() > 0.3 and best_k > 2:
        target_k = best_k - 1
        
        # Construct a starting point for target_k by mapping colors
        working_coloring = best_coloring.copy()
        for v in range(1, n + 1):
            if working_coloring[v] > target_k:
                working_coloring[v] = random.randint(1, target_k)
        
        # Min-conflicts local search within target_k
        for _ in range(500):
            if get_time_left() < 0.1:
                break
                
            conflicts = tools['color_conflicts'](working_coloring)
            if not conflicts:
                best_coloring = working_coloring.copy()
                best_k = target_k
                break
            
            # Select a random conflicting vertex
            u, v = random.choice(conflicts)
            target_v = u if random.random() < 0.5 else v
            
            # Choose color that minimizes conflicts (Tabu-aware)
            best_c = working_coloring[target_v]
            min_c_conflicts = float('inf')
            
            candidates = list(range(1, target_k + 1))
            random.shuffle(candidates)
            
            for c in candidates:
                # Check tabu status
                if tabu_list.get((target_v, c), 0) > iteration:
                    continue
                
                # Calculate conflicts for this color
                c_conflicts = sum(1 for neighbor in tools['adjacency'](target_v) 
                                  if working_coloring[neighbor] == c)
                
                if c_conflicts < min_c_conflicts:
                    min_c_conflicts = c_conflicts
                    best_c = c
            
            working_coloring[target_v] = best_c
            tabu_list[(target_v, best_c)] = iteration + tabu_tenure
            iteration += 1
            
    return best_coloring