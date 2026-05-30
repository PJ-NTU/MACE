# MACE evolved heuristic 09/10 for problem: graph_colouring
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    An optimized graph coloring heuristic using a hybrid approach:
    1. DSATUR for high-quality initial construction.
    2. Iterative reduction via color-class elimination.
    3. Tabu-based local search for conflict resolution when attempting to 
       force a reduction in the number of colors.
    """
    start_time = time.time()
    n = tools['n_vertices']()
    
    # 1. Initial construction
    best_coloring = tools['dsatur_color']()
    best_k = len(tools['colors_used'](best_coloring))
    
    def get_time_left():
        return time_limit_s - (time.time() - start_time)

    # 2. Iterative Improvement Loop
    while get_time_left() > 0.5:
        # Attempt to improve via standard recoloring
        refined = tools['recolor_to_minimize_colors'](best_coloring, max_passes=50)
        new_k = len(tools['colors_used'](refined))
        
        if new_k < best_k:
            best_k = new_k
            best_coloring = refined
            continue
            
        # If no further improvement via simple recoloring, try aggressive reduction
        if best_k > 2:
            target_k = best_k - 1
            # Copy and force reduction
            working = best_coloring.copy()
            max_c = max(tools['colors_used'](working))
            for v in range(1, n + 1):
                if working[v] == max_c:
                    working[v] = random.randint(1, target_k)
            
            # Tabu search to resolve conflicts
            tabu = {}
            for iteration in range(500):
                if get_time_left() < 0.2:
                    break
                
                conflicts = tools['color_conflicts'](working)
                if not conflicts:
                    best_coloring = working.copy()
                    best_k = target_k
                    break
                
                # Pick a random conflict
                u, v = random.choice(conflicts)
                target = u if random.random() < 0.5 else v
                
                # Best move for target: minimize conflicts
                best_c = working[target]
                min_conf = float('inf')
                
                # Sample colors to check
                candidates = list(range(1, target_k + 1))
                random.shuffle(candidates)
                
                for c in candidates:
                    if tabu.get((target, c), 0) > iteration:
                        continue
                    
                    # Count conflicts for color c
                    c_conf = sum(1 for neighbor in tools['adjacency'](target) if working[neighbor] == c)
                    if c_conf < min_conf:
                        min_conf = c_conf
                        best_c = c
                
                working[target] = best_c
                tabu[(target, best_c)] = iteration + 7
            else:
                # If we failed to resolve, we just keep the best_coloring found so far
                pass
        else:
            # Cannot reduce further than 2
            break
            
    return best_coloring