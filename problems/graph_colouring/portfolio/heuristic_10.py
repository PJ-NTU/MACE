# MACE evolved heuristic 10/10 for problem: graph_colouring
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Synthesized hybrid solver:
    - Low-density/Small graphs: Use DSATUR + Recolor (Constructive refinement).
    - High-density/Large graphs: Use Tabu-driven conflict minimization.
    
    Logic:
    - Sparse graphs have smaller chromatic numbers relative to max degree, making
      greedy-based constructive methods highly effective.
    - Dense graphs form tighter cliques, requiring local search to resolve
      conflicts when trying to reduce the number of colors.
    """
    start_time = time.time()
    n = tools['n_vertices']()
    m = tools['n_edges']()
    
    # Calculate density relative to a complete graph
    # Density = 2m / (n*(n-1))
    density = (2 * m) / (n * (n - 1)) if n > 1 else 0
    
    def get_time_left():
        return time_limit_s - (time.time() - start_time)

    # Constructive Refinement Strategy (A-style)
    def run_strategy_a():
        best_coloring = tools['dsatur_color']()
        best_coloring = tools['recolor_to_minimize_colors'](best_coloring, max_passes=100)
        best_score = len(tools['colors_used'](best_coloring))
        
        while get_time_left() > 0.3:
            # Perturbative Greedy
            nodes = list(range(1, n + 1))
            nodes.sort(key=lambda x: tools['degree'](x), reverse=True)
            # Shuffle a small portion to explore different greedy paths
            shuffle_range = min(n, max(2, n // 20))
            for i in range(shuffle_range):
                j = random.randint(i, n - 1)
                nodes[i], nodes[j] = nodes[j], nodes[i]
            
            current_coloring = tools['greedy_color'](order=nodes)
            current_coloring = tools['recolor_to_minimize_colors'](current_coloring, max_passes=50)
            
            if len(tools['colors_used'](current_coloring)) < best_score:
                best_coloring = current_coloring
                best_score = len(tools['colors_used'](best_coloring))
        return best_coloring

    # Tabu Search Strategy (B-style)
    def run_strategy_b():
        best_coloring = tools['dsatur_color']()
        best_k = len(tools['colors_used'](best_coloring))
        
        while get_time_left() > 0.5 and best_k > 2:
            target_k = best_k - 1
            working = best_coloring.copy()
            # Force reduction: reassign vertices of max color to random target colors
            max_c = max(tools['colors_used'](working))
            for v in range(1, n + 1):
                if working[v] == max_c:
                    working[v] = random.randint(1, target_k)
            
            tabu = {}
            for iteration in range(1000):
                if get_time_left() < 0.2: break
                
                conflicts = tools['color_conflicts'](working)
                if not conflicts:
                    best_coloring = working.copy()
                    best_k = target_k
                    break
                
                u, v = random.choice(conflicts)
                target = u if random.random() < 0.5 else v
                
                best_c = working[target]
                min_conf = float('inf')
                candidates = list(range(1, target_k + 1))
                random.shuffle(candidates)
                
                for c in candidates:
                    if tabu.get((target, c), 0) > iteration: continue
                    c_conf = sum(1 for neighbor in tools['adjacency'](target) if working[neighbor] == c)
                    if c_conf < min_conf:
                        min_conf = c_conf
                        best_c = c
                
                working[target] = best_c
                tabu[(target, best_c)] = iteration + 10
            else:
                break
        return best_coloring

    # Heuristic decision boundary
    # If graph is sparse or very small, constructive search is faster and more reliable.
    # Otherwise, use tabu search to handle the dense conflict landscape.
    if density < 0.25 or n < 60:
        return run_strategy_a()
    else:
        return run_strategy_b()