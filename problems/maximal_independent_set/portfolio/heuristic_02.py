# MACE evolved heuristic 02/10 for problem: maximal_independent_set
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Solves the Maximum Independent Set problem using a GRASP-inspired 
    multi-start approach: randomized greedy construction followed by 
    local improvement.
    """
    start_time = time.time()
    graph = instance['graph']
    nodes = list(graph.nodes())
    
    best_mis = []
    best_size = -1
    
    # Heuristic strategy: 
    # 1. Use the pre-built greedy tools as a strong baseline.
    # 2. Perform randomized greedy constructions if time allows.
    # 3. Apply local swap refinement to the best candidate found.
    
    def get_randomized_greedy():
        """Constructs an independent set using randomized min-degree selection."""
        candidates = set(nodes)
        mis = []
        while candidates:
            # Calculate degrees in the residual subgraph
            # Optimization: only recompute degrees for affected nodes
            curr_degrees = {v: sum(1 for neighbor in graph.neighbors(v) if neighbor in candidates) 
                            for v in candidates}
            
            min_deg = min(curr_degrees.values())
            # GRASP: pick from candidates with degree <= min_deg + 1 (randomized)
            choices = [v for v, deg in curr_degrees.items() if deg <= min_deg + 1]
            chosen = random.choice(choices)
            
            mis.append(chosen)
            # Remove chosen and its neighbors
            candidates.remove(chosen)
            for neighbor in graph.neighbors(chosen):
                if neighbor in candidates:
                    candidates.remove(neighbor)
        return mis

    # Try deterministic greedy first
    try:
        greedy_base = tools['greedy_min_degree']()
        if len(greedy_base) > best_size:
            best_mis = greedy_base
            best_size = len(best_mis)
    except:
        pass

    # Iterative improvement loop
    while time.time() - start_time < time_limit_s * 0.8:
        candidate = get_randomized_greedy()
        
        # Local refinement
        try:
            refined = tools['apply_local_swap'](candidate, t_limit=0.1)
            current_size = len(refined)
        except:
            refined = candidate
            current_size = len(candidate)
            
        if current_size > best_size:
            best_size = current_size
            best_mis = refined
            
        if time.time() - start_time > time_limit_s * 0.9:
            break
            
    return {"mis_nodes": best_mis}