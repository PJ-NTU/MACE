# MACE evolved heuristic 10/10 for problem: maximal_independent_set
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Maximal Independent Set solver.
    
    Strategy:
    1. Warm start using the ILP solver for a high-quality backbone.
    2. Use a prioritized randomized construction phase (GRASP) that balances
       greedy choices with stochastic exploration.
    3. Apply aggressive local search techniques (2,1-swaps and randomized
       perturbation) to escape local optima.
    """
    start_time = time.time()
    graph = instance['graph']
    
    # 1. Warm Start: ILP for small graphs or greedy for large ones
    # We allocate 25% of time to the exact solver.
    best_mis = tools['greedy_min_degree']()
    
    if len(graph.nodes()) < 600:
        ilp_res = tools['ilp_max_independent_set'](time_limit_s=max(1.0, time_limit_s * 0.25))
        if ilp_res and len(ilp_res) > len(best_mis):
            best_mis = ilp_res

    # 2. Iterative Improvement Loop
    # We maintain a robust best solution and repeatedly attempt to improve it
    # using a combination of construction and local search.
    while time.time() - start_time < time_limit_s * 0.9:
        # Construction: Randomized Greedy
        # MODIFICATION: Increased exploration by allowing candidates within 
        # min_deg + 2 to broaden the search space on complex topologies.
        current_nodes = set(graph.nodes())
        candidate_mis = []
        
        while current_nodes:
            # Degree calculation in residual graph
            degrees = {v: len([n for n in graph.neighbors(v) if n in current_nodes]) 
                       for v in current_nodes}
            
            min_deg = min(degrees.values())
            # Restricted Candidate List (RCL)
            candidates = [v for v, d in degrees.items() if d <= min_deg + 2]
            
            chosen = random.choice(candidates)
            candidate_mis.append(chosen)
            
            # Remove chosen and its neighbors
            current_nodes.remove(chosen)
            for neighbor in graph.neighbors(chosen):
                if neighbor in current_nodes:
                    current_nodes.remove(neighbor)
        
        # Refinement: 2-for-1 swaps are highly effective for MIS
        improved = tools['apply_swap_2_for_1'](candidate_mis)
        
        if len(improved) > len(best_mis):
            best_mis = improved
        
        # Periodic local search intensification
        # We favor more frequent local searches than h_a to refine the current best
        if len(best_mis) > 0 and random.random() < 0.2:
            best_mis = tools['apply_local_swap'](best_mis, t_limit=min(0.2, time_limit_s * 0.02))
            
    # Final intensification
    best_mis = tools['apply_local_swap'](best_mis, t_limit=max(0.1, time_limit_s * 0.05))
    
    return {"mis_nodes": list(best_mis)}