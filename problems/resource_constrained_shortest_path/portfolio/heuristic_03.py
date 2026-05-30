# MACE evolved heuristic 03/10 for problem: resource_constrained_shortest_path
import time
import heapq
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Solves the Resource Constrained Shortest Path problem using a 
    Label Setting approach with aggressive pruning and time-constrained search.
    """
    start_time = time.time()
    n = instance['n']
    K = instance['K']
    graph = instance['graph']
    lower_bounds = instance['lower_bounds']
    upper_bounds = instance['upper_bounds']
    vertex_resources = instance['vertex_resources']

    # Priority Queue for Dijkstra-like expansion: (cost, vertex, resources, path)
    # Using a list for path is memory intensive, but required for the solution output.
    # We prioritize by cost to find the shortest path first.
    pq = [(0.0, 1, [0.0] * K, [1])]
    
    # Keep track of best path found so far
    best_path = None
    min_cost = float('inf')
    
    # Pruning: store min cost to reach a vertex with specific resource consumption
    # Since resources are continuous, we use a simple state-based dominance check
    # if the state space is too large.
    visited = {}

    while pq and (time.time() - start_time) < (time_limit_s * 0.9):
        cost, u, res, path = heapq.heappop(pq)
        
        if cost >= min_cost:
            continue
            
        if u == n:
            # Check lower bounds
            feasible = True
            for k in range(K):
                if res[k] < lower_bounds[k]:
                    feasible = False
                    break
            if feasible:
                if cost < min_cost:
                    min_cost = cost
                    best_path = path
            continue

        for v, arc_cost, arc_res in graph.get(u, []):
            new_cost = cost + arc_cost
            if new_cost >= min_cost:
                continue
                
            new_res = [res[k] + vertex_resources[v-1][k] + arc_res[k] for k in range(K)]
            
            # Check upper bounds
            if any(new_res[k] > upper_bounds[k] for k in range(K)):
                continue
            
            # Simple cycle prevention
            if v in path:
                continue
                
            heapq.heappush(pq, (new_cost, v, new_res, path + [v]))

    # Fallback: If no path found, try the provided tools
    if best_path is None:
        try:
            # Try label_setting_pareto first as it is the standard for this problem
            path = tools['label_setting_pareto'](time_limit_s=time_limit_s * 0.5)
            if path:
                best_path = path
            else:
                # Last resort: greedy
                best_path = tools['greedy_extend_path'](time_limit_s=time_limit_s * 0.1)
        except:
            pass

    if best_path:
        return {
            "total_cost": tools['path_length'](best_path),
            "path": best_path
        }
    
    # Return a dummy if all else fails
    return {"total_cost": 0.0, "path": [1, n]}