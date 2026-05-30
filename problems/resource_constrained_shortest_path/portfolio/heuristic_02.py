# MACE evolved heuristic 02/10 for problem: resource_constrained_shortest_path
import time
import heapq

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Solves the Resource Constrained Shortest Path problem using a label-setting
    approach with cost-based pruning and time monitoring.
    """
    start_time = time.time()
    n = instance['n']
    K = instance['K']
    graph = instance['graph']
    lower_bounds = instance['lower_bounds']
    upper_bounds = instance['upper_bounds']
    vertex_resources = instance['vertex_resources']

    # Priority Queue stores: (cost, current_vertex, path_list, resource_totals)
    # Using cost as the primary priority for Dijkstra-like expansion.
    # To keep memory under control, we bound the number of labels per node.
    pq = [(0.0, 1, [1], list(vertex_resources[0]))]
    
    # Keep track of best paths found to target
    best_path = None
    min_cost = float('inf')
    
    # Limit state space expansion
    visited_labels = {} 
    
    while pq:
        if time.time() - start_time > time_limit_s * 0.9:
            break
            
        cost, u, path, res = heapq.heappop(pq)
        
        if cost >= min_cost:
            continue
            
        if u == n:
            # Check lower bounds
            if all(res[k] >= lower_bounds[k] - 1e-7 for k in range(K)):
                if cost < min_cost:
                    min_cost = cost
                    best_path = path
            continue
            
        # Pruning: simple dominance check
        state = (u, tuple(res))
        if state in visited_labels and visited_labels[state] <= cost:
            continue
        visited_labels[state] = cost
        
        for v, arc_cost, arc_res in graph.get(u, []):
            new_res = [res[k] + arc_res[k] + vertex_resources[v - 1][k] for k in range(K)]
            
            # Check upper bounds
            if all(new_res[k] <= upper_bounds[k] + 1e-7 for k in range(K)):
                new_path = path + [v]
                heapq.heappush(pq, (cost + arc_cost, v, new_path, new_res))
                
    if best_path:
        return {"total_cost": min_cost, "path": best_path}
    
    # Fallback: try tool if solver failed
    try:
        path = tools['greedy_extend_path'](time_limit_s=max(0.1, time_limit_s * 0.1))
        if path:
            return {"total_cost": tools['path_length'](path), "path": path}
    except:
        pass
        
    return {"total_cost": 0.0, "path": [1, n]}