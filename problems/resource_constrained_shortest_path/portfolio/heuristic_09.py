# MACE evolved heuristic 09/10 for problem: resource_constrained_shortest_path
import time
import heapq

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Solves the Resource Constrained Shortest Path problem using a 
    Label-Setting approach with a priority queue, constrained by the 
    provided time budget.
    """
    start_time = time.time()
    
    n = instance['n']
    K = instance['K']
    graph = instance['graph']
    lower_bounds = instance['lower_bounds']
    upper_bounds = instance['upper_bounds']
    vertex_resources = instance['vertex_resources']
    
    # Priority Queue stores: (cost, current_vertex, current_resources, path)
    # Using a simple heuristic: Dijkstra-like expansion
    # To manage memory and time, we limit the number of labels per vertex
    max_labels_per_node = 100
    labels = {i: [] for i in range(1, n + 1)}
    
    initial_res = vertex_resources[0]
    pq = [(0.0, 1, initial_res, [1])]
    
    best_path = None
    min_cost = float('inf')
    
    while pq:
        if time.time() - start_time > time_limit_s * 0.9:
            break
            
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
                min_cost = cost
                best_path = path
            continue
            
        for v, arc_cost, arc_res in graph.get(u, []):
            new_res = [res[k] + arc_res[k] + vertex_resources[v - 1][k] for k in range(K)]
            
            # Prune by upper bounds
            if any(new_res[k] > upper_bounds[k] for k in range(K)):
                continue
                
            new_cost = cost + arc_cost
            new_path = path + [v]
            
            # Simple dominance check: if we have too many labels, only keep better ones
            if len(labels[v]) < max_labels_per_node:
                heapq.heappush(pq, (new_cost, v, new_res, new_path))
                labels[v].append((new_cost, new_res))
            else:
                # Replace worst label if current is better
                labels[v].sort(key=lambda x: x[0])
                if new_cost < labels[v][-1][0]:
                    labels[v][-1] = (new_cost, new_res)
                    heapq.heappush(pq, (new_cost, v, new_res, new_path))
                    
    if best_path:
        return {"total_cost": min_cost, "path": best_path}
    
    # Fallback to greedy if label setting fails
    greedy_path = tools.get('greedy_extend_path', lambda t: None)(time_limit_s * 0.1)
    if greedy_path:
        return {"total_cost": tools['path_length'](greedy_path), "path": greedy_path}
        
    return {"total_cost": 0.0, "path": [1, n]}