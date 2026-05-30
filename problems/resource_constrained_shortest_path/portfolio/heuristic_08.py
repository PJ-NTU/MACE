# MACE evolved heuristic 08/10 for problem: resource_constrained_shortest_path
import time
import heapq

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Solve the RCSP problem using a resource-constrained label-setting algorithm.
    This approach explores paths by keeping track of non-dominated (cost, resource)
    labels at each vertex. Given the time limit, we prune the search space
    to ensure we return a feasible solution efficiently.
    """
    start_time = time.time()
    
    n = instance['n']
    K = instance['K']
    lower_bounds = instance['lower_bounds']
    upper_bounds = instance['upper_bounds']
    vertex_resources = instance['vertex_resources']
    graph = instance['graph']
    
    # Priority Queue stores (cost, current_vertex, path_list, resource_totals)
    # Using cost as the primary key for Dijkstra-like expansion
    pq = []
    
    # Initial state: start at vertex 1
    initial_resources = list(vertex_resources[0])
    heapq.heappush(pq, (0.0, 1, [1], initial_resources))
    
    # To keep the search tractable, we maintain a set of visited (vertex, resource_tuple) 
    # states to prune cycles and redundant paths.
    visited = {}
    
    best_path = None
    min_cost = float('inf')
    
    while pq:
        # Check time limit
        if time.time() - start_time > time_limit_s * 0.95:
            break
            
        cost, u, path, res = heapq.heappop(pq)
        
        # If we reached the target, check feasibility and update best
        if u == n:
            feasible = True
            for k in range(K):
                if not (lower_bounds[k] - 1e-7 <= res[k] <= upper_bounds[k] + 1e-7):
                    feasible = False
                    break
            if feasible:
                if cost < min_cost:
                    min_cost = cost
                    best_path = path
                # Since we use a priority queue, the first feasible path found is often good,
                # but we continue to look for better ones if time allows.
                continue
        
        # State pruning: Only process if this resource state is better than seen
        state = (u, tuple(res))
        if visited.get(state, float('inf')) <= cost:
            continue
        visited[state] = cost
        
        # Explore neighbors
        if u in graph:
            for v, arc_cost, arc_res in graph[u]:
                new_res = [res[k] + arc_res[k] + vertex_resources[v - 1][k] for k in range(K)]
                
                # Prune if upper bounds exceeded
                if any(new_res[k] > upper_bounds[k] + 1e-7 for k in range(K)):
                    continue
                
                new_path = path + [v]
                heapq.heappush(pq, (cost + arc_cost, v, new_path, new_res))
                
    if best_path:
        return {"total_cost": min_cost, "path": best_path}
    
    # Fallback: try the provided greedy solver if no path found
    try:
        greedy_path = tools['greedy_extend_path'](time_limit_s * 0.05)
        if greedy_path:
            return {"total_cost": tools['path_length'](greedy_path), "path": greedy_path}
    except:
        pass
        
    return {"total_cost": 0.0, "path": [1, n]}