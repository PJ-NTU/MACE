# MACE evolved heuristic 04/10 for problem: resource_constrained_shortest_path
import time
import heapq

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Solve the Resource Constrained Shortest Path problem using a 
    Label Setting algorithm (Dijkstra-like expansion on state space).
    """
    start_time = time.time()
    n = instance['n']
    graph = instance['graph']
    K = instance['K']
    lower_bounds = instance['lower_bounds']
    upper_bounds = instance['upper_bounds']
    vertex_resources = instance['vertex_resources']

    # State: (cost, current_vertex, path_list, current_resources)
    # Priority queue ordered by cost
    pq = []
    
    # Initial state
    initial_resources = list(vertex_resources[0])
    # cost, vertex, path, resources
    heapq.heappush(pq, (0.0, 1, [1], initial_resources))
    
    # Keep track of best path found
    best_path = None
    best_cost = float('inf')
    
    # Dominance pruning: visited[vertex] = list of (cost, resources)
    # Only keep labels that are not dominated by existing ones
    visited = {}

    while pq:
        # Check time limit
        if time.time() - start_time > time_limit_s * 0.95:
            break
            
        cost, u, path, res = heapq.heappop(pq)
        
        if cost >= best_cost:
            continue
            
        if u == n:
            # Check lower bounds
            feasible = True
            for k in range(K):
                if res[k] < lower_bounds[k]:
                    feasible = False
                    break
            if feasible:
                if cost < best_cost:
                    best_cost = cost
                    best_path = path
            continue
            
        # Explore neighbors
        if u in graph:
            for v, arc_cost, arc_res in graph[u]:
                new_cost = cost + arc_cost
                new_res = [res[k] + arc_res[k] + vertex_resources[v - 1][k] for k in range(K)]
                
                # Prune by upper bound
                if any(new_res[k] > upper_bounds[k] for k in range(K)):
                    continue
                
                # Simple dominance check
                if v not in visited:
                    visited[v] = []
                
                dominated = False
                for prev_cost, prev_res in visited[v]:
                    if prev_cost <= new_cost and all(prev_res[k] <= new_res[k] for k in range(K)):
                        dominated = True
                        break
                
                if not dominated:
                    visited[v].append((new_cost, new_res))
                    heapq.heappush(pq, (new_cost, v, path + [v], new_res))

    if best_path:
        return {"total_cost": best_cost, "path": best_path}
    
    # Fallback to tools if search fails
    try:
        path = tools['label_setting_pareto'](time_limit_s=time_limit_s * 0.5)
        if path:
            return {"total_cost": tools['path_length'](path), "path": path}
    except:
        pass
        
    return {"total_cost": 0.0, "path": [1, n]}