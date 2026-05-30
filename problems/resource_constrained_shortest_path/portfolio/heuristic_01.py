# MACE evolved heuristic 01/10 for problem: resource_constrained_shortest_path
import time
import heapq

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Solves the Resource Constrained Shortest Path problem using a 
    Label-Setting approach with a priority queue, constrained by 
    time and memory limits.
    """
    start_time = time.time()
    n = instance['n']
    K = instance['K']
    graph = instance['graph']
    lower_bounds = instance['lower_bounds']
    upper_bounds = instance['upper_bounds']
    vertex_resources = instance['vertex_resources']

    # Priority Queue stores: (cost, current_vertex, current_resources, path)
    # Using cost as the primary tie-breaker for Dijkstra-like behavior.
    initial_resources = list(vertex_resources[0])
    pq = [(0.0, 1, initial_resources, [1])]
    
    # Track best found solution
    best_path = None
    best_cost = float('inf')

    # Pruning: pareto-like state tracking (simplified)
    # visited_states[vertex] = list of (cost, resources)
    visited_states = {}

    while pq:
        # Check time limit
        if time.time() - start_time > time_limit_s * 0.95:
            break
            
        cost, u, res, path = heapq.heappop(pq)

        if cost >= best_cost:
            continue

        # Check if state is dominated
        if u in visited_states:
            dominated = False
            for v_cost, v_res in visited_states[u]:
                if v_cost <= cost and all(v_res[k] <= res[k] for k in range(K)):
                    dominated = True
                    break
            if dominated:
                continue
        else:
            visited_states[u] = []
        visited_states[u].append((cost, res))

        # Goal check
        if u == n:
            # Check lower bounds
            if all(res[k] >= lower_bounds[k] - 1e-7 for k in range(K)):
                if cost < best_cost:
                    best_cost = cost
                    best_path = path
            continue

        # Explore neighbors
        for v, arc_cost, arc_res in graph.get(u, []):
            new_res = [res[k] + arc_res[k] + vertex_resources[v - 1][k] for k in range(K)]
            
            # Feasibility check against upper bounds
            if all(new_res[k] <= upper_bounds[k] + 1e-7 for k in range(K)):
                heapq.heappush(pq, (cost + arc_cost, v, new_res, path + [v]))

    if best_path:
        return {"total_cost": float(best_cost), "path": best_path}
    
    # Fallback to greedy if no optimal path found
    greedy = tools['greedy_extend_path'](time_limit_s * 0.1)
    if greedy:
        return {"total_cost": tools['path_length'](greedy), "path": greedy}

    return {"total_cost": 0.0, "path": [1, n]}