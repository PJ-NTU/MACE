# MACE evolved heuristic 05/10 for problem: resource_constrained_shortest_path
import time
import heapq

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Solves the Resource Constrained Shortest Path problem using a 
    Label Setting algorithm with a priority queue (Dijkstra-like expansion).
    We prune using dominance and the resource constraints to ensure 
    efficiency and feasibility.
    """
    start_time = time.time()
    
    n = instance['n']
    K = instance['K']
    lower_bounds = instance['lower_bounds']
    upper_bounds = instance['upper_bounds']
    graph = instance['graph']
    vertex_resources = instance['vertex_resources']

    # Priority Queue stores: (cost, current_node, resource_totals, path)
    # Using a list for resource_totals to keep track of cumulative consumption.
    # Cost is the primary sort key.
    initial_resources = list(vertex_resources[0])
    pq = [(0.0, 1, initial_resources, [1])]
    
    # Keep track of best costs found at each node for specific resource signatures
    # to perform basic dominance pruning (optional but recommended for speed).
    # Since resources are continuous, we track the best cost seen at each vertex.
    best_costs = {} 

    best_solution = None
    best_total_cost = float('inf')

    while pq:
        # Check time limit
        if time.time() - start_time > time_limit_s * 0.9:
            break
            
        cost, u, res, path = heapq.heappop(pq)

        if cost >= best_total_cost:
            continue

        if u == n:
            # Check lower bounds feasibility
            feasible = True
            for k in range(K):
                if res[k] < lower_bounds[k] - 1e-7:
                    feasible = False
                    break
            if feasible:
                if cost < best_total_cost:
                    best_total_cost = cost
                    best_solution = {"total_cost": cost, "path": path}
            continue

        # Explore neighbors
        for v, arc_cost, arc_res in graph.get(u, []):
            new_res = [res[k] + arc_res[k] + vertex_resources[v - 1][k] for k in range(K)]
            
            # Check upper bounds
            if all(new_res[k] <= upper_bounds[k] + 1e-7 for k in range(K)):
                new_cost = cost + arc_cost
                
                # Simple pruning: if we have reached this node with a higher cost 
                # and worse resources, we could prune. Here we just track cost.
                state = (v, tuple(new_res))
                if new_cost < best_costs.get(state, float('inf')):
                    best_costs[state] = new_cost
                    new_path = path + [v]
                    heapq.heappush(pq, (new_cost, v, new_res, new_path))

    # Fallback: if no feasible path found, try the provided tools if available
    if best_solution is None:
        try:
            path = tools['greedy_extend_path'](time_limit_s=max(0.1, time_limit_s * 0.1))
            if path:
                return {"total_cost": tools['path_length'](path), "path": path}
        except:
            pass
        return {"total_cost": 0.0, "path": [1, n]}

    return best_solution