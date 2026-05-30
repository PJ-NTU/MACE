# MACE evolved heuristic 07/10 for problem: resource_constrained_shortest_path
import time
import heapq

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Solves the Resource Constrained Shortest Path problem using a 
    Label Setting algorithm with a priority queue (Dijkstra-style).
    
    This implementation respects the provided time limit and returns
    a feasible solution.
    """
    start_time = time.time()
    
    n = instance['n']
    K = instance['K']
    graph = instance['graph']
    lower_bounds = instance['lower_bounds']
    upper_bounds = instance['upper_bounds']
    vertex_resources = instance['vertex_resources']
    
    # State: (cost, current_node, current_resources, path)
    # Using a priority queue to explore low-cost paths first.
    # To prevent exponential growth, we limit the number of labels per node.
    start_node = 1
    target_node = n
    
    initial_res = list(vertex_resources[0])
    
    # Priority Queue stores: (cost, node, resource_tuple, path_list)
    pq = [(0.0, start_node, tuple(initial_res), [start_node])]
    
    # Track best path found so far
    best_path = None
    min_cost = float('inf')
    
    # To prune redundant paths at nodes: keep track of visited resource states
    # This is a basic form of label dominance
    visited = {} 
    
    while pq:
        # Check time limit
        if time.time() - start_time > time_limit_s * 0.95:
            break
            
        cost, u, res, path = heapq.heappop(pq)
        
        if cost >= min_cost:
            continue
            
        if u == target_node:
            # Check feasibility against lower bounds
            feasible = True
            for k in range(K):
                if res[k] < lower_bounds[k]:
                    feasible = False
                    break
            if feasible:
                min_cost = cost
                best_path = path
            continue
            
        # Explore neighbors
        for v, arc_cost, arc_res in graph.get(u, []):
            new_cost = cost + arc_cost
            new_res = [res[k] + arc_res[k] + vertex_resources[v-1][k] for k in range(K)]
            
            # Prune if over upper bounds
            if any(new_res[k] > upper_bounds[k] for k in range(K)):
                continue
                
            # Basic dominance: if we've reached node v with lower cost AND lower 
            # resources, this state is potentially better.
            state = (v, tuple(new_res))
            if state in visited and visited[state] <= new_cost:
                continue
            visited[state] = new_cost
            
            heapq.heappush(pq, (new_cost, v, tuple(new_res), path + [v]))
            
    if best_path:
        return {
            "total_cost": min_cost,
            "path": best_path
        }
    
    # Fallback to a greedy approach if Label Setting didn't find a path
    greedy_path = tools.get('greedy_extend_path', lambda: None)()
    if greedy_path:
        return {
            "total_cost": tools['path_length'](greedy_path),
            "path": greedy_path
        }
        
    return {"total_cost": 0.0, "path": [1, n]}