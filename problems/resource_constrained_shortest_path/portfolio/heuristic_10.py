# MACE evolved heuristic 10/10 for problem: resource_constrained_shortest_path
import time
import heapq
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    A dispatcher-style RCSP solver.
    
    Hypothesis:
    - Label-setting (Parent A) is optimal for small-to-medium graphs and 
      highly constrained instances where pruning is effective.
    - Beam search (Parent B) is superior for larger or looser graphs where 
      the state space explodes for exact methods, necessitating stochastic 
      exploration.
    
    Dispatch Criterion:
    Instances with high node count (n > 500) or high average out-degree 
    are treated as 'large', triggering the stochastic beam search to avoid
    memory overflow and time complexity issues. Smaller, more constrained
    graphs use the exact label-setting approach.
    """
    start_time = time.time()
    n = instance['n']
    graph = instance['graph']
    
    # Calculate density/size features
    avg_out_degree = sum(len(neighbors) for neighbors in graph.values()) / n
    is_large = n > 400 or avg_out_degree > 10.0
    
    if not is_large:
        # Use Label-Setting (A-style) for exact/near-exact pruning
        K = instance['K']
        lower_bounds = instance['lower_bounds']
        upper_bounds = instance['upper_bounds']
        vertex_resources = instance['vertex_resources']

        pq = [(0.0, 1, [1], list(vertex_resources[0]))]
        best_path = None
        min_cost = float('inf')
        visited = {}

        while pq and (time.time() - start_time < time_limit_s * 0.7):
            cost, u, path, res = heapq.heappop(pq)
            if cost >= min_cost: continue
            if u == n:
                if all(res[k] >= lower_bounds[k] - 1e-9 for k in range(K)):
                    min_cost = cost
                    best_path = path
                continue
            
            if u not in visited: visited[u] = []
            is_dominated = False
            for v_cost, v_res in visited[u]:
                if v_cost <= cost and all(v_res[k] <= res[k] + 1e-9 for k in range(K)):
                    is_dominated = True; break
            if is_dominated: continue
            visited[u].append((cost, res))
            
            for v, arc_cost, arc_res in graph.get(u, []):
                new_res = [res[k] + arc_res[k] + vertex_resources[v - 1][k] for k in range(K)]
                if all(new_res[k] <= upper_bounds[k] + 1e-9 for k in range(K)):
                    heapq.heappush(pq, (cost + arc_cost, v, path + [v], new_res))
    else:
        # Use Beam Search (B-style) for larger state spaces
        best_path = None
        min_cost = float('inf')
        beam_width = 20
        
        while time.time() - start_time < time_limit_s * 0.7:
            beam = [(0.0, 1, list(instance['vertex_resources'][0]), [1])]
            while beam:
                new_beam = []
                candidates = []
                for cost, u, res, path in beam:
                    if u == n:
                        if all(res[k] >= instance['lower_bounds'][k] - 1e-7 for k in range(instance['K'])):
                            if cost < min_cost:
                                min_cost = cost
                                best_path = path
                        continue
                    for v, arc_cost, arc_res in graph.get(u, []):
                        new_res = [res[k] + arc_res[k] + instance['vertex_resources'][v - 1][k] for k in range(instance['K'])]
                        if all(new_res[k] <= instance['upper_bounds'][k] + 1e-7 for k in range(instance['K'])):
                            candidates.append((cost + arc_cost + random.uniform(0, 0.05), v, new_res, path + [v]))
                if not candidates: break
                candidates.sort(key=lambda x: x[0])
                beam = candidates[:beam_width]

    if best_path:
        return {"total_cost": tools['path_length'](best_path), "path": best_path}
    
    # Final safety fallback
    path = tools.get('greedy_extend_path', lambda t: None)(time_limit_s * 0.2)
    if path:
        return {"total_cost": tools['path_length'](path), "path": path}
        
    return {"total_cost": 0.0, "path": [1, n]}