# MACE evolved heuristic 10/10 for problem: travelling_salesman_problem
import time
import random
import math

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Dispatcher-style solver for TSP.
    
    Hypothesis: 
    - Smaller, sparser, or more clustered instances benefit from Parent B's 
      multi-start NN initialization to locate the global basin early.
    - Larger or highly uniform instances benefit from Parent A's focus on 
      long-duration refinement of a single high-quality seed.
    
    Metric: The variance of pairwise distances relative to the average 
    distance can indicate structure; we use instance size and a small 
    sample of distances to dispatch.
    """
    start_time = time.time()
    nodes = instance['nodes']
    num_nodes = len(nodes)
    
    if num_nodes <= 1:
        return {'tour': list(range(num_nodes))}

    # Feature extraction: Sample distance variance to guess instance structure
    sample_size = min(num_nodes, 50)
    indices = random.sample(range(num_nodes), sample_size)
    distances = []
    for i in range(sample_size):
        for j in range(i + 1, sample_size):
            p1, p2 = nodes[indices[i]], nodes[indices[j]]
            dist = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            distances.append(dist)
    
    if distances:
        avg_dist = sum(distances) / len(distances)
        var_dist = sum((d - avg_dist)**2 for d in distances) / len(distances)
        cv = math.sqrt(var_dist) / (avg_dist + 1e-9)
    else:
        cv = 0

    # Dispatch Logic:
    # If instance is small (< 100 nodes) or has high distance variance (structured/clustered),
    # use multi-start NN initialization (Parent B).
    # Otherwise, use single-seed refinement (Parent A).
    is_b_regime = (num_nodes < 150) or (cv > 0.5)

    if is_b_regime:
        # Parent B strategy
        best_tour = None
        best_cost = float('inf')
        init_time_limit = time_limit_s * 0.15
        while time.time() - start_time < init_time_limit:
            start_node = random.randint(0, num_nodes - 1)
            candidate_tour = tools['nn_construct'](start_node=start_node)
            candidate_cost = tools['tour_length'](candidate_tour)
            if candidate_cost < best_cost:
                best_cost = candidate_cost
                best_tour = list(candidate_tour)
    else:
        # Parent A strategy
        best_tour = tools['nn_construct'](start_node=0)
        best_tour = tools['apply_2opt'](best_tour, time_limit_s=time_limit_s * 0.2, first_improvement=True)
        best_cost = tools['tour_length'](best_tour)

    # Common Refinement Loop
    while time.time() - start_time < time_limit_s * 0.95:
        # Perturbation
        perturbed_tour = list(best_tour)
        if num_nodes > 4:
            idx1, idx2 = sorted(random.sample(range(num_nodes), 2))
            perturbed_tour[idx1:idx2] = reversed(perturbed_tour[idx1:idx2])

        remaining_time = time_limit_s - (time.time() - start_time)
        if remaining_time <= 0.05:
            break
            
        current_tour = tools['apply_2opt'](
            perturbed_tour, 
            time_limit_s=max(0.05, remaining_time * 0.5), 
            first_improvement=True
        )
        
        current_tour = tools['apply_or_opt_single'](
            current_tour, 
            time_limit_s=max(0.05, remaining_time * 0.5)
        )

        current_cost = tools['tour_length'](current_tour)
        if current_cost < best_cost:
            best_cost = current_cost
            best_tour = list(current_tour)

    return {'tour': best_tour}