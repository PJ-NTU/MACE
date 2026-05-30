# MACE evolved heuristic 04/10 for problem: packing_unequal_rectangles_and_squares_area
import time
import random
import math

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Dispatcher-style heuristic:
    - For high-density instances (small total area relative to container area), 
      prioritize the destructive-reconstructive ILS (Parent A) to fill gaps.
    - For low-density or highly-constrained instances, prioritize the 
      Simulated Annealing (Parent B) which explores local moves more flexibly.
    """
    start_time = time.time()
    n = instance['n']
    R = instance['R']
    items = instance['items']
    rotation_allowed = instance['rotation']
    
    # Calculate container capacity vs item total area
    container_area = math.pi * (R ** 2)
    total_item_area = sum(l * w for l, w in items)
    
    # Dispatch Logic:
    # If the instance is very crowded (total item area > 70% of container),
    # the greedy-reconstructive approach is better at finding tight packings.
    # Otherwise, use the SA approach for better local exploration.
    if total_item_area / container_area > 0.7:
        return _solve_ils(instance, tools, time_limit_s)
    else:
        return _solve_sa(instance, tools, time_limit_s)

def _solve_ils(instance, tools, time_limit_s):
    start_time = time.time()
    n = instance['n']
    rotation_allowed = instance['rotation']
    
    current_placements = tools['greedy_by_area_first'](time_limit_s=time_limit_s * 0.2, grid=20)
    best_placements = list(current_placements)
    best_area = tools['total_area'](best_placements)
    
    items_with_area = sorted([(i, tools['item_area'](i)) for i in range(n)], key=lambda x: x[1], reverse=True)
    sorted_indices = [x[0] for x in items_with_area]
    rng = random.Random(42)
    
    while time.time() - start_time < time_limit_s * 0.95:
        work_placements = list(best_placements)
        num_remove = rng.randint(1, max(1, n // 3))
        for _ in range(num_remove):
            idx = rng.choice(sorted_indices)
            work_placements[idx] = (-1.0, -1.0, 0.0)
            
        for idx in sorted_indices:
            if work_placements[idx][0] == -1:
                possible_thetas = [0, 90] if rotation_allowed else [0]
                for _ in range(10):
                    theta = rng.choice(possible_thetas)
                    pos = tools['random_feasible_position'](idx, theta=theta, placements=work_placements, max_tries=20, rng=rng)
                    if pos:
                        new_state = tools['try_place'](work_placements, idx, pos[0], pos[1], theta)
                        if new_state:
                            work_placements = new_state
                            break
                            
        current_area = tools['total_area'](work_placements)
        if current_area > best_area:
            is_valid, _ = tools['is_feasible']({'placements': work_placements})
            if is_valid:
                best_placements = work_placements
                best_area = current_area
    return {'placements': best_placements}

def _solve_sa(instance, tools, time_limit_s):
    start_time = time.time()
    n = instance['n']
    rotation = instance['rotation']
    best_placements = tools['greedy_by_area_first'](time_limit_s=time_limit_s * 0.2, grid=20)
    best_area = tools['total_area'](best_placements)
    current_placements = list(best_placements)
    current_area = best_area
    temp = 1.0
    cooling_rate = 0.9999
    rng = random.Random(42)
    
    while time.time() - start_time < time_limit_s * 0.95:
        move_type = rng.random()
        new_placements = list(current_placements)
        if move_type < 0.6:
            idx = rng.randint(0, n - 1)
            theta = rng.choice([0, 90]) if rotation else 0
            pos = tools['random_feasible_position'](idx, theta, current_placements, max_tries=20, rng=rng)
            if pos:
                new_placements[idx] = (pos[0], pos[1], theta)
        else:
            idx = rng.randint(0, n - 1)
            new_placements[idx] = (-1.0, -1.0, 0.0)
            
        is_valid, _ = tools['is_feasible']({'placements': new_placements})
        if is_valid:
            new_area = tools['total_area'](new_placements)
            delta = new_area - current_area
            if delta > 0 or (temp > 1e-6 and math.exp(delta / (temp + 1e-8)) > rng.random()):
                current_placements = new_placements
                current_area = new_area
                if current_area > best_area:
                    best_area = current_area
                    best_placements = list(current_placements)
            temp *= cooling_rate
    return {'placements': best_placements}