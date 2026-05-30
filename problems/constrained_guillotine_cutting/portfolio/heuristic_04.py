# MACE evolved heuristic 04/10 for problem: constrained_guillotine_cutting
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    A dispatch-style heuristic for Constrained Guillotine Cutting.
    
    Dispatch Hypothesis:
    - Instances with a high 'diversity-to-stock' ratio (many small pieces relative 
      to stock area) benefit from the destructive-reconstructive approach (A), 
      which helps escape local optima in highly constrained packing.
    - Instances with a low 'diversity-to-stock' ratio (fewer, larger pieces) 
      benefit from the simulated annealing-like cooling schedule (B), which 
      systematically explores packing orders for larger, more impactful pieces.
    """
    start_time = time.time()
    stock_l, stock_w = tools['stock_dims']()
    stock_area = stock_l * stock_w
    n_types = tools['n_piece_types']()
    
    total_potential_area = 0
    for i in range(1, n_types + 1):
        l, w = tools['piece_dims'](i)
        total_potential_area += (l * w) * tools['piece_demand_max'](i)
    
    # Feature: Area Density Ratio
    # If total potential area is much larger than stock, the problem is 'tightly packed'.
    # If smaller or comparable, it's 'sparse'.
    density_ratio = total_potential_area / stock_area
    
    # Dispatch: 
    # High density_ratio -> A (Destructive-Reconstructive)
    # Low density_ratio -> B (Annealing-based construction)
    if density_ratio > 1.5:
        return _run_heuristic_a(tools, time_limit_s, start_time)
    else:
        return _run_heuristic_b(tools, time_limit_s, start_time)

def _run_heuristic_a(tools, time_limit_s, start_time):
    m = tools['n_piece_types']()
    indices = list(range(1, m + 1))
    
    def get_stochastic_order():
        densities = []
        for i in indices:
            l, w = tools['piece_dims'](i)
            v = tools['piece_value'](i)
            densities.append((v / (l * w + 1e-9), i))
        densities.sort(key=lambda x: x[0], reverse=True)
        order = [d[1] for d in densities]
        for i in range(len(order)):
            if random.random() < 0.2:
                j = random.randint(0, len(order) - 1)
                order[i], order[j] = order[j], order[i]
        return order

    best_sol = {"total_value": 0, "placements": []}
    while time.time() - start_time < time_limit_s * 0.9:
        if best_sol["placements"]:
            keep_count = int(len(best_sol["placements"]) * 0.5)
            work_placements = random.sample(best_sol["placements"], keep_count)
        else:
            work_placements = []
        
        for p_idx in get_stochastic_order():
            while tools['used_count'](work_placements, p_idx) < tools['piece_demand_max'](p_idx):
                attempt = tools['try_place_piece'](work_placements, p_idx)
                if attempt is not None:
                    work_placements = attempt
                else:
                    break
        val = tools['total_value_of'](work_placements)
        if val > best_sol["total_value"]:
            best_sol = {"total_value": val, "placements": work_placements}
    
    return best_sol

def _run_heuristic_b(tools, time_limit_s, start_time):
    m = tools['n_piece_types']()
    indices = list(range(1, m + 1))
    
    def get_stochastic_order(mode, temp):
        items = []
        for i in indices:
            l, w = tools['piece_dims'](i)
            v = tools['piece_value'](i)
            score = (v / (l * w + 1e-9)) if mode == 'density' else (l * w)
            items.append((score, i))
        items.sort(key=lambda x: x[0], reverse=True)
        order = [x[1] for x in items]
        jitter_prob = 0.4 * temp
        for i in range(len(order)):
            if random.random() < jitter_prob:
                j = random.randint(0, len(order) - 1)
                order[i], order[j] = order[j], order[i]
        return order

    best_sol = {"total_value": 0, "placements": []}
    iteration = 0
    while time.time() - start_time < time_limit_s * 0.85:
        temp = max(0.1, 1.0 - ((time.time() - start_time) / (time_limit_s * 0.85)))
        mode = 'density' if iteration % 2 == 0 else 'area'
        work_placements = []
        for p_idx in get_stochastic_order(mode, temp):
            while tools['used_count'](work_placements, p_idx) < tools['piece_demand_max'](p_idx):
                attempt = tools['try_place_piece'](work_placements, p_idx)
                if attempt is not None:
                    work_placements = attempt
                else:
                    break
        val = tools['total_value_of'](work_placements)
        if val > best_sol["total_value"]:
            best_sol = {"total_value": val, "placements": work_placements}
        iteration += 1
            
    if best_sol["placements"]:
        refined = tools['apply_local_swap'](best_sol["placements"], t_limit=max(0.1, time_limit_s * 0.1))
        best_sol = {"total_value": tools['total_value_of'](refined), "placements": refined}
    return best_sol