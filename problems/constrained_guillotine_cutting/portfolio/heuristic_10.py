# MACE evolved heuristic 10/10 for problem: constrained_guillotine_cutting
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Modified GRASP: Replaces the fixed-size RCL (30%) with a temperature-based 
    Boltzmann selection (softmax) to allow more exploration early on and 
    exploitation as time progresses.
    """
    start_time = time.time()
    m = tools['n_piece_types']()
    best_sol = {"total_value": 0, "placements": []}
    
    piece_meta = []
    for i in range(1, m + 1):
        l, w = tools['piece_dims'](i)
        v = tools['piece_value'](i)
        piece_meta.append({'id': i, 'val': v, 'area': l * w, 'ratio': min(l, w) / max(l, w)})

    while time.time() - start_time < time_limit_s * 0.9:
        working_placements = []
        # Temperature decays over the course of the construction attempt
        temp = 1.0 
        
        for _ in range(100): 
            valid_moves = []
            for t in range(1, m + 1):
                if tools['used_count'](working_placements, t) < tools['piece_demand_max'](t):
                    trial = tools['try_place_piece'](working_placements, t)
                    if trial is not None:
                        score = piece_meta[t-1]['val'] * (1.0 + piece_meta[t-1]['ratio'])
                        valid_moves.append((score, trial))
            
            if not valid_moves:
                break
            
            # Boltzmann selection instead of fixed RCL
            scores = [m[0] / temp for m in valid_moves]
            exp_scores = [2.71828 ** s for s in scores]
            sum_exp = sum(exp_scores)
            probs = [e / sum_exp for e in exp_scores]
            
            choice = random.choices(valid_moves, weights=probs, k=1)[0]
            working_placements = choice[1]
            temp = max(0.1, temp * 0.95)
            
        val = tools['total_value_of'](working_placements)
        if val > best_sol["total_value"]:
            best_sol = {"total_value": val, "placements": working_placements}
    
    is_valid, _ = tools['is_feasible'](best_sol)
    if not is_valid:
        fallback = tools['guillotine_pack_BFD']()
        return {"total_value": tools['total_value_of'](fallback), "placements": fallback}
        
    return best_sol