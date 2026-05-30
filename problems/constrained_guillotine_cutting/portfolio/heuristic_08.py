# MACE evolved heuristic 08/10 for problem: constrained_guillotine_cutting
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    A robust hybrid heuristic for the Constrained Guillotine Cutting problem.
    Combines diverse constructive baselines with a focused local search strategy
    that balances exploration (random perturbations) and exploitation (value-density
    greedy replacement).
    """
    start_time = time.time()
    m = tools['n_piece_types']()
    
    # Precompute metadata for informed greedy choices
    piece_meta = []
    for i in range(1, m + 1):
        l, w = tools['piece_dims'](i)
        val = tools['piece_value'](i)
        piece_meta.append({
            'id': i, 
            'val': val, 
            'area': l * w, 
            'density': val / (l * w + 1e-9)
        })

    best_sol = {"total_value": 0, "placements": []}

    def update_best(placements):
        nonlocal best_sol
        val = tools['total_value_of'](placements)
        if val > best_sol["total_value"]:
            best_sol = {"total_value": val, "placements": list(placements)}

    # 1. Construction Phase: Generate diverse starting points
    # Use different sorting strategies to explore the solution space
    strategies = [
        sorted(range(1, m + 1), key=lambda i: piece_meta[i-1]['density'], reverse=True),
        sorted(range(1, m + 1), key=lambda i: piece_meta[i-1]['area'], reverse=True),
        sorted(range(1, m + 1), key=lambda i: piece_meta[i-1]['val'], reverse=True),
    ]

    for order in strategies:
        if time.time() - start_time > time_limit_s * 0.2:
            break
        try:
            p = tools['bottom_left_pack_demand_aware'](piece_order=order)
            update_best(p)
        except:
            continue

    try:
        bfd = tools['guillotine_pack_BFD']()
        update_best(bfd)
    except:
        pass

    # 2. Refinement Phase: Local Search
    # Use apply_local_swap as a powerful engine for local improvement
    time_remaining = time_limit_s - (time.time() - start_time)
    if time_remaining > 0.1:
        try:
            improved = tools['apply_local_swap'](
                best_sol["placements"], 
                t_limit=time_remaining * 0.5
            )
            update_best(improved)
        except:
            pass

    # 3. Iterative Perturbation Phase
    # If time remains, perform a "shake" and re-fill to escape local optima
    current_placements = list(best_sol["placements"])
    candidates = sorted(piece_meta, key=lambda x: x['density'], reverse=True)
    
    while time.time() - start_time < time_limit_s * 0.95:
        if not current_placements:
            break
            
        # Remove a random subset to create space
        num_to_remove = max(1, len(current_placements) // 4)
        for _ in range(num_to_remove):
            if current_placements:
                current_placements.pop(random.randrange(len(current_placements)))
        
        # Try to fill with high-density pieces
        for cand in candidates:
            if tools['used_count'](current_placements, cand['id']) < tools['piece_demand_max'](cand['id']):
                attempt = tools['try_place_piece'](current_placements, cand['id'])
                if attempt:
                    current_placements = attempt
                    update_best(current_placements)
        
        # If we failed to improve, reset to best known
        if tools['total_value_of'](current_placements) < best_sol["total_value"]:
            current_placements = list(best_sol["placements"])

    # Final sanity check
    is_valid, _ = tools['is_feasible'](best_sol)
    if not is_valid:
        # Fallback to the most robust construction
        fallback = tools['guillotine_pack_BFD']()
        return {"total_value": tools['total_value_of'](fallback), "placements": fallback}
        
    return best_sol