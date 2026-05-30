# MACE evolved heuristic 09/10 for problem: constrained_guillotine_cutting
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    A robust heuristic combining diversified construction with a focused
    local search. It uses a 'shatter-and-rebuild' metaheuristic combined
    with density-based prioritization to navigate the guillotine-constrained space.
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

    # Phase 1: Aggressive Initial Construction (Multi-start)
    # Different orderings to cover different parts of the search space
    strategies = [
        sorted(range(1, m + 1), key=lambda i: piece_meta[i-1]['density'], reverse=True),
        sorted(range(1, m + 1), key=lambda i: piece_meta[i-1]['area'], reverse=True),
        sorted(range(1, m + 1), key=lambda i: piece_meta[i-1]['val'], reverse=True),
        random.sample(range(1, m + 1), m)
    ]

    for order in strategies:
        if time.time() - start_time > time_limit_s * 0.2:
            break
        try:
            p = tools['bottom_left_pack_demand_aware'](piece_order=order)
            update_best(p)
        except:
            continue

    # Phase 2: Hill Climbing with "Destructive-Reconstructive" cycles
    # We maintain a current solution and iteratively try to improve it
    current_placements = list(best_sol["placements"])
    
    # Sort candidates by density for replacement logic
    candidates_by_density = sorted(piece_meta, key=lambda x: x['density'], reverse=True)
    
    while time.time() - start_time < time_limit_s * 0.95:
        # If current is empty, restart with a known good construction
        if not current_placements:
            current_placements = list(tools['guillotine_pack_BFD']())
            continue
        
        # 1. Destruction: Remove 20-50% of the pieces
        # Bias removal toward lower density pieces to create better opportunities
        num_to_remove = max(1, len(current_placements) // 3)
        for _ in range(num_to_remove):
            if not current_placements: break
            # Select by inverse density probability
            idx = random.randrange(len(current_placements))
            current_placements.pop(idx)
        
        # 2. Reconstruct: Use stochastic greedy filling
        order = sorted(candidates_by_density, key=lambda x: x['density'] * random.uniform(0.5, 1.5), reverse=True)
        for cand in order:
            if tools['used_count'](current_placements, cand['id']) < tools['piece_demand_max'](cand['id']):
                attempt = tools['try_place_piece'](current_placements, cand['id'])
                if attempt is not None:
                    current_placements = attempt
        
        # 3. Acceptance
        val = tools['total_value_of'](current_placements)
        if val > best_sol["total_value"]:
            update_best(current_placements)
        else:
            # If no improvement, revert to best if we are stuck (simple restart logic)
            if random.random() < 0.3:
                current_placements = list(best_sol["placements"])

    # Final Verification
    is_valid, _ = tools['is_feasible'](best_sol)
    if not is_valid:
        # Fallback to a guaranteed valid construction
        fallback = tools['bottom_left_pack_demand_aware']()
        return {"total_value": tools['total_value_of'](fallback), "placements": fallback}
        
    return best_sol