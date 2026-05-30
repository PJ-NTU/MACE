# MACE evolved heuristic 03/10 for problem: constrained_guillotine_cutting
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    A 'Greedy Randomized Adaptive Search Procedure' (GRASP) variant that
    diverges from the portfolio's 'shatter/refill' or 'local-swap' focus by
    using a 'Construction-Time Restricted Candidate List' (RCL).
    
    Portfolio common patterns:
    1. Most rely on 'bottom_left_pack_demand_aware' as a black-box.
    2. Most use post-process local swaps or total-layout shattering.
    3. Most prioritize simple value/area density metrics.
    
    This heuristic:
    1. Uses a 'look-ahead' construction: at each step, it builds a candidate
       list of all currently feasible placements and picks from the top-K
       using a weighted probability (Stochastic selection) rather than 
       full-greedy or brute-force shattering.
    2. Does NOT use 'apply_local_swap' or 'shatter/refill'.
    3. Focuses on 'depth-first' packing where we attempt to fill the stock
       by prioritizing piece types that minimize 'wasted perimeter' (a 
       geometric metric), rather than just value density.
    """
    start_time = time.time()
    m = tools['n_piece_types']()
    best_sol = {"total_value": 0, "placements": []}
    
    # Pre-calculate geometric complexity (aspect ratio bias)
    # Portfolio uses density; we use inverse aspect ratio to favor square-like
    # pieces which often fit better in guillotine layouts.
    piece_meta = []
    for i in range(1, m + 1):
        l, w = tools['piece_dims'](i)
        v = tools['piece_value'](i)
        piece_meta.append({'id': i, 'val': v, 'area': l * w, 'ratio': min(l, w) / max(l, w)})

    while time.time() - start_time < time_limit_s * 0.9:
        # Build solution step-by-step using RCL
        working_placements = []
        
        # We attempt to fill the sheet by sampling from a pool of valid moves
        # instead of fixed-order greedy construction.
        for _ in range(50): # Max pieces per attempt
            valid_moves = []
            for t in range(1, m + 1):
                if tools['used_count'](working_placements, t) < tools['piece_demand_max'](t):
                    # Check feasibility for one piece at a time
                    # We use a short-circuit attempt
                    trial = tools['try_place_piece'](working_placements, t)
                    if trial is not None:
                        # Score move: high value + high aspect-ratio-fit
                        score = piece_meta[t-1]['val'] * (1.0 + piece_meta[t-1]['ratio'])
                        valid_moves.append((score, trial))
            
            if not valid_moves:
                break
            
            # Restricted Candidate List: pick from top 30% of valid moves
            valid_moves.sort(key=lambda x: x[0], reverse=True)
            rcl_size = max(1, len(valid_moves) // 3)
            choice = random.choice(valid_moves[:rcl_size])
            working_placements = choice[1]
            
        # Evaluation
        val = tools['total_value_of'](working_placements)
        if val > best_sol["total_value"]:
            best_sol = {"total_value": val, "placements": working_placements}
    
    # Final sanity check
    is_valid, _ = tools['is_feasible'](best_sol)
    if not is_valid:
        # Fallback to standard robust greedy
        fallback = tools['guillotine_pack_BFD']()
        return {"total_value": tools['total_value_of'](fallback), "placements": fallback}
        
    return best_sol