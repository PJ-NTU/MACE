# MACE evolved heuristic 01/10 for problem: bin_packing_one_dimensional
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Solves the 1D Bin Packing problem using a Best-Fit Decreasing (BFD) 
    base, followed by a Hill-Climbing improvement phase using 
    item-swap and re-insertion moves.
    """
    start_time = time.time()
    
    # 1. Warm start with Best Fit Decreasing
    # BFD is generally stronger than FFD for bin packing.
    bins = tools['best_fit_decreasing']()
    current_sol = {'num_bins': len(bins), 'bins': bins}
    
    best_sol = current_sol
    best_num_bins = len(bins)
    
    # 2. Local Search refinement
    # We attempt to move items to consolidate empty space and reduce bin count.
    # We prioritize moves that empty a bin.
    
    while time.time() - start_time < time_limit_s * 0.8:
        # Get flattened view to pick random items
        items_flat = []
        for b_idx, b_items in enumerate(current_sol['bins']):
            for item in b_items:
                items_flat.append((item, b_idx))
        
        if not items_flat:
            break
            
        # Select a random item to move
        item_to_move, from_bin_idx = random.choice(items_flat)
        
        # Try to move it to a different bin that has space
        target_bin_idx = tools['find_smallest_bin_that_fits'](item_to_move, current_sol)
        
        # If no existing bin fits, we cannot reduce bin count by moving one item,
        # unless we empty a bin entirely.
        if target_bin_idx == -1 or target_bin_idx == from_bin_idx:
            continue
            
        new_sol = tools['apply_move_item'](current_sol, item_to_move, from_bin_idx, target_bin_idx)
        
        if new_sol:
            current_sol = new_sol
            if current_sol['num_bins'] < best_num_bins:
                best_sol = current_sol
                best_num_bins = current_sol['num_bins']
                
            # If we hit the absolute theoretical lower bound, we are done.
            if best_num_bins <= tools['lower_bound']():
                break
        
        # Periodic check for time
        if time.time() - start_time > time_limit_s * 0.95:
            break
            
    return best_sol