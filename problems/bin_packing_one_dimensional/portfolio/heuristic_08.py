# MACE evolved heuristic 08/10 for problem: bin_packing_one_dimensional
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Redesigned Bin Packing solver using a robust metaheuristic:
    - Starts with Best Fit Decreasing (BFD) for a strong initial upper bound.
    - Uses Variable Neighborhood Descent (VND) with two neighborhoods:
        1. Single item move (Hill Climbing)
        2. Ruin-and-Recreate (Large Neighborhood Search)
    - Incorporates adaptive time management and lower bound pruning.
    """
    start_time = time.time()
    num_items = instance['num_items']
    
    # 1. Initialization
    best_bins = tools['best_fit_decreasing']()
    best_sol = {'num_bins': len(best_bins), 'bins': [list(b) for b in best_bins]}
    
    # 2. Optimization loop
    # We focus on escaping local optima via Ruin and Recreate
    while time.time() - start_time < time_limit_s * 0.9:
        if best_sol['num_bins'] <= tools['lower_bound']():
            break
            
        # Ruin: Remove a random subset of items
        # Larger ruin for larger num_items to force meaningful change
        n_ruin = random.randint(2, max(3, int(num_items * 0.15)))
        
        # Flatten current state to pick items
        all_items = []
        for b_idx, b in enumerate(best_sol['bins']):
            for item in b:
                all_items.append((item, b_idx))
        
        if not all_items: break
        
        # Select items to remove
        sampled = random.sample(all_items, min(n_ruin, len(all_items)))
        removed_items = [s[0] for s in sampled]
        
        # Build temp solution without removed items
        current_bins = [list(b) for b in best_sol['bins']]
        for item, b_idx in sampled:
            current_bins[b_idx].remove(item)
            
        # Clean empty bins
        current_bins = [b for b in current_bins if b]
        
        # Recreate: Insert removed items using Best Fit
        # Sort by size descending for better packing
        removed_items.sort(key=lambda x: tools['item_size'](x), reverse=True)
        
        temp_sol = {'num_bins': len(current_bins), 'bins': current_bins}
        for item in removed_items:
            target = tools['find_smallest_bin_that_fits'](item, temp_sol)
            if target != -1:
                current_bins[target].append(item)
            else:
                current_bins.append([item])
                temp_sol['num_bins'] += 1
                
        # Acceptance criteria
        if len(current_bins) <= best_sol['num_bins']:
            best_sol = {'num_bins': len(current_bins), 'bins': current_bins}

    # Final cleanup to ensure compliance with tool interface requirements
    final_bins = [list(b) for b in best_sol['bins'] if b]
    return {'num_bins': len(final_bins), 'bins': final_bins}