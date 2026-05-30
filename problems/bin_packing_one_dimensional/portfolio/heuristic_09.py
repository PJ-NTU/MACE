# MACE evolved heuristic 09/10 for problem: bin_packing_one_dimensional
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Improved solver using a hybrid approach:
    1. Start with Best Fit Decreasing (BFD) to get a strong initial baseline.
    2. Use an Iterated Local Search (ILS) with a Simulated Annealing-inspired 
       acceptance criterion to escape local optima.
    3. Perform 'Ruin-and-Recreate' by removing a subset of items and re-inserting 
       them using a Best Fit strategy to optimize bin packing.
    """
    start_time = time.time()
    
    # 1. Initialization
    best_bins = tools['best_fit_decreasing']()
    best_sol = {'num_bins': len(best_bins), 'bins': [list(b) for b in best_bins]}
    
    # Pre-calculate item sizes for performance
    num_items = instance['num_items']
    items_sizes = [tools['item_size'](i) for i in range(1, num_items + 1)]
    lb = tools['lower_bound']()
    
    current_sol = dict(best_sol)
    
    # 2. Iterative optimization
    # Time buffer for safety
    while time.time() - start_time < time_limit_s * 0.9:
        if best_sol['num_bins'] <= lb:
            break
            
        # Ruin: Remove a random percentage of items
        # Larger ruin for larger instances to escape deeper local optima
        ruin_size = random.randint(2, max(3, num_items // 8))
        
        # Flatten and pick items to remove
        flat_items = []
        for b_idx, b in enumerate(current_sol['bins']):
            for item in b:
                flat_items.append((item, b_idx))
        
        random.shuffle(flat_items)
        to_remove = flat_items[:ruin_size]
        
        # Create a new candidate state
        new_bins = [list(b) for b in current_sol['bins']]
        removed_ids = []
        for item_idx, b_idx in to_remove:
            new_bins[b_idx].remove(item_idx)
            removed_ids.append(item_idx)
            
        # Clean up empty bins
        new_bins = [b for b in new_bins if len(b) > 0]
        
        # Recreate: Best Fit insertion
        removed_ids.sort(key=lambda i: items_sizes[i-1], reverse=True)
        
        temp_sol = {'num_bins': len(new_bins), 'bins': new_bins}
        for item_idx in removed_ids:
            target = tools['find_smallest_bin_that_fits'](item_idx, temp_sol)
            if target != -1:
                temp_sol['bins'][target].append(item_idx)
            else:
                temp_sol['bins'].append([item_idx])
                temp_sol['num_bins'] += 1
                
        # Acceptance: Hill climbing with simple acceptance
        if temp_sol['num_bins'] <= current_sol['num_bins']:
            current_sol = {'num_bins': temp_sol['num_bins'], 'bins': [list(b) for b in temp_sol['bins']]}
            if current_sol['num_bins'] < best_sol['num_bins']:
                best_sol = {'num_bins': current_sol['num_bins'], 'bins': [list(b) for b in current_sol['bins']]}
        else:
            # Occasional diversification (simulated annealing-like)
            if random.random() < 0.05:
                current_sol = {'num_bins': temp_sol['num_bins'], 'bins': [list(b) for b in temp_sol['bins']]}
                
    return best_sol