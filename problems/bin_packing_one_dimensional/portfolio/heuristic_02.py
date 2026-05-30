# MACE evolved heuristic 02/10 for problem: bin_packing_one_dimensional
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Implements a Metaheuristic approach:
    1. Start with Best Fit Decreasing (BFD) as a strong baseline.
    2. Perform Iterated Local Search (ILS) with a Random-Restart/Perturbation strategy
       to escape local optima within the time limit.
    """
    start_time = time.time()
    num_items = instance['num_items']
    
    # Generate initial solution using BFD
    best_bins = tools['best_fit_decreasing']()
    best_solution = {'num_bins': len(best_bins), 'bins': best_bins}
    
    def get_num_bins(bins):
        return len([b for b in bins if len(b) > 0])

    # ILS Loop
    while time.time() - start_time < time_limit_s * 0.9:
        # Perturbation: Remove random items and re-insert
        current_bins = [list(b) for b in best_solution['bins'] if len(b) > 0]
        
        # Ruin: Remove ~10-20% of items
        num_to_remove = max(1, num_items // 10)
        removed_items = []
        for _ in range(num_to_remove):
            b_idx = random.randrange(len(current_bins))
            if current_bins[b_idx]:
                item_idx = current_bins[b_idx].pop(random.randrange(len(current_bins[b_idx])))
                removed_items.append(item_idx)
        
        # Recreate: Simple greedy insertion for removed items
        for item in removed_items:
            placed = False
            # Try to find existing bin
            best_bin_idx = tools['find_smallest_bin_that_fits'](item, {'num_bins': len(current_bins), 'bins': current_bins})
            if best_bin_idx != -1:
                current_bins[best_bin_idx].append(item)
                placed = True
            else:
                current_bins.append([item])
        
        # Clean empty bins and check
        candidate_bins = [b for b in current_bins if len(b) > 0]
        candidate_sol = {'num_bins': len(candidate_bins), 'bins': candidate_bins}
        
        # Update best
        if len(candidate_bins) < len(best_solution['bins']):
            best_solution = candidate_sol
            
        # Early exit if we hit the lower bound
        if len(best_solution['bins']) <= tools['lower_bound']():
            break
            
    return best_solution