# MACE evolved heuristic 06/10 for problem: bin_packing_one_dimensional
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    An optimized metaheuristic for the 1D bin packing problem.
    Uses a combination of Best Fit Decreasing (BFD) initialization,
    and Variable Neighborhood Search (VNS) with adaptive ruin-and-recreate.
    """
    start_time = time.time()
    num_items = instance['num_items']
    
    # 1. Initialization
    # BFD typically provides a very strong starting point (at most 11/9*OPT + 6/9).
    best_bins = tools['best_fit_decreasing']()
    best_solution = {'num_bins': len(best_bins), 'bins': best_bins}
    best_score = len(best_bins)
    
    lower_bound = tools['lower_bound']()
    
    # Helper to maintain valid structure
    def get_clean_bins(bins):
        return [list(b) for b in bins if len(b) > 0]

    # 2. Variable Neighborhood Search Loop
    # We vary the 'ruin' intensity over time: start with small perturbations
    # to refine, then increase intensity to escape local optima.
    iteration = 0
    while time.time() - start_time < time_limit_s * 0.95:
        if best_score <= lower_bound:
            break
            
        iteration += 1
        # Adaptive intensity: larger ruin for longer durations
        intensity = 0.05 + 0.25 * (iteration % 10) / 10.0
        num_to_remove = max(1, int(num_items * intensity))
        
        # Ruin: Remove random items
        current_bins = get_clean_bins(best_solution['bins'])
        removed_items = []
        for _ in range(num_to_remove):
            if not current_bins: break
            b_idx = random.randrange(len(current_bins))
            if current_bins[b_idx]:
                removed_items.append(current_bins[b_idx].pop(random.randrange(len(current_bins[b_idx]))))
        
        # Recreate: Best Fit insertion strategy
        # Sorting removed items by size descending is often superior for bin packing
        removed_items.sort(key=lambda x: tools['item_size'](x), reverse=True)
        
        for item in removed_items:
            # Attempt to find best existing bin
            best_bin_idx = tools['find_smallest_bin_that_fits'](item, {'num_bins': len(current_bins), 'bins': current_bins})
            if best_bin_idx != -1:
                current_bins[best_bin_idx].append(item)
            else:
                current_bins.append([item])
        
        # Local Improvement:
        # Sort bins by load descending to see if we can consolidate smaller bins
        current_bins.sort(key=lambda b: sum(tools['item_size'](i) for i in b), reverse=True)
        
        candidate_bins = get_clean_bins(current_bins)
        candidate_num = len(candidate_bins)
        
        # Acceptance: Hill climbing with a slight bias towards smaller bins
        if candidate_num < best_score:
            best_score = candidate_num
            best_solution = {'num_bins': candidate_num, 'bins': candidate_bins}
        elif candidate_num == best_score:
            # Occasional swap to maintain diversity
            if random.random() < 0.05:
                best_solution = {'num_bins': candidate_num, 'bins': candidate_bins}

    return best_solution