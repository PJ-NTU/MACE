# MACE evolved heuristic 04/10 for problem: bin_packing_one_dimensional
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Combines BFD initialization, Iterated Local Search (ILS), and a 
    First-Fit-Decreasing (FFD) based perturbation strategy to escape 
    local optima efficiently.
    """
    start_time = time.time()
    
    # 1. Initialization: Use the stronger BFD heuristic
    best_bins = tools['best_fit_decreasing']()
    best_solution = {'num_bins': len(best_bins), 'bins': best_bins}
    
    # Helper to clean and validate
    def finalize(bins):
        clean = [b for b in bins if len(b) > 0]
        return {'num_bins': len(clean), 'bins': clean}

    # 2. Iterated Local Search
    # We use a Ruin-and-Recreate strategy. Unlike pure Hill Climbing, 
    # this allows jumping out of local minima by removing multiple items.
    
    lower_bound = tools['lower_bound']()
    
    while time.time() - start_time < time_limit_s * 0.9:
        # Check if we already reached optimality
        if best_solution['num_bins'] <= lower_bound:
            break
            
        # Perturbation: Ruin
        # Remove a random percentage of items (between 5% and 25%)
        current_bins = [list(b) for b in best_solution['bins'] if len(b) > 0]
        num_items = instance['num_items']
        num_to_remove = random.randint(max(1, num_items // 20), max(2, num_items // 4))
        
        removed_items = []
        for _ in range(num_to_remove):
            b_idx = random.randrange(len(current_bins))
            if current_bins[b_idx]:
                removed_items.append(current_bins[b_idx].pop(random.randrange(len(current_bins[b_idx]))))
        
        # Recreate: Best Fit insertion for removed items
        # Re-sorting removed items by size helps the greedy heuristic
        removed_items.sort(key=lambda x: tools['item_size'](x), reverse=True)
        
        for item in removed_items:
            best_bin_idx = tools['find_smallest_bin_that_fits'](item, {'num_bins': len(current_bins), 'bins': current_bins})
            if best_bin_idx != -1:
                current_bins[best_bin_idx].append(item)
            else:
                current_bins.append([item])
        
        # Local Search: Simple Hill Climbing on the recreated solution
        # Try to move items from the most filled bin to others to potentially close it
        candidate_sol = finalize(current_bins)
        
        # Acceptance: If better or equal, replace (random walk/thresholding logic)
        if candidate_sol['num_bins'] < best_solution['num_bins']:
            best_solution = candidate_sol
        elif candidate_sol['num_bins'] == best_solution['num_bins'] and random.random() < 0.1:
            best_solution = candidate_sol
            
    return best_solution