# MACE evolved heuristic 05/10 for problem: bin_packing_one_dimensional
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Improved heuristic using a temperature-based Simulated Annealing acceptance
    criterion to better navigate the local search space compared to the 
    parent's simple random-walk acceptance.
    """
    start_time = time.time()
    
    # 1. Initialization: Use the stronger BFD heuristic
    best_bins = tools['best_fit_decreasing']()
    best_solution = {'num_bins': len(best_bins), 'bins': best_bins}
    
    # Helper to clean and validate
    def finalize(bins):
        clean = [b for b in bins if len(b) > 0]
        return {'num_bins': len(clean), 'bins': clean}

    # 2. Iterated Local Search with Simulated Annealing
    lower_bound = tools['lower_bound']()
    temp = 1.0
    cooling_rate = 0.9995
    
    while time.time() - start_time < time_limit_s * 0.9:
        if best_solution['num_bins'] <= lower_bound:
            break
            
        # Perturbation: Ruin
        current_bins = [list(b) for b in best_solution['bins'] if len(b) > 0]
        num_items = instance['num_items']
        num_to_remove = random.randint(max(1, num_items // 20), max(2, num_items // 4))
        
        removed_items = []
        for _ in range(num_to_remove):
            b_idx = random.randrange(len(current_bins))
            if current_bins[b_idx]:
                removed_items.append(current_bins[b_idx].pop(random.randrange(len(current_bins[b_idx]))))
        
        # Recreate: Best Fit insertion
        removed_items.sort(key=lambda x: tools['item_size'](x), reverse=True)
        for item in removed_items:
            best_bin_idx = tools['find_smallest_bin_that_fits'](item, {'num_bins': len(current_bins), 'bins': current_bins})
            if best_bin_idx != -1:
                current_bins[best_bin_idx].append(item)
            else:
                current_bins.append([item])
        
        candidate_sol = finalize(current_bins)
        
        # Acceptance: Simulated Annealing criterion instead of simple random walk
        delta = candidate_sol['num_bins'] - best_solution['num_bins']
        if delta < 0 or (delta == 0 and random.random() < temp):
            best_solution = candidate_sol
        
        temp *= cooling_rate
            
    return best_solution