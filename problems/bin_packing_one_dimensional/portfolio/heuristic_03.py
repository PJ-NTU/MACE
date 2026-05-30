# MACE evolved heuristic 03/10 for problem: bin_packing_one_dimensional
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    A robust solver for the one-dimensional bin packing problem using a 
    Best-Fit Decreasing (BFD) initialization followed by a hill-climbing 
    local search phase to refine the bin count.
    """
    start_time = time.time()
    
    # 1. Start with a strong greedy heuristic
    # BFD is generally superior to FFD for bin packing.
    bins = tools['best_fit_decreasing']()
    best_solution = {
        'num_bins': len(bins),
        'bins': bins
    }
    
    # Check if we hit the theoretical lower bound immediately
    if best_solution['num_bins'] <= tools['lower_bound']():
        return best_solution

    # 2. Local Search: Attempt to move items from the most sparsely filled bin
    # to other bins to potentially eliminate the last bin.
    # We continue until we run out of time.
    while time.time() - start_time < time_limit_s * 0.8:
        current_bins = best_solution['bins']
        if len(current_bins) <= 1:
            break
            
        # Identify the bin with the smallest load (the best candidate to empty)
        # We use simple index-based logic to avoid overhead
        bin_loads = [(tools['bin_load'](b), i) for i, b in enumerate(current_bins)]
        bin_loads.sort()
        
        # Try to move items from the emptiest bin to others
        target_bin_idx = bin_loads[0][1]
        items_to_move = list(current_bins[target_bin_idx])
        
        # Shuffle to try different move orders
        random.shuffle(items_to_move)
        
        temp_solution = {
            'num_bins': len(current_bins),
            'bins': [list(b) for i, b in enumerate(current_bins) if i != target_bin_idx]
        }
        
        possible = True
        for item in items_to_move:
            dest = tools['find_smallest_bin_that_fits'](item, temp_solution)
            if dest != -1:
                temp_solution['bins'][dest].append(item)
            else:
                possible = False
                break
        
        if possible:
            # Successfully emptied a bin
            temp_solution['num_bins'] = len(temp_solution['bins'])
            best_solution = temp_solution
            # If we reached the theoretical lower bound, exit early
            if best_solution['num_bins'] <= tools['lower_bound']():
                break
        else:
            # If move failed, try a random perturbation (swap)
            # Pick two random bins and swap one item if possible
            if len(current_bins) > 1:
                b1, b2 = random.sample(range(len(current_bins)), 2)
                if current_bins[b1] and current_bins[b2]:
                    i1 = random.choice(current_bins[b1])
                    i2 = random.choice(current_bins[b2])
                    
                    # Try swapping i1 and i2
                    new_b1 = [x for x in current_bins[b1] if x != i1] + [i2]
                    new_b2 = [x for x in current_bins[b2] if x != i2] + [i1]
                    
                    if tools['is_bin_feasible'](new_b1) and tools['is_bin_feasible'](new_b2):
                        new_bins = [list(b) for b in current_bins]
                        new_bins[b1] = new_b1
                        new_bins[b2] = new_b2
                        best_solution = {'num_bins': len(new_bins), 'bins': new_bins}
        
    return best_solution