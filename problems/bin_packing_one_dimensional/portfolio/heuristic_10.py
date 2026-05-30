# MACE evolved heuristic 10/10 for problem: bin_packing_one_dimensional
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Dispatcher-style heuristic for 1D Bin Packing.
    
    Strategy:
    - If the lower bound is very close to the BFD result, the problem is likely 
      'tight' or 'easy'. We use a refined Hill Climbing approach to polish it.
    - If there is a significant gap between BFD and the theoretical lower bound,
      the problem is 'loose' or 'complex'. We use a Ruin-and-Recreate strategy 
      to explore the configuration space more aggressively.
    """
    start_time = time.time()
    num_items = instance['num_items']
    
    # 1. Setup Initial Solution
    bfd_bins = tools['best_fit_decreasing']()
    best_sol = {'num_bins': len(bfd_bins), 'bins': bfd_bins}
    lb = tools['lower_bound']()
    
    # Heuristic Dispatch criterion: Gap between BFD and Lower Bound
    # If gap is 0 or very small, use local hill climbing to find a perfect fit.
    # If gap is large, use global search (Ruin-and-Recreate).
    gap = best_sol['num_bins'] - lb
    
    if gap <= 1 or num_items < 30:
        # Hill Climbing (Polishing)
        current_sol = best_sol
        while time.time() - start_time < time_limit_s * 0.85:
            if current_sol['num_bins'] <= lb:
                break
            
            # Select random item to move
            b_idx = random.randrange(len(current_sol['bins']))
            if not current_sol['bins'][b_idx]:
                continue
            item_idx = random.choice(current_sol['bins'][b_idx])
            
            # Try to move to a different bin
            target_bin = tools['find_smallest_bin_that_fits'](item_idx, current_sol)
            if target_bin != -1 and target_bin != b_idx:
                new_sol = tools['apply_move_item'](current_sol, item_idx, b_idx, target_bin)
                if new_sol and new_sol['num_bins'] <= current_sol['num_bins']:
                    current_sol = new_sol
                    if current_sol['num_bins'] < best_sol['num_bins']:
                        best_sol = current_sol
    else:
        # Ruin and Recreate (Global Search)
        while time.time() - start_time < time_limit_s * 0.85:
            if best_sol['num_bins'] <= lb:
                break
                
            # Ruin: Remove 10-20% of items
            current_bins = [list(b) for b in best_sol['bins'] if len(b) > 0]
            num_to_remove = max(2, int(num_items * 0.15))
            removed = []
            for _ in range(num_to_remove):
                b = random.choice(current_bins)
                if b:
                    removed.append(b.pop(random.randrange(len(b))))
            
            # Clean empty bins
            current_bins = [b for b in current_bins if b]
            
            # Recreate: Best Fit
            removed.sort(key=lambda x: tools['item_size'](x), reverse=True)
            for item in removed:
                idx = tools['find_smallest_bin_that_fits'](item, {'num_bins': len(current_bins), 'bins': current_bins})
                if idx != -1:
                    current_bins[idx].append(item)
                else:
                    current_bins.append([item])
            
            candidate = {'num_bins': len(current_bins), 'bins': current_bins}
            if candidate['num_bins'] <= best_sol['num_bins']:
                best_sol = candidate
                
    return best_sol