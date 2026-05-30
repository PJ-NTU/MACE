# MACE evolved heuristic 07/10 for problem: bin_packing_one_dimensional
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Dispatcher-style heuristic:
    - Small instances (low number of items) are solved with BFD + Local Search (A-style),
      which is more stable for tight search spaces.
    - Large/Complex instances are solved with Iterated Local Search (B-style),
      which provides better diversification and escape from local optima.
    """
    start_time = time.time()
    num_items = instance['num_items']
    items = instance['items']
    bin_capacity = instance['bin_capacity']
    
    # Feature calculation: Density measure
    # If density is very high, items are harder to pack; ILS (B) is better.
    # If number of items is low, hill climbing (A) is sufficient.
    avg_item_size = sum(items) / num_items
    density = avg_item_size / bin_capacity
    
    # Heuristic switch:
    # Use A-style (local hill climbing) if items are few.
    # Use B-style (ILS with Ruin & Recreate) if items are many or density is high.
    if num_items < 50 and density < 0.3:
        # A-style: Best Fit Decreasing + Hill Climbing
        bins = tools['best_fit_decreasing']()
        current_sol = {'num_bins': len(bins), 'bins': bins}
        best_sol = current_sol
        best_num_bins = len(bins)
        
        while time.time() - start_time < time_limit_s * 0.8:
            items_flat = []
            for b_idx, b_items in enumerate(current_sol['bins']):
                for item in b_items:
                    items_flat.append((item, b_idx))
            if not items_flat: break
                
            item_to_move, from_bin_idx = random.choice(items_flat)
            target_bin_idx = tools['find_smallest_bin_that_fits'](item_to_move, current_sol)
            
            if target_bin_idx != -1 and target_bin_idx != from_bin_idx:
                new_sol = tools['apply_move_item'](current_sol, item_to_move, from_bin_idx, target_bin_idx)
                if new_sol:
                    current_sol = new_sol
                    if current_sol['num_bins'] < best_num_bins:
                        best_sol = current_sol
                        best_num_bins = current_sol['num_bins']
            if best_num_bins <= tools['lower_bound'](): break
        return best_sol

    else:
        # B-style: Iterated Local Search with Ruin-and-Recreate
        best_bins = tools['best_fit_decreasing']()
        best_solution = {'num_bins': len(best_bins), 'bins': best_bins}
        
        def finalize(bins):
            clean = [list(b) for b in bins if len(b) > 0]
            return {'num_bins': len(clean), 'bins': clean}

        lower_bound = tools['lower_bound']()
        while time.time() - start_time < time_limit_s * 0.9:
            if best_solution['num_bins'] <= lower_bound: break
            
            current_bins = [list(b) for b in best_solution['bins'] if len(b) > 0]
            # Adaptive ruin size based on time and problem size
            num_to_remove = random.randint(2, max(3, num_items // 10))
            removed_items = []
            for _ in range(num_to_remove):
                b_idx = random.randrange(len(current_bins))
                if current_bins[b_idx]:
                    removed_items.append(current_bins[b_idx].pop(random.randrange(len(current_bins[b_idx]))))
            
            removed_items.sort(key=lambda x: tools['item_size'](x), reverse=True)
            for item in removed_items:
                best_bin_idx = tools['find_smallest_bin_that_fits'](item, {'num_bins': len(current_bins), 'bins': current_bins})
                if best_bin_idx != -1:
                    current_bins[best_bin_idx].append(item)
                else:
                    current_bins.append([item])
            
            candidate_sol = finalize(current_bins)
            if candidate_sol['num_bins'] <= best_solution['num_bins']:
                best_solution = candidate_sol
                
        return best_solution