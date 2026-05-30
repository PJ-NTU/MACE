# MACE evolved heuristic 02/10 for problem: set_covering
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Solves the Set Covering Problem using a GRASP-inspired approach:
    1. Initial greedy construction with a randomized cost-effectiveness ratio.
    2. Local search (remove redundant columns).
    3. Iterative refinement using ILP on sub-neighborhoods (LNS) if time permits.
    """
    start_time = time.time()
    m = instance['m']
    n = instance['n']
    
    def get_time_left():
        return time_limit_s - (time.time() - start_time)

    def greedy_randomized_construction(alpha=0.2):
        """Builds a feasible solution using randomized greedy."""
        selected = set()
        uncovered = set(range(1, m + 1))
        
        while uncovered:
            # Calculate cost-effectiveness for all columns
            candidates = []
            for col in range(1, n + 1):
                if col in selected:
                    continue
                
                # Rows this column covers that are currently uncovered
                covers = tools['column_covers'](col)
                new_rows = len(covers.intersection(uncovered))
                
                if new_rows > 0:
                    cost = tools['column_cost'](col)
                    ratio = cost / new_rows
                    candidates.append((ratio, col, new_rows))
            
            if not candidates:
                break
                
            # Sort by ratio
            candidates.sort(key=lambda x: x[0])
            
            # Pick from the top 'alpha' percent of candidates
            limit = max(1, int(len(candidates) * alpha))
            idx = random.randint(0, limit - 1)
            _, col, _ = candidates[idx]
            
            selected.add(col)
            uncovered.difference_update(tools['column_covers'](col))
            
        return list(selected)

    # 1. Initial State
    best_selection = tools['greedy_cover_by_cost_ratio']()
    best_selection = tools['remove_redundant'](best_selection)
    best_cost = tools['cost_of_selection'](best_selection)
    
    # 2. Iterative Improvement
    while get_time_left() > 0.5:
        # LNS-style: Try to solve a sub-problem of the full ILP
        # We relax the problem by fixing a subset of columns and 
        # allowing the ILP solver to optimize the rest.
        
        # Keep 70% of current best, allow ILP to re-optimize the other 30%
        subset_size = int(len(best_selection) * 0.3)
        if subset_size == 0:
            break
            
        must_exclude = random.sample(best_selection, subset_size)
        must_include = [c for c in best_selection if c not in must_exclude]
        
        # Give the ILP solver a small time budget to improve the neighborhood
        sub_time = min(get_time_left(), 2.0)
        new_cols = tools['ilp_solve_cover'](
            must_include=must_include, 
            time_limit_s=sub_time
        )
        
        if new_cols is not None:
            new_cols = tools['remove_redundant'](new_cols)
            new_cost = tools['cost_of_selection'](new_cols)
            if new_cost < best_cost:
                best_cost = new_cost
                best_selection = new_cols
        else:
            # If ILP fails to find a better one, try a fresh randomized greedy
            greedy_cols = greedy_randomized_construction(alpha=0.3)
            greedy_cols = tools['remove_redundant'](greedy_cols)
            if tools['cost_of_selection'](greedy_cols) < best_cost:
                best_selection = greedy_cols
                best_cost = best_cost
                
    return tools['make_solution'](best_selection)