# MACE evolved heuristic 04/10 for problem: set_covering
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Solve the Set Covering Problem using a Greedy initialization followed by
    a Randomized Local Search (LNS-style) to refine the solution within the 
    time limit.
    """
    start_time = time.time()
    
    # 1. Initial feasible solution using the greedy heuristic
    # The greedy_cover_by_cost_ratio is a standard, robust starting point.
    best_selection = tools['greedy_cover_by_cost_ratio']()
    best_selection = tools['remove_redundant'](best_selection)
    best_cost = tools['cost_of_selection'](best_selection)
    
    # 2. Iterative Improvement (Randomized Local Search / LNS)
    # While we have time, try to remove a subset of columns and repair the 
    # cover using a greedy strategy to see if we can find a cheaper valid set.
    while time.time() - start_time < time_limit_s * 0.8:
        if not best_selection:
            break
            
        # Create a candidate by removing a random subset of columns
        # (e.g., remove 10-20% of the current set)
        drop_count = max(1, len(best_selection) // 5)
        candidate = list(best_selection)
        for _ in range(drop_count):
            if candidate:
                candidate.pop(random.randrange(len(candidate)))
        
        # Repair the cover
        uncovered = list(tools['uncovered_rows'](candidate))
        if uncovered:
            # Simple greedy repair: pick cheapest columns covering the remaining rows
            random.shuffle(uncovered)
            for row in uncovered:
                # Find cheapest column covering this row not already in candidate
                best_col = None
                best_val = float('inf')
                for col in tools['columns_covering_row'](row):
                    cost = tools['column_cost'](col)
                    if cost < best_val:
                        best_val = cost
                        best_col = col
                if best_col is not None:
                    candidate.append(best_col)
        
        # Post-process to remove redundant columns
        candidate = tools['remove_redundant'](candidate)
        
        # Check feasibility and cost
        if tools['is_full_cover'](candidate):
            current_cost = tools['cost_of_selection'](candidate)
            if current_cost < best_cost:
                best_cost = current_cost
                best_selection = candidate
        
        # Yield to safety
        if random.random() < 0.01:
            if time.time() - start_time > time_limit_s:
                break
                
    return tools['make_solution'](best_selection)