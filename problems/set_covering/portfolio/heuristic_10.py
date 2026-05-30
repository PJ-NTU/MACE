# MACE evolved heuristic 10/10 for problem: set_covering
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Synthesized heuristic:
    1. Initial baseline from ILP solver for high-quality starting point.
    2. LNS-based local search:
       - Destruct: Remove a random subset of columns (size 10-25%).
       - Repair: Greedy repair using cost-effectiveness (cost/new_coverage).
       - Optimize: Redundancy removal and ILP-based local fine-tuning.
    """
    start_time = time.time()
    
    # 1. Start with an optimal or near-optimal baseline
    best_sol_list = tools['ilp_solve_cover'](time_limit_s=min(time_limit_s * 0.3, 2.0))
    if best_sol_list is None:
        best_sol_list = tools['greedy_cover_by_cost_ratio']()
    
    best_sol_list = tools['remove_redundant'](best_sol_list)
    best_cost = tools['cost_of_selection'](best_sol_list)
    
    # 2. Iterative Local Search (LNS)
    # We use a mix of random removal and greedy repair to explore the landscape
    while time.time() - start_time < time_limit_s * 0.9:
        # Destruct: Remove a random portion
        if len(best_sol_list) <= 1:
            break
            
        remove_n = max(1, int(len(best_sol_list) * random.uniform(0.1, 0.3)))
        current = list(best_sol_list)
        random.shuffle(current)
        for _ in range(remove_n):
            current.pop()
            
        # Repair: Greedy by cost-effectiveness
        uncovered = tools['uncovered_rows'](current)
        while uncovered:
            # Pick a row to cover
            target_row = next(iter(uncovered))
            best_ratio = float('inf')
            best_col = None
            
            # Find best column to cover this row
            for col in tools['columns_covering_row'](target_row):
                covers = tools['column_covers'](col)
                new_covered = len(covers.intersection(uncovered))
                if new_covered > 0:
                    ratio = tools['column_cost'](col) / new_covered
                    if ratio < best_ratio:
                        best_ratio = ratio
                        best_col = col
            
            if best_col is None:
                break
                
            current.append(best_col)
            uncovered = tools['uncovered_rows'](current)
            
        # Refine: Redundancy removal
        current = tools['remove_redundant'](current)
        
        # Evaluate
        if tools['is_full_cover'](current):
            cost = tools['cost_of_selection'](current)
            if cost < best_cost:
                best_cost = cost
                best_sol_list = current
        
        # Local Fine-tuning (occasional ILP injection)
        if random.random() < 0.1 and (time.time() - start_time < time_limit_s * 0.7):
            # Attempt to improve the current set by letting ILP optimize a subset
            # Randomly lock some columns to keep the search local
            subset = random.sample(current, min(len(current), 10))
            improved = tools['ilp_solve_cover'](must_include=subset, time_limit_s=0.5)
            if improved:
                improved = tools['remove_redundant'](improved)
                if tools['cost_of_selection'](improved) < best_cost:
                    best_cost = tools['cost_of_selection'](improved)
                    best_sol_list = improved
                    
    return tools['make_solution'](best_sol_list)