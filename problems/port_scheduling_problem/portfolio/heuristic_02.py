# MACE evolved heuristic 02/10 for problem: port_scheduling_problem
import time
import random
import math

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Improved PSP heuristic:
    1. Sorts vessels by a 'Benefit Density' score (Priority / Expected Resource Usage).
    2. Uses a randomized greedy construction to build a high-quality initial solution.
    3. Performs an Iterated Local Search (ILS) with a focus on 'Swap' and 'Retry'
       moves, using time-limit-aware iterations to balance exploration and 
       exploitation.
    """
    start_time = time.time()
    n = instance['vessel_num']
    m = instance['penalty_parameter']
    
    # Pre-calculate priority density
    def get_vessel_score(i):
        # Higher score = more likely to be included
        alpha = instance['vessel_priority_weights'][i]
        # Normalize cost factors to estimate resource footprint
        cost_factor = (instance['vessel_waiting_costs'][i] + instance['vessel_jit_costs'][i]) * 0.05
        return alpha / (1.0 + cost_factor)

    vessels = sorted(range(n), key=get_vessel_score, reverse=True)

    def get_empty_sol():
        return {
            'vessel_assignments': {i: None for i in range(n)},
            'inbound_tugboats': {i: [] for i in range(n)},
            'outbound_tugboats': {i: [] for i in range(n)}
        }

    def try_insert(sol, i):
        # Use find_feasible_assignment which is optimized for time-windows
        assignment = tools['find_feasible_assignment'](i, sol)
        if not assignment:
            return sol
        
        cost = tools['assignment_cost'](
            i, 
            assignment['berth_id'], 
            assignment['berth_start'], 
            assignment['inbound_tugs'], 
            assignment['outbound_tugs']
        )
        
        # Only accept if individual assignment cost is lower than the penalty
        if cost < (m * instance['vessel_priority_weights'][i]):
            return tools['apply_assignment'](
                sol, i, assignment['berth_id'], 
                assignment['berth_start'], assignment['inbound_tugs'], 
                assignment['outbound_tugs']
            )
        return sol

    # 1. Construction Phase
    current_sol = get_empty_sol()
    for i in vessels:
        current_sol = try_insert(current_sol, i)
        
    best_sol = current_sol
    best_obj = tools['objective'](best_sol)

    # 2. Improvement Phase (ILS)
    # Perform random removal of some vessels to see if higher-priority ones can fit
    while time.time() - start_time < time_limit_s * 0.95:
        trial_sol = {
            'vessel_assignments': current_sol['vessel_assignments'].copy(),
            'inbound_tugboats': current_sol['inbound_tugboats'].copy(),
            'outbound_tugboats': current_sol['outbound_tugboats'].copy()
        }
        
        # Perturbation: Remove a random subset of assigned vessels
        assigned = [i for i in range(n) if trial_sol['vessel_assignments'][i] is not None]
        if not assigned:
            break
            
        num_remove = random.randint(1, min(len(assigned), 3))
        to_remove = random.sample(assigned, num_remove)
        for i in to_remove:
            trial_sol['vessel_assignments'][i] = None
            trial_sol['inbound_tugboats'][i] = []
            trial_sol['outbound_tugboats'][i] = []
            
        # Re-fill with shuffled order
        unassigned = [i for i in range(n) if trial_sol['vessel_assignments'][i] is None]
        random.shuffle(unassigned)
        for i in unassigned:
            trial_sol = try_insert(trial_sol, i)
            
        # Evaluate
        is_feas, _ = tools['is_feasible'](trial_sol)
        if is_feas:
            new_obj = tools['objective'](trial_sol)
            if new_obj < best_obj:
                best_obj = new_obj
                current_sol = trial_sol
                best_sol = trial_sol
            elif random.random() < 0.05: # Simulated annealing-like acceptance
                current_sol = trial_sol
                
    return best_sol