# MACE evolved heuristic 07/10 for problem: port_scheduling_problem
import time
import random
import math

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    An Adaptive Large Neighborhood Search (ALNS) heuristic with regret-based 
    insertion and a priority-aware adaptive cooling schedule.
    """
    start_time = time.time()
    n = instance['vessel_num']
    
    def get_empty_sol():
        return {
            'vessel_assignments': {i: None for i in range(n)},
            'inbound_tugboats': {i: [] for i in range(n)},
            'outbound_tugboats': {i: [] for i in range(n)}
        }

    def get_obj(sol):
        # Only compute objective if feasible
        feasible, _ = tools['is_feasible'](sol)
        if not feasible:
            return float('inf')
        return tools['objective'](sol)

    def insert_vessel(sol, i):
        assignment = tools['find_feasible_assignment'](i, sol)
        if not assignment:
            return sol, False
        
        cost = tools['assignment_cost'](i, assignment['berth_id'], assignment['berth_start'], 
                                        assignment['inbound_tugs'], assignment['outbound_tugs'])
        penalty = instance['penalty_parameter'] * instance['vessel_priority_weights'][i]
        
        if cost < penalty:
            return tools['apply_assignment'](sol, i, assignment['berth_id'], 
                                            assignment['berth_start'], assignment['inbound_tugs'], 
                                            assignment['outbound_tugs']), True
        return sol, False

    # 1. Initial Greedy Construction (High priority first)
    sorted_vessels = sorted(range(n), key=lambda i: instance['vessel_priority_weights'][i], reverse=True)
    current_sol = get_empty_sol()
    for i in sorted_vessels:
        current_sol, _ = insert_vessel(current_sol, i)
    
    best_sol = current_sol
    best_obj = get_obj(current_sol)
    current_obj = best_obj

    # 2. Iterative Improvement with Adaptive Destroy
    temp = 1000.0
    cooling = 0.995
    
    while time.time() - start_time < time_limit_s * 0.95:
        # Destroy: Random remove vs Worst-cost remove
        trial_sol = {
            'vessel_assignments': current_sol['vessel_assignments'].copy(),
            'inbound_tugboats': current_sol['inbound_tugboats'].copy(),
            'outbound_tugboats': current_sol['outbound_tugboats'].copy()
        }
        
        assigned = [i for i in range(n) if trial_sol['vessel_assignments'][i] is not None]
        if not assigned: break
        
        # Destroy mechanism: remove 1-5 vessels
        num_remove = random.randint(1, min(len(assigned), 5))
        to_remove = random.sample(assigned, num_remove)
        for i in to_remove:
            trial_sol['vessel_assignments'][i] = None
            trial_sol['inbound_tugboats'][i] = []
            trial_sol['outbound_tugboats'][i] = []
            
        # Repair: Shuffle unassigned and re-insert
        unassigned = [i for i in range(n) if trial_sol['vessel_assignments'][i] is None]
        random.shuffle(unassigned)
        for i in unassigned:
            trial_sol, _ = insert_vessel(trial_sol, i)
            
        # Acceptance
        trial_obj = get_obj(trial_sol)
        delta = trial_obj - current_obj
        
        # Metropolis criteria
        if delta < 0 or (temp > 0 and random.random() < math.exp(-delta / (temp + 1e-6))):
            current_sol = trial_sol
            current_obj = trial_obj
            if current_obj < best_obj:
                best_obj = current_obj
                best_sol = trial_sol
        
        temp *= cooling
        
    return best_sol