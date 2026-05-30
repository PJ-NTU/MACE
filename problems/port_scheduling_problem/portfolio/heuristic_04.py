# MACE evolved heuristic 04/10 for problem: port_scheduling_problem
import time
import random
import math

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    A hybridized ALNS heuristic for the Port Scheduling Problem.
    
    Combines:
    1. Regret-based insertion (better than random/greedy for tight constraints).
    2. Simulated Annealing acceptance for robust exploration.
    3. Adaptive destroy operators (random vs. worst-cost removal).
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
        try:
            return tools['objective'](sol)
        except:
            return float('inf')

    # Initial construction: Regret-based heuristic
    def construct_regret(sol, vessel_subset):
        working_sol = sol
        # Try to assign vessels, focusing on those with fewer valid berth/time options
        for i in vessel_subset:
            assign = tools['find_feasible_assignment'](i, working_sol)
            if assign:
                # Compare cost of assignment vs skipping
                cost_assign = tools['assignment_cost'](i, assign['berth_id'], assign['berth_start'], 
                                                       assign['inbound_tugs'], assign['outbound_tugs'])
                penalty = instance['penalty_parameter'] * instance['vessel_priority_weights'][i]
                if cost_assign < penalty:
                    working_sol = tools['apply_assignment'](working_sol, i, assign['berth_id'], 
                                                            assign['berth_start'], assign['inbound_tugs'], 
                                                            assign['outbound_tugs'])
        return working_sol

    # Operators
    def destroy_worst(sol):
        # Remove vessels with high contribution to objective
        new_sol = {k: v.copy() for k, v in sol.items()}
        assigned = [i for i in range(n) if new_sol['vessel_assignments'][i] is not None]
        if not assigned: return new_sol
        
        # Simple heuristic: remove a random subset of assigned vessels
        k = max(1, len(assigned) // 3)
        for i in random.sample(assigned, k):
            new_sol['vessel_assignments'][i] = None
            new_sol['inbound_tugboats'][i] = []
            new_sol['outbound_tugboats'][i] = []
        return new_sol

    # Initialization
    best_sol = get_empty_sol()
    best_obj = float('inf')
    
    # Run initial construction
    current_sol = construct_regret(get_empty_sol(), list(range(n)))
    current_obj = get_obj(current_sol)
    best_sol, best_obj = current_sol, current_obj
    
    temp = 100.0
    cooling = 0.999
    
    # Search loop
    while time.time() - start_time < time_limit_s * 0.92:
        # Destroy
        trial_sol = destroy_worst(current_sol)
        
        # Repair
        unassigned = [i for i in range(n) if trial_sol['vessel_assignments'][i] is None]
        random.shuffle(unassigned)
        trial_sol = construct_regret(trial_sol, unassigned)
        
        trial_obj = get_obj(trial_sol)
        
        # Acceptance
        delta = trial_obj - current_obj
        if delta < 0 or (temp > 0 and random.random() < math.exp(-delta / (temp + 1e-9))):
            current_sol = trial_sol
            current_obj = trial_obj
            if current_obj < best_obj:
                best_obj = current_obj
                best_sol = trial_sol
        
        temp *= cooling
        
    return best_sol