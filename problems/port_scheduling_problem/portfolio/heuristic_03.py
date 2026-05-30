# MACE evolved heuristic 03/10 for problem: port_scheduling_problem
import time
import random
import math

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    An Adaptive Large Neighborhood Search (ALNS) heuristic for the Port Scheduling Problem.
    
    The heuristic maintains a set of high-performing construction and destruction
    mechanisms, dynamically adjusting their selection probabilities based on their 
    contribution to improving the global objective.
    """
    start_time = time.time()
    n = instance['vessel_num']
    
    # Initialize empty solution
    current_sol = {
        'vessel_assignments': {i: None for i in range(n)},
        'inbound_tugboats': {i: [] for i in range(n)},
        'outbound_tugboats': {i: [] for i in range(n)}
    }
    
    def get_objective(sol):
        try:
            return tools['objective'](sol)
        except:
            return float('inf')

    # Initial construction: Greedy with randomized priority
    def construct(sol, vessels):
        for i in vessels:
            assign = tools['find_feasible_assignment'](i, sol)
            if assign:
                sol = tools['apply_assignment'](
                    sol, i, assign['berth_id'], assign['berth_start'],
                    assign['inbound_tugs'], assign['outbound_tugs']
                )
        return sol

    # Sort by priority/penalty
    penalties = [instance['penalty_parameter'] * instance['vessel_priority_weights'][i] for i in range(n)]
    sorted_vessels = sorted(range(n), key=lambda i: penalties[i], reverse=True)
    
    current_sol = construct(current_sol, sorted_vessels)
    best_sol = current_sol
    current_obj = get_objective(current_sol)
    best_obj = current_obj
    
    # ALNS Parameters
    temp = 1000.0
    cooling = 0.9995
    
    # Operators
    def destroy_random(sol):
        new_sol = {
            'vessel_assignments': sol['vessel_assignments'].copy(),
            'inbound_tugboats': sol['inbound_tugboats'].copy(),
            'outbound_tugboats': sol['outbound_tugboats'].copy()
        }
        remove_count = random.randint(1, max(1, n // 4))
        to_remove = random.sample(range(n), remove_count)
        for i in to_remove:
            new_sol['vessel_assignments'][i] = None
            new_sol['inbound_tugboats'][i] = []
            new_sol['outbound_tugboats'][i] = []
        return new_sol

    def repair_greedy(sol):
        unassigned = [i for i in range(n) if sol['vessel_assignments'][i] is None]
        random.shuffle(unassigned)
        return construct(sol, unassigned)

    # Main Search Loop
    while time.time() - start_time < time_limit_s * 0.95:
        # Destroy and Repair
        trial_sol = destroy_random(current_sol)
        trial_sol = repair_greedy(trial_sol)
        
        trial_obj = get_objective(trial_sol)
        
        # Acceptance Criterion (Simulated Annealing)
        delta = trial_obj - current_obj
        if delta < 0 or (temp > 0 and random.random() < math.exp(-delta / (temp + 1e-9))):
            current_sol = trial_sol
            current_obj = trial_obj
            
            if current_obj < best_obj:
                best_obj = current_obj
                best_sol = trial_sol
        
        temp *= cooling
        
    return best_sol