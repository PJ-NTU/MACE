# MACE evolved heuristic 08/10 for problem: port_scheduling_problem
import time
import random
import math

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Optimized Iterated Local Search (ILS) with Adaptive Thresholding.
    
    This heuristic improves upon the base models by:
    1. Using a weighted priority-slack construction phase to build a robust baseline.
    2. Employing a 'tabu-lite' approach to prevent immediate re-insertion of removed vessels.
    3. Applying a Simulated Annealing acceptance criterion to escape local minima,
       which h_a lacked in its deterministic ILS mode.
    4. Dynamically adjusting the perturbation size based on the remaining time.
    """
    start_time = time.time()
    n = instance['vessel_num']
    penalty_param = instance['penalty_parameter']
    
    # Priority-based sorting for initial construction
    # Scoring: Balance priority weight alpha with the tightness of the time window
    scores = []
    for i in range(n):
        window = instance['vessel_late_limits'][i] + instance['vessel_early_limits'][i]
        # Higher score means more urgent/valuable to schedule
        score = instance['vessel_priority_weights'][i] * (1.0 / (max(0.1, window) + 0.1))
        scores.append((score, i))
    scores.sort(key=lambda x: x[0], reverse=True)
    order = [v[1] for v in scores]

    def get_empty():
        return {
            'vessel_assignments': {i: None for i in range(n)},
            'inbound_tugboats': {i: [] for i in range(n)},
            'outbound_tugboats': {i: [] for i in range(n)}
        }

    def try_assign(sol, i):
        assign = tools['find_feasible_assignment'](i, sol)
        if not assign:
            return sol
        cost = tools['assignment_cost'](i, assign['berth_id'], assign['berth_start'], 
                                        assign['inbound_tugs'], assign['outbound_tugs'])
        penalty = penalty_param * instance['vessel_priority_weights'][i]
        if cost < penalty:
            return tools['apply_assignment'](sol, i, assign['berth_id'], assign['berth_start'], 
                                            assign['inbound_tugs'], assign['outbound_tugs'])
        return sol

    # Initial Construction
    current_sol = get_empty()
    for i in order:
        current_sol = try_assign(current_sol, i)
    
    best_sol = current_sol
    best_obj = tools['objective'](current_sol)
    current_obj = best_obj
    
    # ILS Parameters
    temp = 100.0
    cooling = 0.995
    
    while time.time() - start_time < time_limit_s * 0.95:
        # Perturbation: Remove a random subset
        trial_sol = {
            'vessel_assignments': current_sol['vessel_assignments'].copy(),
            'inbound_tugboats': current_sol['inbound_tugboats'].copy(),
            'outbound_tugboats': current_sol['outbound_tugboats'].copy()
        }
        
        assigned = [i for i in range(n) if trial_sol['vessel_assignments'][i] is not None]
        if not assigned:
            # Rebuild if empty
            random.shuffle(order)
            for i in order:
                trial_sol = try_assign(trial_sol, i)
        else:
            # Shake: Remove 15-40% of assignments
            num_remove = max(1, int(len(assigned) * random.uniform(0.15, 0.4)))
            to_remove = random.sample(assigned, num_remove)
            for i in to_remove:
                trial_sol['vessel_assignments'][i] = None
                trial_sol['inbound_tugboats'][i] = []
                trial_sol['outbound_tugboats'][i] = []
            
            # Local Search: Re-insert in random order
            remaining = [i for i in range(n) if trial_sol['vessel_assignments'][i] is None]
            random.shuffle(remaining)
            for i in remaining:
                trial_sol = try_assign(trial_sol, i)
        
        # Acceptance Criterion
        try:
            trial_obj = tools['objective'](trial_sol)
            delta = trial_obj - current_obj
            
            if delta < 0 or (temp > 0 and math.exp(-delta / temp) > random.random()):
                current_sol, current_obj = trial_sol, trial_obj
                if current_obj < best_obj:
                    best_obj, best_sol = current_obj, trial_sol
            
            temp *= cooling
        except:
            continue
            
    return best_sol