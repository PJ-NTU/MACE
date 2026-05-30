# MACE evolved heuristic 05/10 for problem: port_scheduling_problem
import time
import random
import math

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Dispatcher-style heuristic for PSP.
    
    Hypothesis: 
    - Instances with high 'vessel density' (many vessels relative to time horizon 
      and berths) create a highly constrained search space where the Iterated 
      Local Search (A) excels at finding feasible local optima.
    - Instances with sparse constraints allow for broader exploration, where 
      the ALNS-style (B) approach with simulated annealing effectively navigates 
      the trade-off between Unserved-vessel penalties and operational costs.
    """
    start_time = time.time()
    n = instance['vessel_num']
    J = instance['berth_num']
    T = instance['time_periods']
    
    # Calculate density metric: (number of vessels) / (berths * avg_duration_ratio)
    # High density -> high constraint pressure
    avg_duration = sum(instance['vessel_durations']) / max(1, n)
    density = n / max(1, J * (T / max(1, avg_duration)))
    
    # Heuristic: If density > 0.5, prefer the focused ILS (A-Style).
    # Otherwise, prefer the broader ALNS (B-Style).
    use_ils = density > 0.5
    
    def get_empty_sol():
        return {
            'vessel_assignments': {i: None for i in range(n)},
            'inbound_tugboats': {i: [] for i in range(n)},
            'outbound_tugboats': {i: [] for i in range(n)}
        }

    def try_insert(sol, i):
        assignment = tools['find_feasible_assignment'](i, sol)
        if not assignment:
            return sol
        cost = tools['assignment_cost'](i, assignment['berth_id'], assignment['berth_start'], 
                                        assignment['inbound_tugs'], assignment['outbound_tugs'])
        penalty = instance['penalty_parameter'] * instance['vessel_priority_weights'][i]
        if cost < penalty:
            return tools['apply_assignment'](sol, i, assignment['berth_id'], 
                                            assignment['berth_start'], assignment['inbound_tugs'], 
                                            assignment['outbound_tugs'])
        return sol

    # Initial construction
    vessels = sorted(range(n), key=lambda i: instance['vessel_priority_weights'][i], reverse=True)
    sol = get_empty_sol()
    for i in vessels:
        sol = try_insert(sol, i)
    
    best_sol = sol
    best_obj = tools['objective'](sol)
    
    # Optimization Loop
    while time.time() - start_time < time_limit_s * 0.95:
        # Perturbation: Remove subset
        trial_sol = {
            'vessel_assignments': sol['vessel_assignments'].copy(),
            'inbound_tugboats': sol['inbound_tugboats'].copy(),
            'outbound_tugboats': sol['outbound_tugboats'].copy()
        }
        
        assigned = [i for i in range(n) if trial_sol['vessel_assignments'][i] is not None]
        if not assigned: break
        
        num_remove = random.randint(1, min(len(assigned), 4))
        to_remove = random.sample(assigned, num_remove)
        for i in to_remove:
            trial_sol['vessel_assignments'][i] = None
            trial_sol['inbound_tugboats'][i] = []
            trial_sol['outbound_tugboats'][i] = []
            
        # Re-fill
        unassigned = [i for i in range(n) if trial_sol['vessel_assignments'][i] is None]
        random.shuffle(unassigned)
        for i in unassigned:
            trial_sol = try_insert(trial_sol, i)
            
        # Evaluation
        if tools['is_feasible'](trial_sol)[0]:
            new_obj = tools['objective'](trial_sol)
            # ILS logic: greedy acceptance (if use_ils) or SA acceptance (if !use_ils)
            if new_obj < best_obj:
                best_obj = new_obj
                sol = trial_sol
                best_sol = trial_sol
            elif not use_ils and random.random() < 0.1: # SA cooling
                sol = trial_sol
                
    return best_sol