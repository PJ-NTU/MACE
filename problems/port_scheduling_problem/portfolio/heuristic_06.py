# MACE evolved heuristic 06/10 for problem: port_scheduling_problem
import time
import random
import math

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Dispatcher-style solver for the Port Scheduling Problem.
    
    Heuristic Hypothesis:
    - Instances with high density (n >> J) and tight constraints are "bottleneck-heavy".
      These benefit from Parent A's structured Iterated Local Search (ILS) which 
      aggressively prunes and re-inserts based on priority weights.
    - Instances with sparse or loosely constrained workloads benefit from Parent B's 
      Adaptive Local Search, which uses a 'Benefit Density' score to find high-value 
      local optima faster.
    """
    start_time = time.time()
    n = instance['vessel_num']
    J = instance['berth_num']
    T = instance['time_periods']
    
    # Feature extraction: Determine density
    # Density = n / (J * (T/avg_duration))
    avg_dur = sum(instance['vessel_durations']) / max(1, n)
    density = n / (J * (T / max(1, avg_dur)))
    
    # Dispatch Logic
    # High density -> Use ILS (structural stability)
    # Low density -> Use Adaptive Randomized Greedy (exploration of sparse space)
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
        
        cost = tools['assignment_cost'](
            i, assignment['berth_id'], assignment['berth_start'],
            assignment['inbound_tugs'], assignment['outbound_tugs']
        )
        penalty = instance['penalty_parameter'] * instance['vessel_priority_weights'][i]
        
        if cost < penalty:
            return tools['apply_assignment'](
                sol, i, assignment['berth_id'], assignment['berth_start'],
                assignment['inbound_tugs'], assignment['outbound_tugs']
            )
        return sol

    # Construction Phase
    if use_ils:
        # Sort by priority/window (A-style)
        scores = [(instance['vessel_priority_weights'][i] / (1.0 + instance['vessel_early_limits'][i] + instance['vessel_late_limits'][i]), i) for i in range(n)]
        vessel_order = [x[1] for x in sorted(scores, key=lambda x: x[0], reverse=True)]
    else:
        # Sort by priority/footprint (B-style)
        scores = [(instance['vessel_priority_weights'][i] / (1.0 + instance['vessel_durations'][i] * 0.1), i) for i in range(n)]
        vessel_order = [x[1] for x in sorted(scores, key=lambda x: x[0], reverse=True)]

    current_sol = get_empty_sol()
    for i in vessel_order:
        current_sol = try_insert(current_sol, i)
    
    best_sol = current_sol
    try:
        best_obj = tools['objective'](best_sol)
    except:
        best_obj = float('inf')

    # Optimization Loop
    while time.time() - start_time < time_limit_s * 0.92:
        trial_sol = {
            'vessel_assignments': best_sol['vessel_assignments'].copy(),
            'inbound_tugboats': best_sol['inbound_tugboats'].copy(),
            'outbound_tugboats': best_sol['outbound_tugboats'].copy()
        }
        
        assigned = [i for i in range(n) if trial_sol['vessel_assignments'][i] is not None]
        
        if assigned:
            # ILS style: remove 33%, B-style: remove 1-5
            remove_count = max(1, len(assigned) // 3) if use_ils else random.randint(1, min(len(assigned), 5))
            for i in random.sample(assigned, remove_count):
                trial_sol['vessel_assignments'][i] = None
                trial_sol['inbound_tugboats'][i] = []
                trial_sol['outbound_tugboats'][i] = []
        
        # Re-insertion
        shuffled = list(vessel_order)
        random.shuffle(shuffled)
        for i in shuffled:
            if trial_sol['vessel_assignments'][i] is None:
                trial_sol = try_insert(trial_sol, i)
        
        try:
            new_obj = tools['objective'](trial_sol)
            if new_obj < best_obj:
                best_obj = new_obj
                best_sol = trial_sol
        except:
            continue
            
    return best_sol