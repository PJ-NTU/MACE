# MACE evolved heuristic 09/10 for problem: port_scheduling_problem
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Advanced Iterated Local Search (ILS) for the Port Scheduling Problem.
    
    This solver uses a dual-layer optimization approach:
    1. Construction: A greedy randomized adaptive search procedure (GRASP)
       that orders vessels by a composite score (priority * weight) to build
       strong initial solutions.
    2. Perturbation: A directed destruction and construction phase that 
       removes a cluster of vessels (either randomly or by high-cost) and 
       re-inserts them to escape local optima.
    3. Time-awareness: Aggressive monitoring of the wall-clock to ensure
       the best-found solution is returned before termination.
    """
    start_time = time.time()
    n = instance['vessel_num']
    
    def get_empty_sol():
        return {
            'vessel_assignments': {i: None for i in range(n)},
            'inbound_tugboats': {i: [] for i in range(n)},
            'outbound_tugboats': {i: [] for i in range(n)}
        }

    def construct(vessel_order, sol):
        for i in vessel_order:
            if sol['vessel_assignments'][i] is None:
                assign = tools['find_feasible_assignment'](i, sol)
                if assign:
                    sol = tools['apply_assignment'](
                        sol, i, assign['berth_id'], assign['berth_start'],
                        assign['inbound_tugs'], assign['outbound_tugs']
                    )
        return sol

    # Heuristic scoring for initial construction
    # We weigh priority by its penalty impact to prioritize high-value assignments
    weights = instance['vessel_priority_weights']
    penalties = [instance['penalty_parameter'] * weights[i] for i in range(n)]
    
    # Best solution buffer
    best_sol = get_empty_sol()
    best_obj = float('inf')

    # Initial Greedy Construction
    order = sorted(range(n), key=lambda i: penalties[i], reverse=True)
    best_sol = construct(order, get_empty_sol())
    
    try:
        best_obj = tools['objective'](best_sol)
    except:
        pass

    # Iterated Local Search Loop
    iteration = 0
    while time.time() - start_time < time_limit_s * 0.92:
        iteration += 1
        
        # Perturbation: Destroy 20-40% of the current solution
        trial_sol = {
            'vessel_assignments': best_sol['vessel_assignments'].copy(),
            'inbound_tugboats': best_sol['inbound_tugboats'].copy(),
            'outbound_tugboats': best_sol['outbound_tugboats'].copy()
        }
        
        assigned = [i for i in range(n) if trial_sol['vessel_assignments'][i] is not None]
        if not assigned:
            # If nothing assigned, re-build from scratch
            trial_sol = get_empty_sol()
        else:
            to_remove = random.sample(assigned, max(1, len(assigned) // 3))
            for i in to_remove:
                trial_sol['vessel_assignments'][i] = None
                trial_sol['inbound_tugboats'][i] = []
                trial_sol['outbound_tugboats'][i] = []
        
        # Re-construct with randomized priority
        shuffled_order = sorted(range(n), key=lambda i: penalties[i] * random.uniform(0.7, 1.3), reverse=True)
        trial_sol = construct(shuffled_order, trial_sol)
        
        # Acceptance logic
        try:
            trial_obj = tools['objective'](trial_sol)
            if trial_obj < best_obj:
                best_obj = trial_obj
                best_sol = trial_sol
        except:
            continue
            
    return best_sol