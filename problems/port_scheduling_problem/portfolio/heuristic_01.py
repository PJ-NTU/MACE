# MACE evolved heuristic 01/10 for problem: port_scheduling_problem
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    A GRASP-inspired heuristic for the Port Scheduling Problem.
    1. Sort vessels by priority-to-cost ratio (greedy potential).
    2. Iteratively attempt to insert vessels into the schedule using 
       the provided toolset.
    3. Use a randomized construction approach to explore different 
       scheduling orderings.
    """
    start_time = time.time()
    n = instance['vessel_num']
    
    # Initialize empty solution
    best_solution = {
        'vessel_assignments': {i: None for i in range(n)},
        'inbound_tugboats': {i: [] for i in range(n)},
        'outbound_tugboats': {i: [] for i in range(n)}
    }
    best_obj = float('inf')

    # Calculate heuristic priority: αᵢ / (avg_service_time_cost)
    # Higher priority means we want to fit these first.
    def get_priority(i):
        # Rough estimate of cost-benefit
        return instance['vessel_priority_weights'][i]

    vessels = list(range(n))
    
    # Main loop: Try different orderings / randomized greedy constructions
    while time.time() - start_time < time_limit_s * 0.85:
        # Sort by priority with some noise (GRASP)
        shuffled_vessels = sorted(
            vessels, 
            key=lambda i: get_priority(i) * random.uniform(0.8, 1.2), 
            reverse=True
        )
        
        current_sol = {
            'vessel_assignments': {i: None for i in range(n)},
            'inbound_tugboats': {i: [] for i in range(n)},
            'outbound_tugboats': {i: [] for i in range(n)}
        }
        
        for i in shuffled_vessels:
            # Check if time remains
            if time.time() - start_time > time_limit_s * 0.95:
                break
                
            # Try finding a feasible assignment
            assignment = tools['find_feasible_assignment'](i, current_sol)
            
            if assignment:
                # Evaluate if it's worth adding (compare to penalty)
                # Simple heuristic: always take if feasible
                current_sol = tools['apply_assignment'](
                    current_sol, 
                    i, 
                    assignment['berth_id'], 
                    assignment['berth_start'],
                    assignment['inbound_tugs'],
                    assignment['outbound_tugs']
                )
        
        # Calculate objective
        try:
            current_obj = tools['objective'](current_sol)
            if current_obj < best_obj:
                best_obj = current_obj
                best_solution = current_sol
        except Exception:
            continue
            
    return best_solution