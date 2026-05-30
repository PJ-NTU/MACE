# MACE evolved heuristic 07/10 for problem: vehicle_routing_period_routing
import time
import random
import math

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Redesigned PVRP Heuristic:
    1. Robust initial schedule assignment using ILP with a focus on load balancing.
    2. Iterative improvement using Simulated Annealing on the schedule space.
    3. Efficient daily route generation using cached TSP heuristics.
    """
    start_time = time.time()
    period_length = instance["period_length"]
    
    # 1. Initialization
    initial_schedules = tools['ilp_schedule_assignment'](time_limit_s=min(time_limit_s * 0.3, 5.0))
    if initial_schedules is None:
        initial_schedules = tools['assign_schedules_greedy']()
    
    current_schedules = initial_schedules.copy()
    
    def get_solution_cost(schedules):
        total_dist = 0.0
        tours_dict = {}
        for d in range(1, period_length + 1):
            custs = tools['period_required_customers'](d, schedules)
            if not custs:
                tours_dict[d] = []
                continue
            
            # Use provided solve_period_routing but enforce capacity constraints
            day_tours = tools['solve_period_routing'](d, custs, time_limit_s=0.1)
            valid, _ = tools['period_routes_valid'](d, day_tours)
            if not valid:
                return None, None
            
            tours_dict[d] = day_tours
            for tour in day_tours:
                for i in range(len(tour) - 1):
                    total_dist += tools['distance'](tour[i], tour[i+1])
        return tours_dict, total_dist

    best_schedules, best_tours = current_schedules.copy(), {}
    best_cost = float('inf')
    
    # Initial evaluation
    tours, cost = get_solution_cost(current_schedules)
    if tours is not None:
        best_schedules, best_tours, best_cost = current_schedules.copy(), tours, cost

    # 2. Simulated Annealing / Local Search
    cust_ids = list(current_schedules.keys())
    temp = 1.0
    cooling_rate = 0.99
    
    while time.time() - start_time < time_limit_s * 0.85:
        # Move: Pick one customer and change its schedule
        cid = random.choice(cust_ids)
        old_sched = current_schedules[cid]
        new_sched = random.choice(tools['customer_candidate_schedules'](cid))
        
        if new_sched == old_sched:
            continue
            
        current_schedules[cid] = new_sched
        tours, cost = get_solution_cost(current_schedules)
        
        # Acceptance criteria
        if tours is not None:
            if cost < best_cost:
                best_cost = cost
                best_schedules = current_schedules.copy()
                best_tours = tours
            elif random.random() < math.exp((best_cost - cost) / (temp + 1e-9)):
                # Accept worse solution
                pass
            else:
                # Revert
                current_schedules[cid] = old_sched
        else:
            # Infeasible, revert
            current_schedules[cid] = old_sched
            
        temp *= cooling_rate
        
    # 3. Final check
    if best_cost == float('inf'):
        # Fallback to default if no valid solution found
        return tools['solve_default'](time_limit_s=max(0.1, time_limit_s - (time.time() - start_time)))
        
    return {"selected_schedules": best_schedules, "tours": best_tours}