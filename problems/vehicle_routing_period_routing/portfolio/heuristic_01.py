# MACE evolved heuristic 01/10 for problem: vehicle_routing_period_routing
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Synthesized heuristic for the PVRP:
    1. Utilize ILP schedule assignment as a high-quality backbone.
    2. Iterate over schedule assignments using a local search (perturbation)
       to explore the space of assignments.
    3. For each assignment, solve daily routing using the robust provided 
       solve_period_routing tool, ensuring hard constraints are met.
    """
    start_time = time.time()
    period_length = instance["period_length"]
    
    # 1. Warm start with ILP
    schedules = tools['ilp_schedule_assignment'](time_limit_s=max(1.0, time_limit_s * 0.2))
    if schedules is None:
        schedules = tools['assign_schedules_greedy']()
    
    best_schedules = schedules.copy()
    best_tours = {}
    best_cost = float('inf')
    
    def evaluate_schedules(scheds):
        tours = {}
        total_dist = 0.0
        for d in range(1, period_length + 1):
            cust_ids = tools['period_required_customers'](d, scheds)
            if not cust_ids:
                tours[d] = []
                continue
            
            day_tours = tools['solve_period_routing'](d, cust_ids, time_limit_s=0.2)
            valid, _ = tools['period_routes_valid'](d, day_tours)
            if not valid:
                return None, None
            
            tours[d] = day_tours
            for tour in day_tours:
                for i in range(len(tour) - 1):
                    total_dist += tools['distance'](tour[i], tour[i+1])
        return tours, total_dist

    # Initial evaluation
    initial_tours, initial_cost = evaluate_schedules(best_schedules)
    if initial_tours is not None:
        best_tours = initial_tours
        best_cost = initial_cost

    # 2. Local Search refinement
    # Explore by swapping schedules for a subset of customers
    cust_ids = list(best_schedules.keys())
    while time.time() - start_time < time_limit_s * 0.9:
        # Perturb
        candidate_schedules = best_schedules.copy()
        for _ in range(max(1, len(cust_ids) // 10)):
            cid = random.choice(cust_ids)
            candidate_schedules[cid] = random.choice(tools['customer_candidate_schedules'](cid))
            
        tours, cost = evaluate_schedules(candidate_schedules)
        if tours is not None and cost < best_cost:
            best_cost = cost
            best_schedules = candidate_schedules
            best_tours = tours
            
    # Final safety check
    if best_cost == float('inf'):
        return tools['solve_default'](time_limit_s=max(0.1, time_limit_s - (time.time() - start_time)))

    return {"selected_schedules": best_schedules, "tours": best_tours}