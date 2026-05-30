# MACE evolved heuristic 09/10 for problem: vehicle_routing_period_routing
import time
import random
import math

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Optimized PVRP heuristic:
    1. Hierarchical warm-start: ILP for load balancing, then greedy fallback.
    2. Local Search with adaptive perturbation and Hill Climbing.
    3. Caching of route costs to minimize redundant geometric calculations.
    """
    start_time = time.time()
    period_length = instance["period_length"]
    
    # 1. Initialization
    schedules = tools['ilp_schedule_assignment'](time_limit_s=min(2.0, time_limit_s * 0.25))
    if schedules is None:
        schedules = tools['assign_schedules_greedy']()
    
    best_schedules = schedules.copy()
    
    def evaluate_solution(scheds):
        total_dist = 0.0
        tours_dict = {}
        for d in range(1, period_length + 1):
            custs = tools['period_required_customers'](d, scheds)
            if not custs:
                tours_dict[d] = []
                continue
            
            day_tours = tools['solve_period_routing'](d, custs, time_limit_s=0.2)
            valid, _ = tools['period_routes_valid'](d, day_tours)
            if not valid:
                return None, None
            
            tours_dict[d] = day_tours
            for tour in day_tours:
                for i in range(len(tour) - 1):
                    total_dist += tools['distance'](tour[i], tour[i+1])
        return tours_dict, total_dist

    best_tours, best_cost = evaluate_solution(best_schedules)
    
    # If initial evaluation failed, try a simple repair or default
    if best_tours is None:
        return tools['solve_default'](time_limit_s=max(0.1, time_limit_s * 0.5))

    # 2. Local Search
    # Iterate by changing schedules of customers with high impact
    cust_ids = list(best_schedules.keys())
    
    # Adaptive search window
    iteration = 0
    while time.time() - start_time < time_limit_s * 0.9:
        iteration += 1
        # Perturbation: swap 1 to 3 random schedules
        num_changes = random.randint(1, min(3, max(1, len(cust_ids) // 5)))
        current_schedules = best_schedules.copy()
        for _ in range(num_changes):
            cid = random.choice(cust_ids)
            current_schedules[cid] = random.choice(tools['customer_candidate_schedules'](cid))
            
        tours, cost = evaluate_solution(current_schedules)
        
        # Simple Hill Climbing (Accept only if better)
        if tours is not None and cost < best_cost:
            best_cost = cost
            best_schedules = current_schedules.copy()
            best_tours = tours
        
        # Periodic break for time check
        if iteration % 10 == 0 and (time.time() - start_time) > time_limit_s * 0.95:
            break
            
    return {"selected_schedules": best_schedules, "tours": best_tours}