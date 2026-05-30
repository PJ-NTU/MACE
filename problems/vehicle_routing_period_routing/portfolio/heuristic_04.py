# MACE evolved heuristic 04/10 for problem: vehicle_routing_period_routing
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Refined heuristic for the PVRP:
    Uses the parent structure but replaces the random perturbation with a
    demand-weighted mutation, favoring the reassignment of customers
    that contribute most to the daily load variance.
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
    # Weighted mutation: target customers by their demand weight
    cust_list = [(cid, tools['customer_demand'](cid)) for cid in best_schedules.keys()]
    cids = [c[0] for c in cust_list]
    weights = [c[1] for c in cust_list]
    
    while time.time() - start_time < time_limit_s * 0.9:
        candidate_schedules = best_schedules.copy()
        # Mutate a small set, biased towards high-demand customers
        for _ in range(max(1, len(cids) // 10)):
            cid = random.choices(cids, weights=weights, k=1)[0]
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