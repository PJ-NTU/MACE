# MACE evolved heuristic 06/10 for problem: vehicle_routing_period_routing
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Hybrid heuristic for PVRP:
    - Uses ILP-based warm start for schedule assignment.
    - Employs a local search strategy with demand-weighted perturbation.
    - Uses a robust fallback to the default solver if the local search fails to produce a feasible solution.
    """
    start_time = time.time()
    period_length = instance["period_length"]
    
    # 1. Warm start: ILP assignment is generally superior to random greedy
    schedules = tools['ilp_schedule_assignment'](time_limit_s=max(0.5, time_limit_s * 0.2))
    if schedules is None:
        schedules = tools['assign_schedules_greedy']()
    
    best_schedules = schedules.copy()
    best_tours = {}
    best_cost = float('inf')
    
    # Pre-sort customers by demand to focus perturbation on high-impact nodes
    cust_ids = [c['id'] for c in instance['customers']]
    cust_weights = [tools['customer_demand'](cid) + 1.0 for cid in cust_ids]

    def evaluate(scheds):
        tours = {}
        total_dist = 0.0
        for d in range(1, period_length + 1):
            req = tools['period_required_customers'](d, scheds)
            if not req:
                tours[d] = []
                continue
            
            # Budgeted routing
            day_tours = tools['solve_period_routing'](d, req, time_limit_s=max(0.1, (time_limit_s * 0.7) / (period_length * 5)))
            
            # Check feasibility
            valid, _ = tools['period_routes_valid'](d, day_tours)
            if not valid:
                return None, None
            
            tours[d] = day_tours
            for tour in day_tours:
                for i in range(len(tour) - 1):
                    total_dist += tools['distance'](tour[i], tour[i+1])
        return tours, total_dist

    # Initial evaluation
    initial_tours, initial_cost = evaluate(best_schedules)
    if initial_tours is not None:
        best_tours = initial_tours
        best_cost = initial_cost

    # 2. Local Search
    # We use a time-limited loop to explore neighbors via schedule mutations
    while time.time() - start_time < time_limit_s * 0.85:
        # Clone and mutate
        candidate_schedules = best_schedules.copy()
        # Mutate 1-2 customers to keep moves local and high-probability of feasibility
        for _ in range(random.randint(1, 2)):
            cid = random.choices(cust_ids, weights=cust_weights, k=1)[0]
            candidate_schedules[cid] = random.choice(tools['customer_candidate_schedules'](cid))
            
        tours, cost = evaluate(candidate_schedules)
        if tours is not None and cost < best_cost:
            best_cost = cost
            best_schedules = candidate_schedules
            best_tours = tours
            
    # 3. Final check / Fallback
    if best_cost == float('inf'):
        # If local search failed to find a valid solution, return the default reliable one
        return tools['solve_default'](time_limit_s=max(0.1, time_limit_s - (time.time() - start_time)))

    return {"selected_schedules": best_schedules, "tours": best_tours}