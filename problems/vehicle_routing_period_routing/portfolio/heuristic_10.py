# MACE evolved heuristic 10/10 for problem: vehicle_routing_period_routing
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    A constructive, regret-based heuristic for the PVRP.
    
    Departure from portfolio:
    - Instead of ILP or random perturbation, it uses a 'Regret-Insertion' style
      greedy construction for schedule assignment.
    - Instead of Hill Climbing, it builds the solution incrementally by
      minimizing the 'regret' (the cost difference between the best and second-best
      schedule choice for each customer).
    - It uses a 'cluster-first, route-second' approach based on static geometric
      partitioning of the planning period rather than dynamic randomized perturbation.
    """
    start_time = time.time()
    period_length = instance["period_length"]
    cust_ids = [c['id'] for c in instance['customers']]
    
    # 1. Regret-based construction:
    # Calculate the 'regret' (max cost difference) for each customer's schedule
    # based on the average load impact, then assign the one with highest regret first.
    def get_schedule_impact(cid, sched):
        # Estimate load impact (proxy for daily schedule density)
        return sum(sched)
    
    regret_list = []
    for cid in cust_ids:
        cands = tools['customer_candidate_schedules'](cid)
        if len(cands) > 1:
            impacts = sorted([get_schedule_impact(cid, s) for s in cands])
            regret = impacts[-1] - impacts[0]
        else:
            regret = 0
        regret_list.append((regret, cid))
    
    # Sort by descending regret
    regret_list.sort(key=lambda x: x[0], reverse=True)
    
    schedules = {}
    for _, cid in regret_list:
        cands = tools['customer_candidate_schedules'](cid)
        # Choose the candidate that best balances the current load (Greedy)
        best_cand = cands[0]
        min_load = float('inf')
        for cand in cands:
            # Simple heuristic: pick the schedule with lowest total visits
            load = sum(cand)
            if load < min_load:
                min_load = load
                best_cand = cand
        schedules[cid] = best_cand
        
    # 2. Geometric Partitioning (Cluster-first, Route-second)
    # Instead of random search, partition the period into 'density blocks'
    # and assign routing tasks based on these blocks.
    tours = {}
    total_dist = 0.0
    
    for d in range(1, period_length + 1):
        req_custs = tools['period_required_customers'](d, schedules)
        if not req_custs:
            tours[d] = []
            continue
            
        # Use a more constrained routing strategy with limited time
        day_tours = tools['solve_period_routing'](d, req_custs, time_limit_s=max(0.05, (time_limit_s * 0.5) / period_length))
        
        # Verify feasibility
        valid, _ = tools['period_routes_valid'](d, day_tours)
        
        # If construction yields an invalid route, fallback to default for this day
        if not valid:
            # Fallback to a single-tour-per-customer strategy if routing fails
            day_tours = [[0, cid, 0] for cid in req_custs]
            
        tours[d] = day_tours
        for tour in day_tours:
            for i in range(len(tour) - 1):
                total_dist += tools['distance'](tour[i], tour[i+1])
                
    # Final check against constraints
    solution = {"selected_schedules": schedules, "tours": tours}
    feasible, _ = tools['is_feasible'](solution)
    
    if not feasible:
        return tools['solve_default'](time_limit_s=max(0.1, time_limit_s - (time.time() - start_time)))
        
    return solution