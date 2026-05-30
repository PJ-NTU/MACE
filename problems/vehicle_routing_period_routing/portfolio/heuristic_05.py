# MACE evolved heuristic 05/10 for problem: vehicle_routing_period_routing
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Optimized heuristic for the Period Vehicle Routing Problem.
    
    Strategy:
    1. Initial solution via greedy assignment and robust routing.
    2. Local search based on randomized schedule perturbation.
    3. Strict feasibility monitoring using the tools provided.
    4. Adaptive time management to prioritize finding a valid solution before optimizing.
    """
    start_time = time.time()
    
    # 1. Initialization: Start with a strong greedy assignment
    # We use greedy assignment as it's faster and more reliable than raw ILP in tight limits
    current_schedules = tools['assign_schedules_greedy'](seed=42)
    
    def get_tours_and_cost(schedules):
        """Constructs routes for the current schedule and returns (tours, total_dist)."""
        all_tours = {}
        total_dist = 0.0
        
        for day in range(1, instance['period_length'] + 1):
            cust_ids = tools['period_required_customers'](day, schedules)
            if not cust_ids:
                all_tours[day] = []
                continue
            
            # Solve daily CVRP
            day_tours = tools['solve_period_routing'](day, cust_ids, time_limit_s=0.1)
            
            # Verify feasibility
            valid, _ = tools['period_routes_valid'](day, day_tours)
            if not valid:
                return None, float('inf')
            
            # Refine routes
            refined_tours = []
            for tour in day_tours:
                if len(tour) > 3:
                    refined_tours.append(tools['apply_2opt_route'](tour, time_limit_s=0.02))
                else:
                    refined_tours.append(tour)
            
            all_tours[day] = refined_tours
            for tour in refined_tours:
                for i in range(len(tour) - 1):
                    total_dist += tools['distance'](tour[i], tour[i+1])
                    
        return all_tours, total_dist

    # Get baseline
    best_tours, best_cost = get_tours_and_cost(current_schedules)
    
    # If baseline is invalid, fallback to default
    if best_tours is None:
        return tools['solve_default'](time_limit_s=time_limit_s * 0.8)
    
    best_schedules = current_schedules.copy()
    
    # 2. Local Search: Hill climbing with randomized schedule swaps
    # Focus on perturbing the schedules of customers with high demand
    cust_ids = list(best_schedules.keys())
    
    while time.time() - start_time < time_limit_s * 0.85:
        # Create a candidate by perturbing a small number of customers
        candidate_schedules = best_schedules.copy()
        num_perturb = random.randint(1, max(1, len(cust_ids) // 10))
        
        for _ in range(num_perturb):
            cid = random.choice(cust_ids)
            candidates = tools['customer_candidate_schedules'](cid)
            if len(candidates) > 1:
                candidate_schedules[cid] = random.choice(candidates)
        
        # Evaluate
        tours, cost = get_tours_and_cost(candidate_schedules)
        
        # Update if better and feasible
        if tours is not None and cost < best_cost:
            best_cost = cost
            best_schedules = candidate_schedules
            best_tours = tours
            
    return tools['make_solution'](best_schedules, best_tours)