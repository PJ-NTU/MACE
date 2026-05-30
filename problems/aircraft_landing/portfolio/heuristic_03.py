# MACE evolved heuristic 03/10 for problem: aircraft_landing
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Improved solver using a hybrid approach:
    1. MILP-based seeding (if small) or multiple greedy restarts.
    2. Hill-climbing with random restarts on runway assignments.
    3. Tight local packing for landing times.
    """
    start_time = time.time()
    num_planes = instance["num_planes"]
    time_limit_buffer = time_limit_s * 0.95

    # 1. Try MILP if it's within a reasonable size
    if num_planes <= 40:
        sol = tools.get('ilp_aircraft_landing', lambda t: None)(time_limit_s=min(time_limit_s * 0.3, 10.0))
        if sol is not None:
            return {"schedule": sol}

    # 2. Heuristic search: Multi-start greedy + local improvement
    best_schedule = None
    best_penalty = float('inf')

    def get_packed_schedule(assignments):
        """
        Given runway assignments, pack landing times as early as possible 
        to respect sequence and windows.
        """
        schedule = {}
        # Simple greedy scheduling per runway: 
        # Sort by target time, then pack
        for r in range(1, instance["num_runways"] + 1):
            planes_on_r = [p for p, r_id in assignments.items() if r_id == r]
            planes_on_r.sort(key=lambda i: instance["planes"][i-1]["target"])
            
            last_t = -float('inf')
            last_p = None
            
            for i in planes_on_r:
                p_data = instance["planes"][i-1]
                gap = instance["separation"][last_p-1][i-1] if last_p is not None else 0
                earliest_possible = max(p_data["earliest"], last_t + gap)
                
                if earliest_possible > p_data["latest"]:
                    return None
                
                # Landing time: target if possible, else earliest possible
                landing_time = max(earliest_possible, p_data["target"])
                landing_time = min(landing_time, p_data["latest"])
                
                schedule[i] = {"landing_time": float(landing_time), "runway": r}
                last_t = landing_time
                last_p = i
        return schedule

    # Initial bests
    candidates = [tools['greedy_target_time_construct'](), tools['construct_by_appearance_order']()]
    for cand in candidates:
        if cand:
            feas, _ = tools['validate_schedule'](cand)
            if feas:
                p = tools['penalty_of_schedule'](cand)
                if p < best_penalty:
                    best_penalty = p
                    best_schedule = cand

    # Iterative improvement
    while time.time() - start_time < time_limit_buffer:
        # Random neighborhood: reassign a random plane to a random runway
        if best_schedule:
            curr_assignments = {i: best_schedule[i]["runway"] for i in range(1, num_planes + 1)}
            for _ in range(max(1, num_planes // 10)):
                p = random.randint(1, num_planes)
                curr_assignments[p] = random.randint(1, instance["num_runways"])
            
            candidate = get_packed_schedule(curr_assignments)
            if candidate:
                feas, _ = tools['validate_schedule'](candidate)
                if feas:
                    p = tools['penalty_of_schedule'](candidate)
                    if p < best_penalty:
                        best_penalty = p
                        best_schedule = candidate
                        continue
        
        # Occasional random restart
        if random.random() < 0.1:
            rand_assignments = {i: random.randint(1, instance["num_runways"]) for i in range(1, num_planes + 1)}
            candidate = get_packed_schedule(rand_assignments)
            if candidate:
                feas, _ = tools['validate_schedule'](candidate)
                if feas:
                    p = tools['penalty_of_schedule'](candidate)
                    if p < best_penalty:
                        best_penalty = p
                        best_schedule = candidate

    return {"schedule": best_schedule}