# MACE evolved heuristic 05/10 for problem: aircraft_landing
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Optimized solver using a combination of:
    1. Multi-start randomized greedy construction with bias towards target times.
    2. Iterative local search using both swap landings and runway reassignment.
    3. Strict time budget management.
    """
    start_time = time.time()
    # Reserve a small buffer at the end
    end_time = start_time + time_limit_s * 0.92
    
    num_planes = instance['num_planes']
    num_runways = instance['num_runways']
    
    best_schedule = None
    best_penalty = float('inf')

    def construct_randomized():
        # Sort planes by target time with random jitter to explore different sequences
        indices = list(range(1, num_planes + 1))
        # Jitter magnitude scales with number of planes to maintain diversity
        jitter = 50.0 
        indices.sort(key=lambda i: instance['planes'][i-1]['target'] + random.uniform(-jitter, jitter))
        
        schedule = {}
        runway_last_landing = {r: -float('inf') for r in range(1, num_runways + 1)}
        runway_last_id = {r: None for r in range(1, num_runways + 1)}
        
        for i in indices:
            best_r = -1
            best_t = float('inf')
            min_cost = float('inf')
            
            # Try all runways to find the one that minimizes immediate penalty
            # Using a randomized selection among top candidates can further improve exploration
            candidates = []
            for r in range(1, num_runways + 1):
                p = instance['planes'][i-1]
                prev_id = runway_last_id[r]
                gap = instance['separation'][prev_id-1][i-1] if prev_id else 0
                
                earliest_possible = max(p['earliest'], runway_last_landing[r] + gap)
                if earliest_possible <= p['latest']:
                    # Try to land as close to target as possible
                    t = max(earliest_possible, min(p['target'], p['latest']))
                    cost = (p['target'] - t) * p['penalty_early'] if t < p['target'] else \
                           (t - p['target']) * p['penalty_late']
                    candidates.append((r, t, cost))
            
            if not candidates:
                return None
            
            # Pick the best runway assignment
            candidates.sort(key=lambda x: x[2])
            best_r, best_t, min_cost = candidates[0]
            
            schedule[i] = {"landing_time": best_t, "runway": best_r}
            runway_last_landing[best_r] = best_t
            runway_last_id[best_r] = i
            
        return schedule

    # 1. Initial baseline
    try:
        baseline = tools['greedy_target_time_construct']()
        if tools['validate_schedule'](baseline)[0]:
            best_schedule = baseline
            best_penalty = tools['penalty_of_schedule'](baseline)
    except:
        pass

    # 2. Main optimization loop
    while time.time() < end_time:
        candidate = construct_randomized()
        if candidate and tools['validate_schedule'](candidate)[0]:
            # Apply local search to the candidate
            improved = tools['apply_swap_landings'](candidate, t_limit=0.02)
            if improved and tools['validate_schedule'](improved)[0]:
                candidate = improved
            
            # Attempt random runway reassignments for further improvement
            if random.random() < 0.3:
                p_idx = random.randint(1, num_planes)
                r_idx = random.randint(1, num_runways)
                reassigned = tools['apply_reassign_runway'](candidate, p_idx, r_idx)
                if reassigned and tools['validate_schedule'](reassigned)[0]:
                    candidate = reassigned

            current_penalty = tools['penalty_of_schedule'](candidate)
            if current_penalty < best_penalty:
                best_penalty = current_penalty
                best_schedule = candidate
        
        if best_penalty <= 0:
            break

    return {"schedule": best_schedule if best_schedule else tools['greedy_target_time_construct']()}