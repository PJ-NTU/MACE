# MACE evolved heuristic 10/10 for problem: aircraft_landing
import time
import random

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Optimized solver using a high-frequency multi-start approach combined 
    with targeted local search improvements.
    """
    start_time = time.time()
    # Reserve buffer for final cleanup
    deadline = start_time + time_limit_s * 0.95
    
    num_planes = instance['num_planes']
    num_runways = instance['num_runways']
    
    best_schedule = None
    best_penalty = float('inf')

    # Try deterministic greedy first
    try:
        baseline = tools['greedy_target_time_construct']()
        if baseline and tools['validate_schedule'](baseline)[0]:
            best_schedule = baseline
            best_penalty = tools['penalty_of_schedule'](baseline)
    except:
        pass

    def get_randomized_schedule():
        # Perturbed priority scheduling
        planes_indices = list(range(1, num_planes + 1))
        # Mix target time with jitter for diversity
        jitter = random.uniform(0, 100)
        planes_indices.sort(key=lambda i: instance['planes'][i-1]['target'] + random.uniform(-jitter, jitter))
        
        schedule = {}
        runway_usage = {r: [] for r in range(1, num_runways + 1)}
        
        for i in planes_indices:
            p = instance['planes'][i-1]
            best_r = -1
            best_t = -1
            min_p = float('inf')
            
            # Randomized order of runways to try
            runways = list(range(1, num_runways + 1))
            random.shuffle(runways)
            
            for r in runways:
                # Find valid time slot
                earliest = p['earliest']
                # Check against existing planes on this runway
                for (other_id, t_other) in runway_usage[r]:
                    gap = instance['separation'][other_id-1][i-1]
                    earliest = max(earliest, t_other + gap)
                
                if earliest <= p['latest']:
                    # Target the landing
                    t = max(earliest, min(p['target'], p['latest']))
                    cost = (p['target'] - t) * p['penalty_early'] if t < p['target'] else \
                           (t - p['target']) * p['penalty_late']
                    
                    if cost < min_p:
                        min_p = cost
                        best_r = r
                        best_t = t
            
            if best_r != -1:
                schedule[i] = {"landing_time": best_t, "runway": best_r}
                runway_usage[best_r].append((i, best_t))
            else:
                return None
        return schedule

    # Main optimization loop
    # We prioritize speed and quantity of samples, refinement via tools
    while time.time() < deadline:
        cand = get_randomized_schedule()
        if cand:
            # Quick validation
            if tools['validate_schedule'](cand)[0]:
                # Attempt to refine
                refined = tools['apply_swap_landings'](cand, t_limit=0.01)
                if refined and tools['validate_schedule'](refined)[0]:
                    cand = refined
                
                curr_p = tools['penalty_of_schedule'](cand)
                if curr_p < best_penalty:
                    best_penalty = curr_p
                    best_schedule = cand
        
        # Early exit if we found a perfect solution
        if best_penalty <= 0:
            break

    # Fallback to tools if needed
    if best_schedule is None:
        return {"schedule": tools['greedy_target_time_construct']()}
        
    return {"schedule": best_schedule}