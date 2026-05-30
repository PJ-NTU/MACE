# MACE evolved heuristic 07/10 for problem: aircraft_landing
import time
import random
import numpy as np

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    Dispatcher-style solver for Aircraft Landing.
    
    Heuristic Selection Logic:
    - If the problem is 'dense' (high ratio of planes to runways, or tight separation requirements),
      we prioritize the more deterministic/greedy approach (Parent A) to avoid exploring too many
      infeasible states.
    - If the problem is 'sparse' (more runways, looser windows), we prioritize exploration of 
      different landing sequences (Parent B) to find global optima.
    """
    start_time = time.time()
    end_time = start_time + time_limit_s * 0.92
    
    num_planes = instance['num_planes']
    num_runways = instance['num_runways']
    
    # Heuristic feature extraction
    # Density: Average planes per runway
    density = num_planes / max(1, num_runways)
    # Window slack: average (latest - earliest) / target
    windows = instance['planes']
    avg_slack = np.mean([(w['latest'] - w['earliest']) / max(1, w['target']) for w in windows])
    
    # Decision: Use Parent B's randomized exploration for sparse/slack scenarios, 
    # and Parent A's more focused approach for high-density/constrained scenarios.
    use_randomized_choice = (density < 5.0) or (avg_slack > 0.5)

    def construct(randomized_choice=False):
        indices = list(range(1, num_planes + 1))
        # Jitter scales with problem size
        jitter = 50.0 
        indices.sort(key=lambda i: instance['planes'][i-1]['target'] + random.uniform(-jitter, jitter))
        
        schedule = {}
        runway_last_landing = {r: -float('inf') for r in range(1, num_runways + 1)}
        runway_last_id = {r: None for r in range(1, num_runways + 1)}
        
        for i in indices:
            candidates = []
            for r in range(1, num_runways + 1):
                p = instance['planes'][i-1]
                prev_id = runway_last_id[r]
                gap = instance['separation'][prev_id-1][i-1] if prev_id else 0
                earliest_possible = max(p['earliest'], runway_last_landing[r] + gap)
                if earliest_possible <= p['latest']:
                    t = max(earliest_possible, min(p['target'], p['latest']))
                    cost = (p['target'] - t) * p['penalty_early'] if t < p['target'] else \
                           (t - p['target']) * p['penalty_late']
                    candidates.append((r, t, cost))
            
            if not candidates: return None
            
            candidates.sort(key=lambda x: x[2])
            # Dispatch strategy logic
            if randomized_choice and len(candidates) > 1 and random.random() < 0.7:
                best_r, best_t, _ = candidates[1]
            else:
                best_r, best_t, _ = candidates[0]
            
            schedule[i] = {"landing_time": best_t, "runway": best_r}
            runway_last_landing[best_r] = best_t
            runway_last_id[best_r] = i
        return schedule

    best_schedule = None
    best_penalty = float('inf')

    # Baseline
    try:
        baseline = tools['greedy_target_time_construct']()
        if tools['validate_schedule'](baseline)[0]:
            best_schedule = baseline
            best_penalty = tools['penalty_of_schedule'](baseline)
    except: pass

    # Optimization Loop
    while time.time() < end_time:
        candidate = construct(randomized_choice=use_randomized_choice)
        if candidate and tools['validate_schedule'](candidate)[0]:
            # Refinement
            improved = tools['apply_swap_landings'](candidate, t_limit=0.01)
            if improved and tools['validate_schedule'](improved)[0]: candidate = improved
            
            if random.random() < 0.2:
                p_idx = random.randint(1, num_planes)
                r_idx = random.randint(1, num_runways)
                reassigned = tools['apply_reassign_runway'](candidate, p_idx, r_idx)
                if reassigned and tools['validate_schedule'](reassigned)[0]: candidate = reassigned

            current_penalty = tools['penalty_of_schedule'](candidate)
            if current_penalty < best_penalty:
                best_penalty = current_penalty
                best_schedule = candidate
        
        if best_penalty <= 0: break

    return {"schedule": best_schedule if best_schedule else tools['greedy_target_time_construct']()}