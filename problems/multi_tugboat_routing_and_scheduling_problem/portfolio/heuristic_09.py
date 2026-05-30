# MACE evolved heuristic 09/10 for problem: multi_tugboat_routing_and_scheduling_problem
import time
import random
import math

def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
    """
    A dispatcher-style MTRSP solver.
    
    Hypothesis:
    - Instances with high task density (n/K ratio) and tight time windows benefit
      from Parent A (Regret-based + ILS), which explores the search space more
      aggressively to satisfy tight constraints.
    - Instances with lower density or more flexible windows benefit from Parent B
      (Construct-Destroy-Repair), which is more stable and better at optimizing 
      fuel costs once feasibility is reached.
    """
    start_time = time.time()
    n = instance['num_tasks']
    K = instance['num_tugboats']

    # Dispatcher Logic
    # Calculate average time window slack as a proxy for "tightness"
    total_slack = sum(instance['task_time_window_upper'][i] - instance['task_time_window_lower'][i] 
                      for i in range(n))
    avg_slack = total_slack / n if n > 0 else 0
    density = n / (K + 1)
    
    # Heuristic: If tasks are tightly constrained (low slack) and dense, use A.
    # Otherwise, use the more stable B approach.
    use_strategy_a = density > 2.0 or avg_slack < (instance['planning_horizon'] * 0.2)

    def get_empty_sol():
        return {
            'routes': [[] for _ in range(K)],
            'task_tugboats': {s: [] for s in range(1, n + 1)},
            'task_start_times': {s: 0.0 for s in range(1, n + 1)}
        }

    # Common construction logic
    def build_sol(task_order):
        sol = get_empty_sol()
        for s in task_order:
            assignment = tools['find_feasible_assignment'](s, sol)
            if assignment:
                sol = tools['apply_task_assignment'](sol, s, assignment['tug_ids'], assignment['start_time'])
        return sol

    # Strategy Selectors
    if use_strategy_a:
        # A-style: Focus on difficult tasks first (bottleneck heuristic)
        def get_difficulty(s):
            idx = s - 1
            cap = len(tools['tugs_with_enough_hp_alone'](s))
            return (1.0 / (cap + 1.0)) * instance['task_time_window_upper'][idx]
        task_order = sorted(range(1, n + 1), key=get_difficulty)
    else:
        # B-style: Focus on window size and HP requirements
        task_scores = []
        for s in range(1, n + 1):
            w = instance['task_time_window_upper'][s-1] - instance['task_time_window_lower'][s-1]
            hp = instance['task_min_horsepower'][s-1]
            task_scores.append((w, -hp, s))
        task_order = [t[2] for t in sorted(task_scores)]

    best_sol = build_sol(task_order)
    best_obj = tools['objective'](best_sol)

    # Main meta-heuristic loop
    while time.time() - start_time < time_limit_s * 0.95:
        executed = [s for s in range(1, n + 1) if best_sol['task_tugboats'][s]]
        if not executed:
            best_sol = build_sol(random.sample(range(1, n + 1), n))
            best_obj = tools['objective'](best_sol)
            continue
            
        # Adaptive removal
        removal_rate = 0.2 if use_strategy_a else 0.15
        num_remove = max(1, int(len(executed) * removal_rate))
        to_remove = random.sample(executed, num_remove)
        
        work_sol = {
            'routes': [list(r) for r in best_sol['routes']],
            'task_tugboats': {s: list(ts) for s, ts in best_sol['task_tugboats'].items()},
            'task_start_times': {s: ts for s, ts in best_sol['task_start_times'].items()}
        }
        for s in to_remove:
            work_sol['task_tugboats'][s] = []
            for k in range(K):
                if s in work_sol['routes'][k]:
                    work_sol['routes'][k].remove(s)
        
        # Repair phase
        unassigned = [s for s in range(1, n + 1) if not work_sol['task_tugboats'][s]]
        random.shuffle(unassigned)
        for s in unassigned:
            assignment = tools['find_feasible_assignment'](s, work_sol)
            if assignment:
                work_sol = tools['apply_task_assignment'](work_sol, s, assignment['tug_ids'], assignment['start_time'])
        
        try:
            curr_obj = tools['objective'](work_sol)
            if curr_obj < best_obj:
                best_obj = curr_obj
                best_sol = work_sol
        except Exception:
            continue

    return best_sol