"""Step-by-step is_feasible reference for Multi-Tugboat Routing and Scheduling Problem.

Read by spec.py and prepended to feasibility_doc. Gives the LLM a compact,
clearly-labeled view of every constraint (without the cost computation), in
the TSP-style early-return pattern. Empirically (see OVERNIGHT_LOG.md) this
raises first-shot LLM bootstrap success notably over relying on eval_func
source alone — the constraint codes (C1..C11) make it scannable.

All `is_feasible` references below assume the instance kwargs are in scope
(num_tasks, num_tugboats, task_max_tugs, ... — see solve() docstring).
"""

FEASIBILITY_STEPS_PY = r'''def is_feasible(solution):
    # ────── Top-level shape ─────────────────────────────────────────────
    if not isinstance(solution, dict):
        return False, f"solution must be dict, got {type(solution).__name__}"
    for key in ("routes", "task_tugboats", "task_start_times"):
        if key not in solution:
            return False, f"solution missing '{key}' key"

    routes           = solution["routes"]
    task_tugboats    = solution["task_tugboats"]
    task_start_times = solution["task_start_times"]
    n = num_tasks
    K = num_tugboats
    T_max = planning_horizon
    TOL = 1e-6
    expected_task_keys = set(range(1, n + 1))

    if not isinstance(routes, list) or len(routes) != K:
        return False, f"routes must be list of length {K}, got len={len(routes) if hasattr(routes, '__len__') else 'n/a'}"
    if not isinstance(task_tugboats, dict) or set(task_tugboats.keys()) != expected_task_keys:
        return False, f"task_tugboats must be dict with keys exactly {{1..{n}}}"
    if not isinstance(task_start_times, dict) or set(task_start_times.keys()) != expected_task_keys:
        return False, f"task_start_times must be dict with keys exactly {{1..{n}}}"

    # ────── Validate routes ────────────────────────────────────────────
    for k, route in enumerate(routes):
        if not isinstance(route, list):
            return False, f"routes[{k}] must be list, got {type(route).__name__}"
        if len(route) != len(set(route)):
            return False, f"routes[{k}]={route} has duplicate task id (C5)"
        for s in route:
            if not isinstance(s, int) or not (1 <= s <= n):
                return False, f"routes[{k}] entry {s!r} not int in [1, {n}]"

    # ────── Validate task_tugboats + C6 consistency ────────────────────
    for s in range(1, n + 1):
        tugs = task_tugboats[s]
        if not isinstance(tugs, list):
            return False, f"task_tugboats[{s}] must be list, got {type(tugs).__name__}"
        if len(tugs) != len(set(tugs)):
            return False, f"task_tugboats[{s}]={tugs} has duplicate tug id"
        for k in tugs:
            if not isinstance(k, int) or not (0 <= k < K):
                return False, f"task_tugboats[{s}] entry {k!r} not in [0, {K})"
            # (C6) k must list s in its route
            if s not in routes[k]:
                return False, (f"inconsistency: tug {k} in task_tugboats[{s}] "
                               f"but task {s} not in routes[{k}] (C6)")
    # (C6) mirror direction
    for k, route in enumerate(routes):
        for s in route:
            if k not in task_tugboats[s]:
                return False, (f"inconsistency: task {s} in routes[{k}] "
                               f"but tug {k} not in task_tugboats[{s}] (C6)")

    # ────── Per-task checks (executed only) ─────────────────────────────
    for s in range(1, n + 1):
        tugs = task_tugboats[s]
        if not tugs:
            continue  # unexecuted (z_s = 0) — start_time ignored
        # (C1) ≤ Γ_s tugs
        if len(tugs) > task_max_tugs[s - 1]:
            return False, (f"task {s} has {len(tugs)} tugs > "
                           f"task_max_tugs[{s-1}]={task_max_tugs[s-1]} (C1)")
        # (C2) HP sum ≥ H_s^min
        total_hp = sum(tugboat_horsepower[k] for k in tugs)
        if total_hp < task_min_horsepower[s - 1] - TOL:
            return False, (f"task {s} total HP={total_hp} < min "
                           f"{task_min_horsepower[s-1]} (C2)")
        # (C7) start in window
        try:
            tau = float(task_start_times[s])
        except Exception:
            return False, f"task_start_times[{s}]={task_start_times[s]!r} not a number"
        a_s = task_time_window_lower[s - 1]
        b_s = task_time_window_upper[s - 1]
        if tau < a_s - TOL:
            return False, f"task {s} start={tau} < window lower {a_s} (C7)"
        if tau > b_s + TOL:
            return False, f"task {s} start={tau} > window upper {b_s} (C7)"
        # (C8) finish ≤ T_max
        T_s = task_service_time[s - 1]
        if tau + T_s > T_max + TOL:
            return False, f"task {s} finish={tau + T_s} > T_max={T_max} (C8)"

    # ────── Per-tug: time propagation (C9/C10) + fuel (C11) ─────────────
    for k, route in enumerate(routes):
        if not route:
            continue
        hp = tugboat_horsepower[k]
        al = tugboat_alpha[k]
        be = tugboat_beta[k]
        f_max = tugboat_fuel_capacity[k]

        current_time = 0.0
        current_node = 0
        total_fuel = 0.0

        for idx, s in enumerate(route):
            T_s = task_service_time[s - 1]
            key = f"{current_node}_{s}"
            if key not in time_matrix:
                return False, f"time_matrix missing {key!r} (tug {k} step {idx})"
            t_travel = time_matrix[key]
            arrival = current_time + t_travel
            tau = float(task_start_times[s])
            if arrival > tau + TOL:
                code = "C9" if idx == 0 else "C10"
                return False, (f"tug {k} cannot reach task {s}: arrives "
                               f"{arrival:.4f}, start {tau:.4f} ({code})")
            total_fuel += be * hp * t_travel
            total_fuel += al * hp * T_s
            current_time = tau + T_s
            current_node = s

        ret_key = f"{current_node}_{n + 1}"
        if ret_key not in time_matrix:
            return False, f"time_matrix missing {ret_key!r} (tug {k} return)"
        total_fuel += be * hp * time_matrix[ret_key]
        if total_fuel > f_max + TOL:
            return False, f"tug {k} fuel {total_fuel:.4f} > capacity {f_max} (C11)"

    return True, None
'''
