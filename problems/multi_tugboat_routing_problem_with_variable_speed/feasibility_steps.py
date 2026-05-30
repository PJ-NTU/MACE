"""Step-by-step is_feasible reference for the Multi-Tugboat Routing
Problem with Variable Speed (MTRSP-VS).

Read by spec.py and prepended to feasibility_doc. Gives the LLM a compact,
clearly-labeled view of every constraint (without cost computation), in
the TSP-style early-return pattern with each check tagged (Cn).

All `is_feasible` references below assume the instance kwargs are in
scope (num_tasks, num_tugboats, task_max_tugs, ... — see the `solve()`
docstring for the full list).
"""

FEASIBILITY_STEPS_PY = r'''def is_feasible(solution):
    # ────── Top-level shape ──────────────────────────────────────────────
    if not isinstance(solution, dict):
        return False, f"solution must be dict, got {type(solution).__name__}"
    for key in ("routes", "service_speeds", "start_times", "transit_speeds"):
        if key not in solution:
            return False, f"solution missing '{key}' key"

    routes         = solution["routes"]
    service_speeds = solution["service_speeds"]
    start_times    = solution["start_times"]
    transit_speeds = solution["transit_speeds"]
    n = num_tasks
    m = num_tugboats
    L = num_speed_levels
    T_max = planning_horizon
    EPS = 1e-6

    if not isinstance(routes, dict):
        return False, f"routes must be dict, got {type(routes).__name__}"
    if not isinstance(transit_speeds, dict):
        return False, f"transit_speeds must be dict, got {type(transit_speeds).__name__}"
    if not isinstance(service_speeds, dict):
        return False, f"service_speeds must be dict, got {type(service_speeds).__name__}"
    if not isinstance(start_times, dict):
        return False, f"start_times must be dict, got {type(start_times).__name__}"

    tug_keys = set(range(m))
    if set(routes.keys()) != tug_keys:
        return False, f"routes must have keys exactly {{0..{m-1}}}, got {sorted(routes.keys())}"
    if set(transit_speeds.keys()) != tug_keys:
        return False, f"transit_speeds must have keys exactly {{0..{m-1}}}, got {sorted(transit_speeds.keys())}"

    # ────── Per-tug shape + (C7) transit speed count and levels ──────────
    for k in range(m):
        route = routes[k]
        speeds_k = transit_speeds[k]
        if not isinstance(route, list):
            return False, f"routes[{k}] must be list, got {type(route).__name__}"
        if not isinstance(speeds_k, list):
            return False, f"transit_speeds[{k}] must be list, got {type(speeds_k).__name__}"
        if len(set(route)) != len(route):
            return False, f"tug {k} route has duplicate tasks: {route}"
        for s in route:
            if not isinstance(s, int) or not (0 <= s < n):
                return False, f"tug {k} route contains invalid task id {s!r} (must be int in [0, {n}))"
        # (C7) one speed per arc. Empty route ⇒ 0 arcs; else len(route)+1 arcs.
        expected_arcs = (len(route) + 1) if route else 0
        if len(speeds_k) != expected_arcs:
            return False, (f"tug {k} transit_speeds length {len(speeds_k)} != "
                           f"expected {expected_arcs} (len(routes[{k}])+1 if route else 0) (C7)")
        for arc_idx, lvl in enumerate(speeds_k):
            if not isinstance(lvl, int) or not (0 <= lvl < L):
                return False, (f"tug {k} arc {arc_idx} transit speed level {lvl!r} not "
                               f"in {{0..{L-1}}} (C7)")

    # ────── Derive executed_tasks + task_tugboats from routes ────────────
    executed = set()
    task_tugs = {}
    for k in range(m):
        for s in routes[k]:
            executed.add(s)
            task_tugs.setdefault(s, []).append(k)

    if set(service_speeds.keys()) != executed:
        return False, (f"service_speeds keys must equal executed tasks "
                       f"{sorted(executed)}, got {sorted(service_speeds.keys())} (C8)")
    if set(start_times.keys()) != executed:
        return False, (f"start_times keys must equal executed tasks "
                       f"{sorted(executed)}, got {sorted(start_times.keys())} (C9)")

    # ────── Per-task checks (C1/C2/C8/C9/C10) ────────────────────────────
    svc_time = {}
    for s in executed:
        lvl_s = service_speeds[s]
        if not isinstance(lvl_s, int) or not (0 <= lvl_s < L):
            return False, f"service_speeds[{s}] = {lvl_s!r} not in {{0..{L-1}}} (C8)"
        # (C1) tug count limit
        if len(task_tugs[s]) > task_max_tugs[s]:
            return False, (f"task {s} served by {len(task_tugs[s])} tugs > "
                           f"task_max_tugs[{s}]={task_max_tugs[s]} (C1)")
        # (C2) HP sum
        hp_sum = sum(tugboat_horsepower[k] for k in task_tugs[s])
        if hp_sum < task_min_horsepower[s] - EPS:
            return False, (f"task {s} HP sum {hp_sum} < required "
                           f"{task_min_horsepower[s]} (C2)")
        # (C9) time window
        tau = start_times[s]
        if not isinstance(tau, (int, float)):
            return False, f"start_times[{s}] = {tau!r} not numeric (C9)"
        if tau < task_time_window_lower[s] - EPS or tau > task_time_window_upper[s] + EPS:
            return False, (f"task {s} start_time {tau} outside window "
                           f"[{task_time_window_lower[s]}, {task_time_window_upper[s]}] (C9)")
        # (C10) finish ≤ T_max
        d_s = task_service_distance[s]
        v_ls = speed_values[lvl_s]
        T_s = d_s / v_ls
        svc_time[s] = T_s
        if tau + T_s > T_max + EPS:
            return False, (f"task {s} finish time {tau + T_s} > planning_horizon "
                           f"{T_max} (C10)")

    # ────── Time propagation (C11, C12) ──────────────────────────────────
    for k in range(m):
        route = routes[k]
        if not route:
            continue
        speeds_k = transit_speeds[k]
        # (C11) depot → first task
        s0 = route[0]
        v0 = speed_values[speeds_k[0]]
        arrival = depot_to_task_distance[s0] / v0
        if arrival > start_times[s0] + EPS:
            return False, (f"tug {k} arrives at task {s0} at {arrival} > "
                           f"start_time {start_times[s0]} (C11)")
        # (C12) consecutive tasks
        for i in range(1, len(route)):
            si_prev = route[i - 1]
            si = route[i]
            end_prev = start_times[si_prev] + svc_time[si_prev]
            d_ij = task_to_task_distance[si_prev][si]
            v_ij = speed_values[speeds_k[i]]
            arrival = end_prev + d_ij / v_ij
            if arrival > start_times[si] + EPS:
                return False, (f"tug {k} arc {si_prev}->{si}: arrival {arrival} > "
                               f"start_time {start_times[si]} (C12)")

    # ────── (C13) per-tug fuel capacity ──────────────────────────────────
    v_medium = speed_values[1]
    for k in range(m):
        route = routes[k]
        if not route:
            continue
        speeds_k = transit_speeds[k]
        HP_k = tugboat_horsepower[k]
        alpha_k = tugboat_alpha[k]
        beta_k = tugboat_beta[k]
        total = 0.0
        # Service fuel
        for s in route:
            v_ls = speed_values[service_speeds[s]]
            total += alpha_k * HP_k * task_service_distance[s] * (v_ls ** 2) / (v_medium ** 3)
        # Transit fuel: depot → s0, s_i → s_{i+1}, s_last → depot
        s0 = route[0]
        v0 = speed_values[speeds_k[0]]
        total += beta_k * HP_k * depot_to_task_distance[s0] * (v0 ** 2) / (v_medium ** 3)
        for i in range(1, len(route)):
            v_i = speed_values[speeds_k[i]]
            total += beta_k * HP_k * task_to_task_distance[route[i-1]][route[i]] * (v_i ** 2) / (v_medium ** 3)
        s_last = route[-1]
        v_ret = speed_values[speeds_k[-1]]
        total += beta_k * HP_k * task_to_depot_distance[s_last] * (v_ret ** 2) / (v_medium ** 3)
        if total > tugboat_fuel_capacity[k] + EPS:
            return False, (f"tug {k} total fuel {total} > capacity "
                           f"{tugboat_fuel_capacity[k]} (C13)")

    return True, None
'''
