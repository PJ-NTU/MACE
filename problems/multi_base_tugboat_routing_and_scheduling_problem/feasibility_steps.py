"""Step-by-step is_feasible reference for MTRSP-MB.

Read by spec.py and prepended to feasibility_doc. Gives the LLM a compact,
clearly-labeled view of every constraint (without the cost computation), in
the TSP-style early-return pattern. Empirically (see OVERNIGHT_LOG.md) this
raises first-shot LLM bootstrap success notably over relying on eval_func
source alone — the constraint codes (C1..C14) make it scannable.

All `is_feasible` references below assume the instance kwargs are in scope
(num_tasks, tugboat_horsepower, time_matrix, ... — see solve() docstring).
"""

FEASIBILITY_STEPS_PY = r'''def is_feasible(solution):
    # ────── Top-level shape ──────────────────────────────────────────────
    if not isinstance(solution, dict):
        return False, f"solution must be dict, got {type(solution).__name__}"
    for key in ("routes", "task_tugboats", "task_start_times"):
        if key not in solution:
            return False, f"solution missing '{key}' key"

    routes = solution["routes"]
    task_tugboats = solution["task_tugboats"]
    task_start_times = solution["task_start_times"]
    n = num_tasks
    m = num_tugboats
    p = num_bases
    T_max = planning_horizon
    EPS = 1e-7

    if not isinstance(routes, list) or len(routes) != m:
        return False, (f"routes must be a list of length m={m}, got "
                       f"type={type(routes).__name__}")
    if not isinstance(task_tugboats, dict):
        return False, f"task_tugboats must be a dict, got {type(task_tugboats).__name__}"
    if not isinstance(task_start_times, dict):
        return False, f"task_start_times must be a dict, got {type(task_start_times).__name__}"

    # ────── Validate routes element shape + uniqueness per route ─────────
    for k in range(m):
        route = routes[k]
        if not isinstance(route, list):
            return False, f"routes[{k}] must be a list, got {type(route).__name__}"
        seen = set()
        for task_id in route:
            if not isinstance(task_id, int):
                return False, f"routes[{k}] contains non-int task_id {task_id!r}"
            if not (1 <= task_id <= n):
                return False, f"routes[{k}] task_id={task_id} not in [1, {n}]"
            if task_id in seen:
                return False, f"routes[{k}] visits task {task_id} twice"
            seen.add(task_id)

    # ────── Derive executed set + check linkage (C6) ─────────────────────
    executed = set()
    derived_tugs = {}
    for k in range(m):
        for task_id in routes[k]:
            executed.add(task_id)
            derived_tugs.setdefault(task_id, set()).add(k)

    if set(task_tugboats.keys()) != executed:
        return False, (f"task_tugboats keys {sorted(task_tugboats.keys())} must "
                       f"equal tasks in routes {sorted(executed)} (C6)")
    if set(task_start_times.keys()) != executed:
        return False, (f"task_start_times keys {sorted(task_start_times.keys())} "
                       f"must equal tasks in routes {sorted(executed)}")

    for s in executed:
        tugs = task_tugboats[s]
        if not isinstance(tugs, list):
            return False, f"task_tugboats[{s}] must be a list, got {type(tugs).__name__}"
        if len(set(tugs)) != len(tugs):
            return False, f"task_tugboats[{s}]={tugs} has duplicates"
        for k_ in tugs:
            if not isinstance(k_, int) or not (0 <= k_ < m):
                return False, f"task_tugboats[{s}] tug_id={k_!r} not in [0, {m})"
        if set(tugs) != derived_tugs[s]:
            return False, (f"task_tugboats[{s}]={sorted(set(tugs))} disagrees with "
                           f"routes-derived tugs {sorted(derived_tugs[s])} (C6)")
        ts = task_start_times[s]
        if not isinstance(ts, (int, float)) or isinstance(ts, bool):
            return False, f"task_start_times[{s}]={ts!r} must be numeric"
        if ts < 0:
            return False, f"task_start_times[{s}]={ts} < 0"

    # ────── Per-task constraints (C1, C2, C10, C11) ──────────────────────
    for s in executed:
        tugs = task_tugboats[s]
        # (C1) tug count
        if len(tugs) > task_max_tugs[s - 1]:
            return False, (f"task {s} has {len(tugs)} tugs > Γ_s="
                           f"{task_max_tugs[s - 1]} (C1)")
        if len(tugs) < 1:
            return False, f"task {s} has 0 serving tugs"
        # (C2) HP requirement
        total_hp = sum(tugboat_horsepower[k_] for k_ in tugs)
        if total_hp < task_min_horsepower[s - 1]:
            return False, (f"task {s} total HP={total_hp} < H_s^min="
                           f"{task_min_horsepower[s - 1]} (C2)")
        # (C10) time window
        ts = task_start_times[s]
        if ts < task_time_window_lower[s - 1] - EPS or ts > task_time_window_upper[s - 1] + EPS:
            return False, (f"task {s} τ={ts} outside "
                           f"[{task_time_window_lower[s - 1]}, "
                           f"{task_time_window_upper[s - 1]}] (C10)")
        # (C11) finish before T_max
        if ts + task_service_time[s - 1] > T_max + EPS:
            return False, (f"task {s} τ+T_s={ts + task_service_time[s - 1]} > "
                           f"T_max={T_max} (C11)")

    # ────── Base capacity (C7, C8) ───────────────────────────────────────
    depart_count = [0] * p
    for k in range(m):
        if len(routes[k]) == 0:
            continue
        b = tugboat_base_assignment[k]
        if not isinstance(b, int) or not (-p <= b <= -1):
            return False, f"tugboat_base_assignment[{k}]={b} not in {{-1..-{p}}}"
        depart_count[-b - 1] += 1
    for b_idx in range(p):
        if depart_count[b_idx] > base_capacity[b_idx]:
            return False, (f"base {-(b_idx + 1)}: {depart_count[b_idx]} departures > "
                           f"δ_b={base_capacity[b_idx]} (C7)")
        if depart_count[b_idx] > base_capacity[b_idx]:
            return False, (f"base {-(b_idx + 1)}: {depart_count[b_idx]} arrivals > "
                           f"δ_b={base_capacity[b_idx]} (C8)")

    # ────── Per-tug time + fuel (C12, C13, C14) ──────────────────────────
    for k in range(m):
        route = routes[k]
        if not route:
            continue
        home = tugboat_base_assignment[k]
        dest = n - home
        HP_k = tugboat_horsepower[k]
        a_k = tugboat_alpha[k]
        b_k = tugboat_beta[k]

        # (C12) base origin → first task
        first = route[0]
        key_out = f"{home}_{first}"
        if key_out not in time_matrix:
            return False, f"time_matrix missing {key_out!r} (tug {k} depart)"
        t_out = time_matrix[key_out]
        if t_out > task_start_times[first] + EPS:
            return False, (f"tug {k}: arrival at first task {first} = {t_out} > "
                           f"τ={task_start_times[first]} (C12)")
        tug_fuel = b_k * HP_k * t_out + a_k * HP_k * task_service_time[first - 1]

        # (C13) task → task
        for u in range(len(route) - 1):
            i_task = route[u]
            j_task = route[u + 1]
            key_mid = f"{i_task}_{j_task}"
            if key_mid not in time_matrix:
                return False, f"time_matrix missing {key_mid!r} (tug {k} mid)"
            t_mid = time_matrix[key_mid]
            arr_j = task_start_times[i_task] + task_service_time[i_task - 1] + t_mid
            if arr_j > task_start_times[j_task] + EPS:
                return False, (f"tug {k}: arrival at task {j_task}={arr_j} > "
                               f"τ={task_start_times[j_task]} (C13)")
            tug_fuel += b_k * HP_k * t_mid + a_k * HP_k * task_service_time[j_task - 1]

        # last task → base destination, must finish before T_max
        last = route[-1]
        key_in = f"{last}_{dest}"
        if key_in not in time_matrix:
            return False, f"time_matrix missing {key_in!r} (tug {k} return)"
        t_in = time_matrix[key_in]
        return_finish = (task_start_times[last] + task_service_time[last - 1] + t_in)
        if return_finish > T_max + EPS:
            return False, (f"tug {k}: return_finish={return_finish} > T_max={T_max} "
                           f"(C11 along route)")
        tug_fuel += b_k * HP_k * t_in

        # (C14) fuel capacity
        if tug_fuel > tugboat_fuel_capacity[k] + EPS:
            return False, (f"tug {k}: fuel={tug_fuel} > capacity="
                           f"{tugboat_fuel_capacity[k]} (C14)")

    return True, None
'''
