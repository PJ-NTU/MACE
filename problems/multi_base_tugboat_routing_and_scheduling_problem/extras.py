"""Per-problem extras for MTRSP-MB (Multi-Base Tugboat Routing and Scheduling).

Provides helpers so the LLM can compose construction heuristics without
reinventing route construction, time-feasibility checks, or fuel accounting.

Tool groups:
  (1) Queries:                 task_window, task_required_hp, tugs_at_base,
                               unassigned_tasks, route_load_status
  (2) Feasibility primitives:  earliest_arrival_after, has_route_capacity,
                               route_after_insertion_feasible
  (3) Construction:            find_feasible_insertion,
                               find_collaborative_tug_set,
                               append_task_to_route

All functions are PURE — no hidden state. Pass `solution` explicitly. They
respect every constraint that `eval_func` checks (C1..C14 — see config.py).

Conventions:
  - solution is the 3-key dict {routes, task_tugboats, task_start_times}.
  - task ids are 1..n; tugboat ids are 0..m-1; bases are negative ints -1..-p.
  - time_matrix keys are strings like '<i>_<j>' (see config.py).

These tools are optional — the LLM may use any subset, or write everything
from scratch.
"""
from __future__ import annotations

from itertools import combinations


def extra_tools(instance: dict) -> dict:
    """Factory: returns MTRSP-MB-specific tool callables bound to one instance."""
    n = instance["num_tasks"]
    m = instance["num_tugboats"]
    p = instance["num_bases"]
    a_lo = instance["task_time_window_lower"]
    b_hi = instance["task_time_window_upper"]
    T_s_arr = instance["task_service_time"]
    gamma_arr = instance["task_max_tugs"]
    hp_req_arr = instance["task_min_horsepower"]
    hp_arr = instance["tugboat_horsepower"]
    fuel_cap = instance["tugboat_fuel_capacity"]
    alpha = instance["tugboat_alpha"]
    beta = instance["tugboat_beta"]
    base_of_tug = instance["tugboat_base_assignment"]
    base_cap = instance["base_capacity"]
    tm = instance["time_matrix"]
    T_max = instance["planning_horizon"]
    EPS = 1e-7

    # Precompute: tugs grouped by home base
    _tugs_at_base: dict[int, list[int]] = {-(b + 1): [] for b in range(p)}
    for k in range(m):
        _tugs_at_base[base_of_tug[k]].append(k)

    def _route_fuel_and_time(tug_k: int, route: list[int],
                             start_times: dict[int, float]) -> tuple[float, float]:
        """Return (total_fuel, return_to_base_time) if route is time-feasible
        for tug_k under start_times. Raise ValueError if infeasible."""
        if not route:
            return 0.0, 0.0
        hp = hp_arr[tug_k]
        a_k = alpha[tug_k]
        b_k = beta[tug_k]
        home = base_of_tug[tug_k]
        first = route[0]
        t_out = tm[f"{home}_{first}"]
        if t_out > start_times[first] + EPS:
            raise ValueError(f"tug {tug_k} can't reach task {first}")
        fuel = b_k * hp * t_out + a_k * hp * T_s_arr[first - 1]
        for u in range(len(route) - 1):
            i_task = route[u]
            j_task = route[u + 1]
            t_mid = tm[f"{i_task}_{j_task}"]
            arr = start_times[i_task] + T_s_arr[i_task - 1] + t_mid
            if arr > start_times[j_task] + EPS:
                raise ValueError(f"tug {tug_k}: arrival at {j_task} after τ")
            fuel += b_k * hp * t_mid + a_k * hp * T_s_arr[j_task - 1]
        last = route[-1]
        dest = n - home
        t_in = tm[f"{last}_{dest}"]
        return_finish = start_times[last] + T_s_arr[last - 1] + t_in
        fuel += b_k * hp * t_in
        return fuel, return_finish

    # ==================================================================
    # (1) Queries
    # ==================================================================
    def task_window(task_id: int) -> tuple[float, float]:
        """Returns (a_s, b_s) — earliest and latest allowed start (C10)."""
        if not (1 <= task_id <= n):
            raise ValueError(f"task_id={task_id} not in [1, {n}]")
        return a_lo[task_id - 1], b_hi[task_id - 1]

    def task_required_hp(task_id: int) -> tuple[float, int]:
        """Returns (H_s^min, Γ_s) for task_id."""
        if not (1 <= task_id <= n):
            raise ValueError(f"task_id={task_id} not in [1, {n}]")
        return hp_req_arr[task_id - 1], gamma_arr[task_id - 1]

    def tugs_at_base(base_id: int) -> list[int]:
        """Returns list of tug ids whose home base is base_id (negative int)."""
        if base_id not in _tugs_at_base:
            raise ValueError(f"base_id={base_id} not in {sorted(_tugs_at_base.keys())}")
        return list(_tugs_at_base[base_id])

    def unassigned_tasks(solution: dict) -> list[int]:
        """Returns sorted list of task ids NOT yet in any route of solution."""
        served = set(solution["task_tugboats"].keys())
        return [s for s in range(1, n + 1) if s not in served]

    def route_load_status(solution: dict) -> list[dict]:
        """For each tug k, returns a dict with route length, total fuel used,
        return-to-base time. Useful for picking which tug to extend next.
        """
        out = []
        for k in range(m):
            route = solution["routes"][k]
            if not route:
                out.append({"tug_id": k, "num_tasks": 0, "fuel_used": 0.0,
                            "return_time": 0.0,
                            "fuel_remaining": float(fuel_cap[k])})
                continue
            try:
                fuel, ret = _route_fuel_and_time(k, route, solution["task_start_times"])
                out.append({"tug_id": k, "num_tasks": len(route),
                            "fuel_used": fuel, "return_time": ret,
                            "fuel_remaining": fuel_cap[k] - fuel})
            except ValueError as e:
                out.append({"tug_id": k, "num_tasks": len(route),
                            "fuel_used": float("inf"), "return_time": float("inf"),
                            "fuel_remaining": float("-inf"),
                            "error": str(e)})
        return out

    # ==================================================================
    # (2) Feasibility primitives
    # ==================================================================
    def earliest_arrival_after(tug_k: int, route: list[int],
                               start_times: dict[int, float],
                               new_task: int) -> float | None:
        """Returns the earliest possible arrival of tug_k at new_task
        if appended to the end of `route` (with the given start_times).
        Returns None if route prefix itself is infeasible.
        """
        if route:
            try:
                _, ret = _route_fuel_and_time(tug_k, route, start_times)
            except ValueError:
                return None
            last = route[-1]
            t_seg = tm[f"{last}_{new_task}"]
            return start_times[last] + T_s_arr[last - 1] + t_seg
        # Empty route: tug starts from home base at t=0
        home = base_of_tug[tug_k]
        return tm[f"{home}_{new_task}"]

    def has_route_capacity(tug_k: int, route_after: list[int],
                           start_times_after: dict[int, float]) -> bool:
        """Check whether tug_k can perform `route_after` under given start
        times: (a) every arrival ≤ τ, (b) return-to-base ≤ T_max, (c) fuel ≤ cap.
        Returns True if all OK.
        """
        if not route_after:
            return True
        try:
            fuel, ret = _route_fuel_and_time(tug_k, route_after, start_times_after)
        except ValueError:
            return False
        if ret > T_max + EPS:
            return False
        if fuel > fuel_cap[tug_k] + EPS:
            return False
        return True

    def route_after_insertion_feasible(tug_k: int, solution: dict,
                                        new_task: int, tau_new: float,
                                        position: int = -1) -> bool:
        """Test inserting `new_task` at `position` (default end) of tug_k's
        route with start time tau_new. Other tasks keep their existing
        start times. Returns True iff this single-tug change is feasible.
        """
        route = list(solution["routes"][tug_k])
        if position < 0 or position > len(route):
            position = len(route)
        route.insert(position, new_task)
        new_start_times = dict(solution["task_start_times"])
        new_start_times[new_task] = tau_new
        return has_route_capacity(tug_k, route, new_start_times)

    # ==================================================================
    # (3) Construction
    # ==================================================================
    def find_collaborative_tug_set(task_id: int, tau: float,
                                   solution: dict) -> list[int] | None:
        """Find a set of ≤ Γ_s tugs whose HP sum ≥ H_s^min, all of whom can
        APPEND `task_id` (at start time tau) to their current route without
        violating their individual feasibility / fuel / return constraints.
        Greedy by HP desc, then exhaustive up to size Γ_s. Returns the list
        of tug ids or None if no team works.
        """
        if not (1 <= task_id <= n):
            return None
        H_min = hp_req_arr[task_id - 1]
        Gamma = gamma_arr[task_id - 1]

        # Filter candidates: each must be individually able to append task_id
        # at time tau to the END of its route.
        cands = []
        for k in range(m):
            est_arr = earliest_arrival_after(k, solution["routes"][k],
                                             solution["task_start_times"], task_id)
            if est_arr is None or est_arr > tau + EPS:
                continue
            # Simulate full route after insertion and check feasibility
            if route_after_insertion_feasible(k, solution, task_id, tau):
                cands.append((k, hp_arr[k]))
        if not cands:
            return None

        # Greedy: take highest-HP tugs until HP requirement met
        cands_sorted = sorted(cands, key=lambda x: -x[1])
        sel, hp_sum = [], 0.0
        for k_, hp in cands_sorted:
            if len(sel) >= Gamma:
                break
            sel.append(k_)
            hp_sum += hp
            if hp_sum >= H_min:
                # Also check base capacity for any newly-activated tugs
                if _base_capacity_after(solution, sel):
                    return sel
                break

        # Exhaustive fallback up to Γ_s
        for sz in range(1, Gamma + 1):
            for combo in combinations(cands, sz):
                if sum(hp for _, hp in combo) >= H_min:
                    team = [k_ for k_, _ in combo]
                    if _base_capacity_after(solution, team):
                        return team
        return None

    def _base_capacity_after(solution: dict, new_tugs: list[int]) -> bool:
        """Check (C7/C8) base capacity after activating `new_tugs` (i.e.,
        treating their routes as non-empty if currently empty).
        """
        depart_count = [0] * p
        for k in range(m):
            if solution["routes"][k] or k in new_tugs:
                b = base_of_tug[k]
                depart_count[-b - 1] += 1
        for b_idx in range(p):
            if depart_count[b_idx] > base_cap[b_idx]:
                return False
        return True

    def find_feasible_insertion(task_id: int, solution: dict,
                                 try_starts: int = 7) -> dict | None:
        """Find ONE feasible (tugboat team, start time, append position) for
        task_id against the current solution. Tries up to `try_starts` start
        times spread across the task's time window; for each, attempts to
        find a collaborative tug team. Returns dict with keys
        {tug_ids, tau, position} (position = "end" for append) or None.

        Pass directly into append_task_to_route to apply the assignment.
        """
        if not (1 <= task_id <= n):
            return None
        if task_id in solution["task_tugboats"]:
            return None  # already assigned
        a_s = a_lo[task_id - 1]
        b_s = b_hi[task_id - 1]
        T_s_val = T_s_arr[task_id - 1]
        if a_s > b_s:
            return None
        # Candidate start times: a_s, b_s, and a few in between
        if try_starts <= 1:
            candidates = [a_s]
        else:
            candidates = [a_s + (b_s - a_s) * i / (try_starts - 1)
                          for i in range(try_starts)]
        for tau in candidates:
            if tau + T_s_val > T_max + EPS:
                continue
            team = find_collaborative_tug_set(task_id, tau, solution)
            if team is not None:
                return {"tug_ids": team, "tau": float(tau), "position": "end"}
        return None

    def append_task_to_route(solution: dict, task_id: int, tug_ids: list[int],
                              tau: float) -> dict:
        """Return a NEW solution dict with task_id appended (at the END of
        each tug's route) to every tug in tug_ids. Pure function — does NOT
        mutate `solution`.
        """
        new_routes = [list(r) for r in solution["routes"]]
        for k_ in tug_ids:
            new_routes[k_].append(int(task_id))
        new_tt = dict(solution["task_tugboats"])
        new_tt[int(task_id)] = sorted(int(k_) for k_ in tug_ids)
        new_ts = dict(solution["task_start_times"])
        new_ts[int(task_id)] = float(tau)
        return {
            "routes":           new_routes,
            "task_tugboats":    new_tt,
            "task_start_times": new_ts,
        }

    return {
        "task_window":                       task_window,
        "task_required_hp":                  task_required_hp,
        "tugs_at_base":                      tugs_at_base,
        "unassigned_tasks":                  unassigned_tasks,
        "route_load_status":                 route_load_status,
        "earliest_arrival_after":            earliest_arrival_after,
        "has_route_capacity":                has_route_capacity,
        "route_after_insertion_feasible":    route_after_insertion_feasible,
        "find_collaborative_tug_set":        find_collaborative_tug_set,
        "find_feasible_insertion":           find_feasible_insertion,
        "append_task_to_route":              append_task_to_route,
    }


EXTRA_TOOLS_DESCRIPTION = [
    # ─── Queries ────────────────────────────────────────────────────────────
    {
        "name": "task_window",
        "input": "task_id: int",
        "output": "(float, float)",
        "purpose": (
            "Returns (a_s, b_s) — earliest and latest allowed start times "
            "for task `task_id` (C10), in hours."
        ),
    },
    {
        "name": "task_required_hp",
        "input": "task_id: int",
        "output": "(float, int)",
        "purpose": (
            "Returns (H_s^min, Γ_s) — minimum collective horsepower and the "
            "maximum number of collaborating tugs allowed for task `task_id` "
            "(C1, C2)."
        ),
    },
    {
        "name": "tugs_at_base",
        "input": "base_id: int",
        "output": "list[int]",
        "purpose": (
            "Returns the list of tugboat ids whose home base is `base_id` "
            "(a negative integer in {-1, ..., -p}). Useful for restricting "
            "search to a single base (C3, C4, C9)."
        ),
    },
    {
        "name": "unassigned_tasks",
        "input": "solution: dict",
        "output": "list[int]",
        "purpose": (
            "Sorted list of task ids not yet appearing in any route of "
            "`solution`. Useful as the outer loop of a construction heuristic."
        ),
    },
    {
        "name": "route_load_status",
        "input": "solution: dict",
        "output": "list[dict]",
        "purpose": (
            "For each tug k, returns "
            "{'tug_id', 'num_tasks', 'fuel_used', 'return_time', "
            "'fuel_remaining'} (or 'error' if the current route is infeasible). "
            "Useful for picking which tug still has fuel + time slack."
        ),
    },
    # ─── Feasibility primitives ─────────────────────────────────────────────
    {
        "name": "earliest_arrival_after",
        "input": "tug_k: int, route: list[int], start_times: dict, new_task: int",
        "output": "float | None",
        "purpose": (
            "Returns the earliest time tug_k could arrive at `new_task` if "
            "appended to `route` (using existing start_times for prior tasks). "
            "Returns None if `route` itself is already infeasible. Compare "
            "this against `new_task`'s time window to pick a feasible τ."
        ),
    },
    {
        "name": "has_route_capacity",
        "input": "tug_k: int, route_after: list[int], start_times_after: dict",
        "output": "bool",
        "purpose": (
            "Check whether tug_k can perform `route_after` under the given "
            "start times: arrivals ≤ τ at each task (C12/C13), return to base "
            "by T_max (C11), fuel ≤ tug's capacity (C14). True iff all OK."
        ),
    },
    {
        "name": "route_after_insertion_feasible",
        "input": "tug_k: int, solution: dict, new_task: int, tau_new: float, position: int = -1",
        "output": "bool",
        "purpose": (
            "Try inserting `new_task` at `position` (default end) of tug_k's "
            "current route with start time tau_new, keeping other tasks' start "
            "times fixed. Returns True iff this single-tug change keeps tug_k "
            "feasible (C12/C13/C14)."
        ),
    },
    # ─── Construction ───────────────────────────────────────────────────────
    {
        "name": "find_collaborative_tug_set",
        "input": "task_id: int, tau: float, solution: dict",
        "output": "list[int] | None",
        "purpose": (
            "Find a set of ≤ Γ_s tugs whose horsepower sum ≥ H_s^min and who "
            "each (individually) can append task_id at start time τ to their "
            "current route. Greedy by HP descending; falls back to exhaustive "
            "combinations. Returns the tug list or None if no team is feasible."
        ),
    },
    {
        "name": "find_feasible_insertion",
        "input": "task_id: int, solution: dict, try_starts: int = 7",
        "output": "dict | None",
        "purpose": (
            "Find ONE feasible {tug_ids, tau, position='end'} for task_id "
            "against `solution`. Tries up to `try_starts` start times spread "
            "across the task's window; for each, finds a collaborative tug team. "
            "Returns the dict or None if no feasible insertion exists. Pass "
            "directly into append_task_to_route to apply."
        ),
    },
    {
        "name": "append_task_to_route",
        "input": "solution: dict, task_id: int, tug_ids: list[int], tau: float",
        "output": "dict",
        "purpose": (
            "Return a NEW solution dict with task_id appended to the end of "
            "every tug in tug_ids' route. Pure function — does NOT mutate "
            "`solution`."
        ),
    },
]
