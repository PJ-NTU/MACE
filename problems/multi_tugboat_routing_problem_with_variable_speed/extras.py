"""Per-problem extras for the Multi-Tugboat Routing Problem with Variable Speed (MTRSP-VS).

Provides helpers so the LLM can compose construction heuristics without
reinventing speed-cost arithmetic, distance lookups, fuel checks, or
tug-combo search.

Tool groups:
  (1) Queries:                 service_time, transit_time, service_fuel,
                               transit_fuel, depot_distance,
                               unexecuted_tasks
  (2) Feasibility primitives:  task_tugboats, tug_total_fuel,
                               feasible_tug_combinations
  (3) Construction (solo):     find_feasible_assignment, append_task_to_tug
  (4) Construction (collab):   find_collaborative_assignment,
                               append_collaborative_task

All functions are PURE — no hidden state. Pass `solution` explicitly. They
respect every constraint that `eval_func` checks (C1..C13 — see config.py).

These tools are optional — the LLM may use any subset, or write everything
from scratch.
"""
from __future__ import annotations

from itertools import combinations


def extra_tools(instance: dict) -> dict:
    """Factory: returns MTRSP-VS-specific tool callables bound to one instance."""
    # ─── Bind instance fields into closure ────────────────────────────────
    n = instance["num_tasks"]
    m = instance["num_tugboats"]
    L = instance["num_speed_levels"]
    T_max = instance["planning_horizon"]

    task_max_tugs   = instance["task_max_tugs"]
    task_min_hp     = instance["task_min_horsepower"]
    task_tw_lower   = instance["task_time_window_lower"]
    task_tw_upper   = instance["task_time_window_upper"]
    task_service_d  = instance["task_service_distance"]

    tug_hp     = instance["tugboat_horsepower"]
    tug_fuel_cap = instance["tugboat_fuel_capacity"]
    tug_alpha  = instance["tugboat_alpha"]
    tug_beta   = instance["tugboat_beta"]

    speed_values = instance["speed_values"]
    v_medium = speed_values[1]

    depot_to_task = instance["depot_to_task_distance"]
    task_to_depot = instance["task_to_depot_distance"]
    task_to_task  = instance["task_to_task_distance"]

    # ==================================================================
    # (1) Queries
    # ==================================================================
    def service_time(s: int, speed_level: int) -> float:
        """Service duration for task s at speed level ℓ: dₛ / vₗ (hours)."""
        return task_service_d[s] / speed_values[speed_level]

    def transit_time(i: int, j: int, speed_level: int) -> float:
        """Transit duration for arc i→j at speed level ℓ. i, j ∈
        {-1 (depot)} ∪ {0..n-1}. j = -1 means return-to-depot from task i.
        Returns dᵢⱼ / vₗ (hours)."""
        v = speed_values[speed_level]
        if i == -1:
            return depot_to_task[j] / v
        if j == -1:
            return task_to_depot[i] / v
        return task_to_task[i][j] / v

    def service_fuel(s: int, k: int, speed_level: int) -> float:
        """Service fuel for tug k servicing task s at speed ℓ (kg).
        ψ = αₖ · HPₖ · dₛ · (vₗ² / v_medium³)."""
        v = speed_values[speed_level]
        return tug_alpha[k] * tug_hp[k] * task_service_d[s] * (v ** 2) / (v_medium ** 3)

    def transit_fuel(i: int, j: int, k: int, speed_level: int) -> float:
        """Transit fuel on arc i→j for tug k at speed ℓ (kg).
        φ = βₖ · HPₖ · dᵢⱼ · (vₗ² / v_medium³).
        i = -1 means depot→j; j = -1 means i→depot."""
        v = speed_values[speed_level]
        if i == -1:
            d = depot_to_task[j]
        elif j == -1:
            d = task_to_depot[i]
        else:
            d = task_to_task[i][j]
        return tug_beta[k] * tug_hp[k] * d * (v ** 2) / (v_medium ** 3)

    def depot_distance(s: int, direction: str = "to") -> float:
        """direction='to'  → depot → task s entrance (n.m.)
           direction='from' → task s exit → depot (n.m.)"""
        if direction == "to":
            return depot_to_task[s]
        if direction == "from":
            return task_to_depot[s]
        raise ValueError(f"direction must be 'to' or 'from', got {direction!r}")

    def unexecuted_tasks(solution: dict) -> list:
        """Sorted list of task_ids NOT appearing in any route."""
        executed = set()
        for k in range(m):
            executed.update(solution["routes"].get(k, []))
        return [s for s in range(n) if s not in executed]

    # ==================================================================
    # (2) Feasibility primitives
    # ==================================================================
    def task_tugboats(solution: dict, task_id: int) -> list:
        """List of tug_ids whose route contains task_id (collaborative if >1)."""
        return [k for k in range(m) if task_id in solution["routes"].get(k, [])]

    def tug_total_fuel(solution: dict, k: int) -> float:
        """Compute current total fuel for tug k under `solution` (sum of
        service + transit). Useful before appending another task to check
        C13. Returns 0.0 if tug k's route is empty."""
        route = solution["routes"].get(k, [])
        if not route:
            return 0.0
        speeds_k = solution["transit_speeds"].get(k, [])
        if len(speeds_k) != len(route) + 1:
            # Inconsistent solution shape — return inf to flag.
            return float("inf")
        total = 0.0
        for s in route:
            lvl = solution["service_speeds"].get(s)
            if lvl is None:
                return float("inf")
            total += service_fuel(s, k, lvl)
        # depot → s0
        total += transit_fuel(-1, route[0], k, speeds_k[0])
        for i in range(1, len(route)):
            total += transit_fuel(route[i - 1], route[i], k, speeds_k[i])
        total += transit_fuel(route[-1], -1, k, speeds_k[-1])
        return total

    def feasible_tug_combinations(task_id: int, max_size: int = None) -> list:
        """Enumerate all tug subsets of size 1..max_size (default = task_max_tugs[s])
        whose HP sum ≥ task_min_horsepower[task_id]. Returns sorted list of
        tuples of tug_ids, prioritized by HP sum descending."""
        req = task_min_hp[task_id]
        cap = max_size if max_size is not None else task_max_tugs[task_id]
        cap = min(cap, m)
        feas = []
        for sz in range(1, cap + 1):
            for combo in combinations(range(m), sz):
                if sum(tug_hp[k] for k in combo) >= req:
                    feas.append(combo)
        # Sort: prefer smaller crews, then higher total HP
        feas.sort(key=lambda c: (len(c), -sum(tug_hp[k] for k in c)))
        return feas

    # ==================================================================
    # (3) Construction
    # ==================================================================
    def find_feasible_assignment(task_id: int, solution: dict,
                                 prefer_speed: int = 0) -> dict | None:
        """Find ONE feasible way to add task_id to `solution` from scratch
        (i.e., serve it as a SOLO assignment by one tug, appending to that
        tug's route after its last task). Tries: (a) every tug with enough
        spare time + fuel, (b) start times within the task's window,
        (c) speeds in order [prefer_speed, ...rest] — default 0 (slow) so
        the first feasible match is the most fuel-efficient that fits.

        Returns a dict you can pass to `append_task_to_tug`:
            {'tug_id', 'start_time', 'service_speed',
             'transit_speed_to', 'transit_speed_from'}
        or None if no feasible (tug, time, speeds) triple exists.

        NOTE: Only solo assignments — tugs with HP_k < H_s^min are skipped.
        For tasks needing collaborative (multi-tug) service, use
        `find_collaborative_assignment` instead.
        """
        a_s = task_tw_lower[task_id]
        b_s = task_tw_upper[task_id]
        hp_req = task_min_hp[task_id]
        d_s = task_service_d[task_id]

        for k in range(m):
            if tug_hp[k] < hp_req:
                continue
            # Earliest time tug k can be at task_id's entrance
            route = solution["routes"].get(k, [])
            if not route:
                # depot → task_id directly
                origin = -1
                ready_time = 0.0
            else:
                last_task = route[-1]
                lvl_last_service = solution["service_speeds"].get(last_task)
                if lvl_last_service is None:
                    continue
                ready_time = (solution["start_times"][last_task]
                              + service_time(last_task, lvl_last_service))
                origin = last_task

            # Try speeds in preferred order
            speed_order = [prefer_speed] + [l for l in range(L) if l != prefer_speed]
            current_fuel = tug_total_fuel(solution, k)
            cap = tug_fuel_cap[k]

            for trans_to in speed_order:
                tt = transit_time(origin, task_id, trans_to)
                earliest_start = max(a_s, ready_time + tt)
                if earliest_start > b_s + 1e-6:
                    continue
                for svc in speed_order:
                    T_svc = service_time(task_id, svc)
                    if earliest_start + T_svc > T_max + 1e-6:
                        continue
                    for trans_from in speed_order:
                        # Tentative new fuel for k
                        new_fuel = (current_fuel
                                    - transit_fuel(origin, -1, k,
                                                   solution["transit_speeds"].get(k, [1])[-1])
                                        if route else current_fuel)
                        # Recompute correctly: drop OLD return arc if any, add new transit
                        # to task + new service + new return.
                        if route:
                            old_return_speed = solution["transit_speeds"][k][-1]
                            new_fuel = (current_fuel
                                        - transit_fuel(origin, -1, k, old_return_speed)
                                        + transit_fuel(origin, task_id, k, trans_to)
                                        + service_fuel(task_id, k, svc)
                                        + transit_fuel(task_id, -1, k, trans_from))
                        else:
                            new_fuel = (transit_fuel(-1, task_id, k, trans_to)
                                        + service_fuel(task_id, k, svc)
                                        + transit_fuel(task_id, -1, k, trans_from))
                        if new_fuel <= cap + 1e-6:
                            return {
                                "tug_id":             k,
                                "start_time":         earliest_start,
                                "service_speed":      svc,
                                "transit_speed_to":   trans_to,
                                "transit_speed_from": trans_from,
                            }
        return None

    def find_collaborative_assignment(task_id: int, solution: dict,
                                       prefer_speed: int = 0) -> dict | None:
        """Find ONE feasible COLLABORATIVE assignment (1..Γ_s tugs) for
        task_id by appending it to the END of each chosen tug's route, all
        tugs synced to the same start_time. All chosen tugs use the same
        service_speed, transit_speed_to, and transit_speed_from (default =
        prefer_speed = 0 / slow, since service fuel scales as v²/v_medium³
        — slow uses ~64% less fuel than medium per service distance).

        This is a STRICT SUPERSET of `find_feasible_assignment`: it also
        handles solo (size-1 combo) cases, AND handles tasks whose
        H_s^min > max single tug HP (which `find_feasible_assignment`
        cannot serve at all).

        Strategy:
          1. Enumerate combos via `feasible_tug_combinations` (sorted:
             smallest crew first, then highest HP).
          2. For each combo:
             a. For each tug, find its ready_time + origin (last task or
                depot) under `solution`.
             b. arrival_k = ready_time_k + transit_time(origin_k, task_id,
                                                       prefer_speed)
             c. synced_start = max(a_s, max(arrival_k for k in combo)).
                Tugs arriving earlier just wait — no fuel cost for waiting.
             d. Check: synced_start ≤ b_s; synced_start + T_svc ≤ T_max.
             e. For each tug k: drop tug k's OLD return-arc fuel; add
                new transit_to + service + transit_from fuel; check
                ≤ tug_fuel_cap[k].
          3. Return first feasible combo:
             {'tug_ids': list[int], 'start_time': float,
              'service_speed': int, 'transit_speed_to': int,
              'transit_speed_from': int}
             — pass to `append_collaborative_task` to apply.

        Returns None if NO combo is feasible at `prefer_speed`. Call again
        with a different prefer_speed (e.g., 1 / medium for tight time
        windows where slow service exceeds T_max) to retry.
        """
        a_s = task_tw_lower[task_id]
        b_s = task_tw_upper[task_id]
        d_s = task_service_d[task_id]

        # Speed iteration order: prefer_speed first, then the rest
        speed_order = [prefer_speed] + [l for l in range(L) if l != prefer_speed]

        # Per-tug current ready_time + origin under `solution`
        ready_origin = {}
        for k in range(m):
            route_k = solution["routes"].get(k, [])
            if not route_k:
                ready_origin[k] = (0.0, -1)
                continue
            last_task = route_k[-1]
            lvl = solution["service_speeds"].get(last_task)
            if lvl is None:
                continue  # inconsistent solution state for this tug
            rt = solution["start_times"][last_task] + service_time(last_task, lvl)
            ready_origin[k] = (rt, last_task)

        combos = feasible_tug_combinations(task_id)
        for combo in combos:
            # Skip combos containing a tug in an inconsistent state
            if not all(k in ready_origin for k in combo):
                continue
            # Try (transit_to, service, transit_from) ∈ 27 speed triples,
            # prefer_speed-first. Same speed for all tugs in combo.
            for trans_to in speed_order:
                arrivals = [
                    ready_origin[k][0] + transit_time(ready_origin[k][1], task_id, trans_to)
                    for k in combo
                ]
                start = max(a_s, max(arrivals))
                if start > b_s + 1e-6:
                    continue
                for svc in speed_order:
                    T_svc = d_s / speed_values[svc]
                    if start + T_svc > T_max + 1e-6:
                        continue
                    for trans_from in speed_order:
                        ok = True
                        for k in combo:
                            _, og = ready_origin[k]
                            cur_fuel = tug_total_fuel(solution, k)
                            if solution["routes"].get(k, []):
                                old_return_speed = solution["transit_speeds"][k][-1]
                                new_fuel = (cur_fuel
                                            - transit_fuel(og, -1, k, old_return_speed)
                                            + transit_fuel(og, task_id, k, trans_to)
                                            + service_fuel(task_id, k, svc)
                                            + transit_fuel(task_id, -1, k, trans_from))
                            else:
                                new_fuel = (transit_fuel(-1, task_id, k, trans_to)
                                            + service_fuel(task_id, k, svc)
                                            + transit_fuel(task_id, -1, k, trans_from))
                            if new_fuel > tug_fuel_cap[k] + 1e-6:
                                ok = False
                                break
                        if ok:
                            return {
                                "tug_ids":            list(combo),
                                "start_time":         float(start),
                                "service_speed":      int(svc),
                                "transit_speed_to":   int(trans_to),
                                "transit_speed_from": int(trans_from),
                            }
        return None

    def append_collaborative_task(solution: dict, task_id: int,
                                   tug_ids: list, start_time: float,
                                   service_speed: int,
                                   transit_speed_to: int,
                                   transit_speed_from: int) -> dict:
        """Return a NEW solution with task_id appended to the END of each
        tug in `tug_ids`. All tugs use the same transit_speed_to /
        service_speed / transit_speed_from (matches what
        `find_collaborative_assignment` returns).

        Pure function — does NOT mutate `solution`. Strict-shape preserved
        (all m tugs as keys). For per-tug different speeds, call
        `append_task_to_tug` once per tug instead.
        """
        new_routes = {k: list(v) for k, v in solution["routes"].items()}
        new_svc    = dict(solution["service_speeds"])
        new_start  = dict(solution["start_times"])
        new_trans  = {k: list(v) for k, v in solution["transit_speeds"].items()}

        for k in tug_ids:
            route_k = new_routes[k]
            if route_k:
                new_trans[k].pop()  # drop old return-arc speed
            route_k.append(int(task_id))
            new_trans[k].append(int(transit_speed_to))
            new_trans[k].append(int(transit_speed_from))

        new_svc[task_id]   = int(service_speed)
        new_start[task_id] = float(start_time)

        return {
            "routes":         new_routes,
            "service_speeds": new_svc,
            "start_times":    new_start,
            "transit_speeds": new_trans,
        }

    def append_task_to_tug(solution: dict, task_id: int, tug_id: int,
                           start_time: float, service_speed: int,
                           transit_speed_to: int, transit_speed_from: int) -> dict:
        """Return a NEW solution dict with task_id appended to tug_id's
        route. Pure function — does NOT mutate `solution`. Returns the
        modified dict (strict shape preserved). If tug_id's route was
        nonempty, the OLD return-arc speed is replaced.
        """
        new_routes = {k: list(v) for k, v in solution["routes"].items()}
        new_svc    = dict(solution["service_speeds"])
        new_start  = dict(solution["start_times"])
        new_trans  = {k: list(v) for k, v in solution["transit_speeds"].items()}

        route = new_routes[tug_id]
        if route:
            # Drop old return-arc speed
            new_trans[tug_id].pop()
        route.append(task_id)
        new_trans[tug_id].append(int(transit_speed_to))
        new_trans[tug_id].append(int(transit_speed_from))
        new_svc[task_id]   = int(service_speed)
        new_start[task_id] = float(start_time)

        return {
            "routes":         new_routes,
            "service_speeds": new_svc,
            "start_times":    new_start,
            "transit_speeds": new_trans,
        }

    return {
        "service_time":                  service_time,
        "transit_time":                  transit_time,
        "service_fuel":                  service_fuel,
        "transit_fuel":                  transit_fuel,
        "depot_distance":                depot_distance,
        "unexecuted_tasks":              unexecuted_tasks,
        "task_tugboats":                 task_tugboats,
        "tug_total_fuel":                tug_total_fuel,
        "feasible_tug_combinations":     feasible_tug_combinations,
        "find_feasible_assignment":      find_feasible_assignment,
        "find_collaborative_assignment": find_collaborative_assignment,
        "append_task_to_tug":            append_task_to_tug,
        "append_collaborative_task":     append_collaborative_task,
    }


EXTRA_TOOLS_DESCRIPTION = [
    # ─── Queries ───────────────────────────────────────────────────────
    {
        "name": "service_time",
        "input": "s: int, speed_level: int",
        "output": "float",
        "purpose": (
            "Service duration in hours for task s at speed level ℓ ∈ {0,1,2}: "
            "Tₛ^ℓ = dₛ / vₗ. Pair with start_times[s] to compute service finish."
        ),
    },
    {
        "name": "transit_time",
        "input": "i: int, j: int, speed_level: int",
        "output": "float",
        "purpose": (
            "Transit duration in hours on arc i→j at speed ℓ. i=-1 means depot, "
            "j=-1 means depot. Returns dᵢⱼ / vₗ. Convenient for building feasible "
            "arrival times along a route."
        ),
    },
    {
        "name": "service_fuel",
        "input": "s: int, k: int, speed_level: int",
        "output": "float",
        "purpose": (
            "Service fuel (kg) for tug k servicing task s at speed ℓ: "
            "ψ = αₖ·HPₖ·dₛ·(vₗ²/v_medium³). Useful to budget fuel before "
            "committing a tug to a task."
        ),
    },
    {
        "name": "transit_fuel",
        "input": "i: int, j: int, k: int, speed_level: int",
        "output": "float",
        "purpose": (
            "Transit fuel (kg) for tug k on arc i→j at speed ℓ: "
            "φ = βₖ·HPₖ·dᵢⱼ·(vₗ²/v_medium³). i=-1 ⇒ depot→j, j=-1 ⇒ i→depot."
        ),
    },
    {
        "name": "depot_distance",
        "input": "s: int, direction: str ('to' | 'from')",
        "output": "float",
        "purpose": (
            "Depot↔task distance in nautical miles. direction='to': depot → "
            "task s entrance. direction='from': task s exit → depot."
        ),
    },
    {
        "name": "unexecuted_tasks",
        "input": "solution: dict",
        "output": "list[int]",
        "purpose": (
            "Task ids NOT in any route. Useful as the outer loop of a "
            "construction heuristic."
        ),
    },
    # ─── Feasibility primitives ────────────────────────────────────────
    {
        "name": "task_tugboats",
        "input": "solution: dict, task_id: int",
        "output": "list[int]",
        "purpose": (
            "List of tug ids whose route contains task_id. Length > 1 means "
            "collaborative service. Pair with task_max_tugs[task_id] to check "
            "C1 before adding another tug."
        ),
    },
    {
        "name": "tug_total_fuel",
        "input": "solution: dict, k: int",
        "output": "float",
        "purpose": (
            "Current total fuel consumed by tug k under `solution` (sum of "
            "service + transit + return). Use to check C13 before adding "
            "another task to tug k's route. Returns inf if solution shape is "
            "inconsistent for tug k."
        ),
    },
    {
        "name": "feasible_tug_combinations",
        "input": "task_id: int, max_size: int | None",
        "output": "list[tuple[int, ...]]",
        "purpose": (
            "Enumerate tug subsets of size 1..task_max_tugs[task_id] whose HP "
            "sum ≥ task_min_horsepower[task_id] (C1 + C2 satisfied by "
            "construction). Sorted: smallest crew first, then highest total HP. "
            "max_size optionally caps the crew size (e.g., 1 for solo only)."
        ),
    },
    # ─── Construction ──────────────────────────────────────────────────
    {
        "name": "find_feasible_assignment",
        "input": "task_id: int, solution: dict, prefer_speed: int = 0",
        "output": "dict | None",
        "purpose": (
            "Find ONE SOLO feasible assignment for task_id by appending it to "
            "some tug's existing route. Considers fuel capacity (C13), time "
            "window (C9), horizon (C10), and arrival time (C11/C12). Returns "
            "{tug_id, start_time, service_speed, transit_speed_to, transit_speed_from} "
            "to feed directly into append_task_to_tug, or None. SOLO ONLY — "
            "skips tugs with HP_k < H_s^min, so cannot serve tasks that REQUIRE "
            "collaboration. For those, use find_collaborative_assignment "
            "(strict superset of this tool — handles both solo and multi-tug)."
        ),
    },
    {
        "name": "find_collaborative_assignment",
        "input": "task_id: int, solution: dict, prefer_speed: int = 0",
        "output": "dict | None",
        "purpose": (
            "Find ONE feasible collaborative assignment (1..Γ_s tugs synced at "
            "the same start_time) for task_id. STRICT SUPERSET of "
            "find_feasible_assignment — handles solo (size-1 combo) AND tasks "
            "whose H_s^min exceeds every single tug's HP. Enumerates combos "
            "via feasible_tug_combinations (smallest crew first, then highest "
            "HP); for each combo computes synced start = max(arrival_k); "
            "checks time window + horizon + per-tug fuel after appending the "
            "new transit_to + service + return arcs. All chosen tugs use the "
            "SAME prefer_speed for service/transit (default 0 / slow — most "
            "fuel-efficient when time permits). Returns "
            "{tug_ids: list[int], start_time, service_speed, transit_speed_to, "
            "transit_speed_from} to pass into append_collaborative_task, or None. "
            "If None, retry with a different prefer_speed (e.g., 1 / medium) — "
            "slow service may exceed T_max for long-distance tasks."
        ),
    },
    {
        "name": "append_task_to_tug",
        "input": ("solution: dict, task_id: int, tug_id: int, start_time: float, "
                  "service_speed: int, transit_speed_to: int, transit_speed_from: int"),
        "output": "dict",
        "purpose": (
            "Return a NEW solution with task_id appended to tug_id's route. "
            "Drops the tug's OLD return-arc speed and adds the new transit-to + "
            "new return-from arcs. Pure function — does NOT mutate `solution`. "
            "Strict-shape preserved (all m tugs as keys)."
        ),
    },
    {
        "name": "append_collaborative_task",
        "input": ("solution: dict, task_id: int, tug_ids: list[int], "
                  "start_time: float, service_speed: int, "
                  "transit_speed_to: int, transit_speed_from: int"),
        "output": "dict",
        "purpose": (
            "Return a NEW solution with task_id appended to the END of each "
            "tug in tug_ids' route, all sharing start_time / service_speed / "
            "transit speeds (matches find_collaborative_assignment's output). "
            "Pure function — does NOT mutate `solution`. Strict-shape "
            "preserved. For per-tug different speeds, call append_task_to_tug "
            "once per tug instead."
        ),
    },
]
