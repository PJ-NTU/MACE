"""Per-problem extras for the Multi-Tugboat Routing and Scheduling Problem (MTRSP).

Provides helpers so the LLM can compose construction heuristics without
reinventing tug-state walking, fuel-cost simulation, or feasible-assignment
search.

Tool groups:
  (1) Queries:             unassigned_tasks, task_time_window,
                           tugs_with_enough_hp_alone
  (2) Feasibility prims:   tug_current_state, route_fuel_cost,
                           is_route_within_capacity, task_arrival_time
  (3) Construction:        find_feasible_assignment, assignment_fuel_delta
  (4) Mutation:            apply_task_assignment

All functions are PURE — no hidden state. Pass `solution` explicitly. They
respect every constraint that `eval_func` checks (C1..C11 — see config.py).

Conventions:
  - solution is the 3-key dict shape returned by `solve()`.
  - Task ids are 1-indexed; tug ids are 0-indexed.
  - The depot's outgoing index is 0 and incoming index is n+1.

These tools are optional — the LLM may use any subset, or write everything
from scratch.
"""
from __future__ import annotations

from itertools import combinations


_TOL = 1e-6


def extra_tools(instance: dict) -> dict:
    """Factory: returns MTRSP-specific tool callables bound to one instance."""
    # ─── Bind instance fields into closure ─────────────────────────────────
    n          = instance["num_tasks"]
    K          = instance["num_tugboats"]
    max_tugs   = instance["task_max_tugs"]
    min_hp     = instance["task_min_horsepower"]
    a          = instance["task_time_window_lower"]
    b          = instance["task_time_window_upper"]
    T_s_arr    = instance["task_service_time"]
    HP         = instance["tugboat_horsepower"]
    F_max      = instance["tugboat_fuel_capacity"]
    alpha      = instance["tugboat_alpha"]
    beta       = instance["tugboat_beta"]
    tm         = instance["time_matrix"]
    T_max      = instance["planning_horizon"]
    end_depot  = n + 1

    # ==================================================================
    # (1) Queries
    # ==================================================================
    def unassigned_tasks(solution: dict) -> list:
        """Returns sorted list of task ids (1-indexed) with no tugs assigned
        (task_tugboats[s] == []), i.e., unexecuted tasks."""
        tt = solution["task_tugboats"]
        return [s for s in range(1, n + 1) if not tt[s]]

    def task_time_window(task_id: int) -> tuple:
        """Returns (a_s, b_s, T_s) for task s. Useful as a one-liner
        replacement for three separate kwargs lookups."""
        if not (1 <= task_id <= n):
            raise ValueError(f"task_id={task_id} out of [1, {n}]")
        s = task_id
        return a[s - 1], b[s - 1], T_s_arr[s - 1]

    def tugs_with_enough_hp_alone(task_id: int) -> list:
        """Tug ids whose HP_k ≥ H_s^min — i.e. could serve task_id WITHOUT
        collaboration. Returns sorted list of tug_ids (0-indexed)."""
        if not (1 <= task_id <= n):
            raise ValueError(f"task_id={task_id} out of [1, {n}]")
        h = min_hp[task_id - 1]
        return [k for k in range(K) if HP[k] >= h]

    # ==================================================================
    # (2) Feasibility primitives
    # ==================================================================
    def tug_current_state(tug_id: int, solution: dict) -> dict:
        """Walk solution.routes[tug_id] using task_start_times and return
        the tug's state AFTER its current route:
            {'current_time': float, 'current_node': int, 'num_tasks': int}
        Empty route ⇒ tug sits at depot at t=0.

        Does NOT include the depot-return leg; this is the state just after
        the tug's LAST serviced task ends (or at depot/t=0 if no tasks)."""
        if not (0 <= tug_id < K):
            raise ValueError(f"tug_id={tug_id} out of [0, {K})")
        route = solution["routes"][tug_id]
        if not route:
            return {"current_time": 0.0, "current_node": 0, "num_tasks": 0}
        last = route[-1]
        finish_time = float(solution["task_start_times"][last]) + T_s_arr[last - 1]
        return {"current_time": finish_time,
                "current_node": last,
                "num_tasks":    len(route)}

    def task_arrival_time(tug_id: int, task_id: int, solution: dict) -> float:
        """Earliest time tug `tug_id` could arrive at task `task_id`'s entrance
        IF appended to the end of its current route. Equals
            tug_current_state.current_time + time_matrix[from_node → task_id].
        Returns float('inf') if there is no direct travel-time entry (the
        task is already in the tug's route, in which case this query is
        meaningless)."""
        st = tug_current_state(tug_id, solution)
        key = f"{st['current_node']}_{task_id}"
        if key not in tm:
            return float("inf")
        return st["current_time"] + tm[key]

    def route_fuel_cost(tug_id: int, route: list) -> float:
        """Total fuel (kg) consumed by tug `tug_id` along `route`, including
        depot ↔ first/last legs. Pass [] for the no-service / no-fuel case."""
        if not (0 <= tug_id < K):
            raise ValueError(f"tug_id={tug_id} out of [0, {K})")
        if not route:
            return 0.0
        hp = HP[tug_id]; al = alpha[tug_id]; be = beta[tug_id]
        total = 0.0
        cur = 0
        for s in route:
            key = f"{cur}_{s}"
            if key not in tm:
                raise ValueError(f"time_matrix missing key {key!r}")
            total += be * hp * tm[key]
            total += al * hp * T_s_arr[s - 1]
            cur = s
        # Return-to-depot leg
        ret_key = f"{cur}_{end_depot}"
        if ret_key not in tm:
            raise ValueError(f"time_matrix missing key {ret_key!r}")
        total += be * hp * tm[ret_key]
        return total

    def is_route_within_capacity(tug_id: int, route: list) -> bool:
        """True iff route_fuel_cost(tug_id, route) ≤ F_max[tug_id] + ε.
        Cheap pre-check before committing to an assignment."""
        return route_fuel_cost(tug_id, route) <= F_max[tug_id] + _TOL

    # ==================================================================
    # (3) Construction
    # ==================================================================
    def find_feasible_assignment(task_id: int, solution: dict) -> dict | None:
        """Find ONE feasible (tug_ids, start_time) for `task_id` against the
        current solution. Returns dict {'tug_ids': list[int], 'start_time': float}
        or None if no combination of ≤ Γ_s tugs can feasibly serve task_id.

        Strategy (greedy, may miss some valid pairings — but covers the common
        case quickly):
          1. Skip if task_id is already executed in `solution`.
          2. For each tug k, compute arrival_k = task_arrival_time(k, task_id, sol).
          3. Filter tugs with arrival_k ≤ b_s.
          4. Sort filtered by (HP descending, arrival ascending).
          5. For sz = 1..Γ_s: take top-sz, check HP sum ≥ H_s^min,
             start = max(a_s, max(arrival of chosen)),
             check start ≤ b_s, start + T_s ≤ T_max,
             check fuel capacity for every chosen tug after appending task_id.
          6. Return first feasible. If sz iteration exhausts, also try the
             best-HP-sum combinations of size 2..Γ_s as a fallback.

        Guarantees:
          - If returned, the implied solution (after apply_task_assignment)
            satisfies C1, C2, C7, C8, C9 / C10, C11 for task_id.
          - Returns None if greedy + fallback both fail (does NOT mean
            infeasible in general; just no obvious assignment).
        """
        s = task_id
        if not (1 <= s <= n):
            raise ValueError(f"task_id={s} out of [1, {n}]")
        if solution["task_tugboats"][s]:
            return None  # task already executed

        Gamma_s = max_tugs[s - 1]
        H_min = min_hp[s - 1]
        a_s, b_s, T_s = a[s - 1], b[s - 1], T_s_arr[s - 1]

        # Step 2-3: collect candidates with arrival within window
        candidates = []  # (arrival, HP_k, k)
        for k in range(K):
            arrival = task_arrival_time(k, s, solution)
            if arrival > b_s + _TOL:
                continue
            candidates.append((arrival, HP[k], k))
        if not candidates:
            return None

        def _try_combo(chosen):
            """chosen = list of (arrival, hp, k) tuples.
            Returns dict or None."""
            hp_sum = sum(c[1] for c in chosen)
            if hp_sum < H_min - _TOL:
                return None
            if len(chosen) > Gamma_s:
                return None
            start = max(a_s, max(c[0] for c in chosen))
            if start > b_s + _TOL:
                return None
            if start + T_s > T_max + _TOL:
                return None
            # Verify fuel capacity for every chosen tug after appending task_id
            for arrival, hp, k in chosen:
                new_route = solution["routes"][k] + [s]
                if not is_route_within_capacity(k, new_route):
                    return None
            return {"tug_ids": sorted(c[2] for c in chosen),
                    "start_time": start}

        # Step 4-5: HP-greedy, then arrival-greedy
        cands_hp = sorted(candidates, key=lambda c: (-c[1], c[0]))
        for sz in range(1, min(Gamma_s, len(cands_hp)) + 1):
            res = _try_combo(cands_hp[:sz])
            if res is not None:
                return res

        # Fallback: explicit combinations for sz = 2..Γ_s (small sets, cheap)
        # We only enumerate combos over the top-min(3*Γ_s, K) by HP to bound work.
        pool = cands_hp[:min(3 * Gamma_s, len(cands_hp))]
        for sz in range(2, min(Gamma_s, len(pool)) + 1):
            for combo in combinations(pool, sz):
                res = _try_combo(list(combo))
                if res is not None:
                    return res

        return None

    def assignment_fuel_delta(solution: dict, task_id: int,
                              tug_ids: list, start_time: float) -> float:
        """Return the *increase* in total Z_fuel from appending task_id at
        start_time to each tug in tug_ids, vs the current solution. Useful
        for picking the cheapest among candidate assignments.

        NOTE: assumes the assignment is feasible (does NOT recheck); use
        find_feasible_assignment or is_route_within_capacity first.
        """
        if not (1 <= task_id <= n):
            raise ValueError(f"task_id={task_id} out of [1, {n}]")
        delta = 0.0
        for k in tug_ids:
            if not (0 <= k < K):
                raise ValueError(f"tug_id={k} out of [0, {K})")
            old = route_fuel_cost(k, solution["routes"][k])
            new = route_fuel_cost(k, solution["routes"][k] + [task_id])
            delta += new - old
        return delta

    # ==================================================================
    # (4) Mutation
    # ==================================================================
    def apply_task_assignment(solution: dict, task_id: int,
                              tug_ids: list, start_time: float) -> dict:
        """Return a NEW solution dict with task_id appended to each tug's
        route, task_tugboats[task_id] set to sorted(tug_ids), and
        task_start_times[task_id] set to start_time.

        Pure function — does NOT modify `solution`. Preserves strict-key shape
        (task_tugboats / task_start_times keys remain {1..n})."""
        if not (1 <= task_id <= n):
            raise ValueError(f"task_id={task_id} out of [1, {n}]")
        tug_ids_sorted = sorted(int(k) for k in tug_ids)
        for k in tug_ids_sorted:
            if not (0 <= k < K):
                raise ValueError(f"tug_id={k} out of [0, {K})")

        new_routes = [r[:] for r in solution["routes"]]
        for k in tug_ids_sorted:
            new_routes[k].append(int(task_id))

        new_tt = dict(solution["task_tugboats"])
        new_tt[task_id] = tug_ids_sorted

        new_ts = dict(solution["task_start_times"])
        new_ts[task_id] = float(start_time)

        return {"routes":           new_routes,
                "task_tugboats":    new_tt,
                "task_start_times": new_ts}

    return {
        # Queries
        "unassigned_tasks":            unassigned_tasks,
        "task_time_window":            task_time_window,
        "tugs_with_enough_hp_alone":   tugs_with_enough_hp_alone,
        # Feasibility primitives
        "tug_current_state":           tug_current_state,
        "task_arrival_time":           task_arrival_time,
        "route_fuel_cost":             route_fuel_cost,
        "is_route_within_capacity":    is_route_within_capacity,
        # Construction
        "find_feasible_assignment":    find_feasible_assignment,
        "assignment_fuel_delta":       assignment_fuel_delta,
        # Mutation
        "apply_task_assignment":       apply_task_assignment,
    }


EXTRA_TOOLS_DESCRIPTION = [
    # ─── Queries ───────────────────────────────────────────────────────
    {
        "name": "unassigned_tasks",
        "input": "solution: dict",
        "output": "list[int]",
        "purpose": (
            "Task ids (1-indexed) with empty task_tugboats[s] in `solution`. "
            "Useful as the outer loop of a construction heuristic."
        ),
    },
    {
        "name": "task_time_window",
        "input": "task_id: int",
        "output": "(float, float, float)",
        "purpose": (
            "Returns (a_s, b_s, T_s) for task_id — earliest start, latest "
            "start, service duration. One-liner replacement for three "
            "separate kwargs lookups."
        ),
    },
    {
        "name": "tugs_with_enough_hp_alone",
        "input": "task_id: int",
        "output": "list[int]",
        "purpose": (
            "Tug ids whose HP_k ≥ H_s^min — could serve task_id WITHOUT "
            "collaboration (C2 satisfied by a single tug). Sorted ascending."
        ),
    },
    # ─── Feasibility primitives ────────────────────────────────────────
    {
        "name": "tug_current_state",
        "input": "tug_id: int, solution: dict",
        "output": "dict {'current_time': float, 'current_node': int, 'num_tasks': int}",
        "purpose": (
            "Walk solution.routes[tug_id] using task_start_times and return "
            "the tug's state AFTER its current route. current_node = last "
            "serviced task id (or 0 if no tasks); current_time = when the "
            "last task finishes (or 0.0)."
        ),
    },
    {
        "name": "task_arrival_time",
        "input": "tug_id: int, task_id: int, solution: dict",
        "output": "float",
        "purpose": (
            "Earliest time tug tug_id could arrive at task task_id's "
            "entrance IF appended to its current route. Equals "
            "tug_current_state.current_time + time_matrix[from→task_id]. "
            "Returns +∞ if the implied travel arc has no time_matrix entry."
        ),
    },
    {
        "name": "route_fuel_cost",
        "input": "tug_id: int, route: list[int]",
        "output": "float",
        "purpose": (
            "Total fuel (kg) tug tug_id consumes along `route`, including "
            "depot→first and last→depot legs. Used to check C11."
        ),
    },
    {
        "name": "is_route_within_capacity",
        "input": "tug_id: int, route: list[int]",
        "output": "bool",
        "purpose": (
            "True iff route_fuel_cost(tug_id, route) ≤ F_max[tug_id] (C11). "
            "Cheap pre-check before committing to an assignment."
        ),
    },
    # ─── Construction ──────────────────────────────────────────────────
    {
        "name": "find_feasible_assignment",
        "input": "task_id: int, solution: dict",
        "output": "dict | None",
        "purpose": (
            "Find ONE feasible (tug_ids, start_time) for task_id against the "
            "current solution. Greedy by tug HP descending, then "
            "arrival-time ascending; falls back to small-combo exhaustive "
            "search over top-3·Γ_s candidates. Returns dict with keys "
            "{'tug_ids': list[int], 'start_time': float} or None if no "
            "obvious assignment fits all of (C1, C2, C7, C8, C9/C10, C11). "
            "Pass directly into apply_task_assignment."
        ),
    },
    {
        "name": "assignment_fuel_delta",
        "input": "solution: dict, task_id: int, tug_ids: list[int], start_time: float",
        "output": "float",
        "purpose": (
            "How much Z_fuel goes UP if you append task_id to every tug in "
            "tug_ids? Compare against W (penalty_weight) to decide whether "
            "serving the task beats skipping it — typically yes since W is "
            "10000 kg-equivalent."
        ),
    },
    # ─── Mutation ──────────────────────────────────────────────────────
    {
        "name": "apply_task_assignment",
        "input": "solution: dict, task_id: int, tug_ids: list[int], start_time: float",
        "output": "dict",
        "purpose": (
            "Return a NEW solution dict with task_id appended to each tug's "
            "route, task_tugboats[task_id] set, and task_start_times[task_id] "
            "set. Pure — does NOT mutate `solution`. Preserves strict-key "
            "shape (task_tugboats / task_start_times keys remain {1..n})."
        ),
    },
]
