"""Per-problem extras for the Port Scheduling Problem (PSP).

Provides helpers so the LLM can compose construction heuristics without
reinventing tug-combo search, berth-conflict checks, or ETA-window queries.

Tool groups:
  (1) Queries:                 compatible_berths, vessel_time_window,
                               unassigned_vessels
  (2) Feasibility primitives:  is_berth_available, is_tugboat_available
  (3) Construction:            find_tugboat_combination,
                               find_feasible_assignment, assignment_cost
  (4) Mutation:                apply_assignment

All functions are PURE — no hidden state. Pass `solution` explicitly. They
respect every constraint that `eval_func` checks (C2..C15 — see config.py).

Conventions:
  - solution is the 3-key dict-of-int-keys shape returned by `solve()`.
  - tug pairs are [tug_id, t] lists (not tuples).
  - 'kind' parameter is the literal string 'inbound' or 'outbound'.

These tools are optional — the LLM may use any subset, or write everything
from scratch.
"""
from __future__ import annotations

from itertools import combinations


def extra_tools(instance: dict) -> dict:
    """Factory: returns PSP-specific tool callables bound to one instance."""
    # ─── Bind instance fields into closure ────────────────────────────────
    n_vessels = instance["vessel_num"]
    n_berths  = instance["berth_num"]
    n_tugs    = instance["tugboat_num"]
    T         = instance["time_periods"]
    H_max     = instance["max_tugboats_per_service"]
    eps_time  = instance["time_constraint_tolerance"]
    rho_in    = instance["inbound_preparation_time"]
    rho_out   = instance["outbound_preparation_time"]

    sizes      = instance["vessel_sizes"]
    etas       = instance["vessel_etas"]
    durations  = instance["vessel_durations"]
    tau_ins    = instance["vessel_inbound_service_times"]
    tau_outs   = instance["vessel_outbound_service_times"]
    alpha      = instance["vessel_priority_weights"]
    beta       = instance["vessel_waiting_costs"]
    gamma      = instance["vessel_jit_costs"]
    hp_reqs    = instance["vessel_horsepower_requirements"]
    earlies    = instance["vessel_early_limits"]
    lates      = instance["vessel_late_limits"]

    berth_caps = instance["berth_capacities"]
    tug_hps    = instance["tugboat_horsepower"]
    tug_costs  = instance["tugboat_costs"]

    # Precompute: compatible berths per vessel (Tier-1 lookup)
    _compat_cache = [
        [j for j, c in enumerate(berth_caps) if c >= sizes[i]]
        for i in range(n_vessels)
    ]

    # ==================================================================
    # (1) Queries
    # ==================================================================
    def compatible_berths(vessel_id: int) -> list:
        """List of berth_ids with capacity ≥ vessel_size (C2). O(1) cached."""
        if not (0 <= vessel_id < n_vessels):
            raise ValueError(f"vessel_id={vessel_id} out of [0, {n_vessels})")
        return list(_compat_cache[vessel_id])

    def vessel_time_window(vessel_id: int) -> tuple:
        """Returns (earliest_t_in, latest_t_in) per C13, clamped to [0, T)."""
        if not (0 <= vessel_id < n_vessels):
            raise ValueError(f"vessel_id={vessel_id} out of [0, {n_vessels})")
        earliest = max(0, etas[vessel_id] - earlies[vessel_id])
        latest   = min(T - 1, etas[vessel_id] + lates[vessel_id])
        return earliest, latest

    def unassigned_vessels(solution: dict) -> list:
        """Returns sorted list of vessel_ids with vessel_assignments[i] is None."""
        va = solution["vessel_assignments"]
        return [i for i in range(n_vessels) if va.get(i) is None]

    # ==================================================================
    # (2) Feasibility primitives
    # ==================================================================
    def is_berth_available(berth_id: int, t_b: int, vessel_id: int,
                           solution: dict) -> bool:
        """Check C11: is berth_id free during [t_b, t_b + D_vessel)?

        Conflicts with the vessel itself (if already assigned) do NOT count.
        Pass the vessel_id you're trying to (re)assign so its own slot is
        excluded from the conflict check.
        """
        if not (0 <= berth_id < n_berths):
            return False
        D_i = durations[vessel_id]
        new_end = t_b + D_i
        if t_b < 0 or new_end > T:
            return False
        for v, va in solution["vessel_assignments"].items():
            if va is None or v == vessel_id:
                continue
            existing_berth, existing_start = va
            if existing_berth != berth_id:
                continue
            existing_end = existing_start + durations[v]
            # overlap iff NOT (a ends before b starts OR a starts after b ends)
            if not (new_end <= existing_start or t_b >= existing_end):
                return False
        return True

    def is_tugboat_available(tug_id: int, t: int, kind: str, vessel_id: int,
                             solution: dict) -> bool:
        """Check C12: is tug_id free for a `kind` service starting at t,
        accounting for its prep time?

        `kind` ∈ {'inbound', 'outbound'}. The vessel_id's own slot is excluded
        from the conflict check (so you can re-place it without self-conflict).
        """
        if not (0 <= tug_id < n_tugs):
            return False
        if kind == "inbound":
            new_end = t + tau_ins[vessel_id] + rho_in
        elif kind == "outbound":
            new_end = t + tau_outs[vessel_id] + rho_out
        else:
            raise ValueError(f"kind must be 'inbound' or 'outbound', got {kind!r}")
        if t < 0 or new_end > T:
            return False
        for v in range(n_vessels):
            if v == vessel_id:
                continue
            for (k_, t_) in solution["inbound_tugboats"].get(v, []):
                if k_ != tug_id:
                    continue
                ex_end = t_ + tau_ins[v] + rho_in
                if not (new_end <= t_ or t >= ex_end):
                    return False
            for (k_, t_) in solution["outbound_tugboats"].get(v, []):
                if k_ != tug_id:
                    continue
                ex_end = t_ + tau_outs[v] + rho_out
                if not (new_end <= t_ or t >= ex_end):
                    return False
        return True

    # ==================================================================
    # (3) Construction
    # ==================================================================
    def find_tugboat_combination(vessel_id: int, t: int, kind: str,
                                 solution: dict) -> tuple:
        """Find ≤ H_max tugs whose HP sum ≥ vessel's requirement, all
        available for `kind` service starting at t. Greedy by HP descending;
        falls back to exhaustive combinations if greedy fails. Returns
        (tugs, cost) where tugs is [[tug_id, t], ...] or (None, 0) if no
        combination found.
        """
        req = hp_reqs[vessel_id]
        if kind == "inbound":
            dur = tau_ins[vessel_id]
        elif kind == "outbound":
            dur = tau_outs[vessel_id]
        else:
            raise ValueError(f"kind must be 'inbound' or 'outbound', got {kind!r}")

        avail = [
            (k_, tug_hps[k_], tug_costs[k_])
            for k_ in range(n_tugs)
            if is_tugboat_available(k_, t, kind, vessel_id, solution)
        ]
        if not avail:
            return None, 0
        if sum(hp for _, hp, _ in avail) < req:
            return None, 0

        # Greedy: take highest-HP tugs until HP requirement met.
        avail_sorted = sorted(avail, key=lambda x: -x[1])
        selected, total_hp, total_cost = [], 0, 0.0
        for k_, hp, c in avail_sorted:
            if len(selected) >= H_max:
                break
            selected.append([k_, t])
            total_hp += hp
            total_cost += c * dur
            if total_hp >= req:
                return selected, total_cost

        # Greedy fell short — exhaustive search up to H_max.
        for size_ in range(1, H_max + 1):
            for combo in combinations(avail, size_):
                if sum(hp for _, hp, _ in combo) >= req:
                    pairs = [[c[0], t] for c in combo]
                    cost  = sum(c[2] * dur for c in combo)
                    return pairs, cost
        return None, 0

    def find_feasible_assignment(vessel_id: int, solution: dict) -> dict | None:
        """Find ONE feasible (berth, t_b, in_tugs, out_tugs) for `vessel_id`
        against `solution`. Searches t_in candidates near ETA (proximity
        order). Uses tight scheduling: berth starts right after inbound ends,
        outbound starts right after berth ends (zero slack; well within ε_time).
        Returns dict or None if nothing feasible.
        """
        i = vessel_id
        sz_i = sizes[i]
        eta_i = etas[i]
        D_i = durations[i]
        tau_in_i = tau_ins[i]
        tau_out_i = tau_outs[i]
        early = earlies[i]
        late = lates[i]

        earliest_in = max(0, eta_i - early)
        latest_in_for_fit = T - tau_in_i - D_i - tau_out_i  # tight schedule
        latest_in = min(latest_in_for_fit, eta_i + late)
        if earliest_in > latest_in:
            return None

        compatibles = _compat_cache[i]
        if not compatibles:
            return None

        # Candidate t_in order: ETA, then ±1, ±2, ... clamped to [earliest, latest]
        candidates = []
        if earliest_in <= eta_i <= latest_in:
            candidates.append(eta_i)
        for offset in range(1, max(early, late) + 1):
            for delta in (-offset, offset):
                t_cand = eta_i + delta
                if earliest_in <= t_cand <= latest_in:
                    candidates.append(t_cand)

        for t_in in candidates:
            t_b = t_in + tau_in_i
            t_out = t_b + D_i
            # Already filtered by latest_in_for_fit, but defensive:
            if t_out + tau_out_i > T:
                continue

            in_tugs, _ = find_tugboat_combination(i, t_in, "inbound", solution)
            if in_tugs is None:
                continue
            out_tugs, _ = find_tugboat_combination(i, t_out, "outbound", solution)
            if out_tugs is None:
                continue

            # Same-tug-reuse safety (if both services share a tug, prep must fit)
            in_ids = {k_ for k_, _ in in_tugs}
            out_ids = {k_ for k_, _ in out_tugs}
            if in_ids & out_ids:
                if t_in + tau_in_i + rho_in > t_out:
                    continue

            for j in compatibles:
                if is_berth_available(j, t_b, i, solution):
                    return {
                        "berth_id":      j,
                        "berth_start":   t_b,
                        "inbound_tugs":  in_tugs,
                        "outbound_tugs": out_tugs,
                    }
        return None

    def assignment_cost(vessel_id: int, berth_id: int, t_b: int,
                        in_tugs: list, out_tugs: list) -> float:
        """Per-vessel Z₂+Z₃+Z₄ estimate (NOT weighted by λ). Use to compare
        candidate assignments against the Z₁ = M·αᵢ skip cost. Returns inf if
        in_tugs / out_tugs are empty (invalid).
        """
        if not in_tugs or not out_tugs:
            return float("inf")
        t_in  = in_tugs[0][1]
        t_out = out_tugs[0][1]
        tau_in_i  = tau_ins[vessel_id]
        tau_out_i = tau_outs[vessel_id]

        port_time = (t_out + tau_out_i) - t_in
        z2 = alpha[vessel_id] * beta[vessel_id] * port_time
        z3 = alpha[vessel_id] * gamma[vessel_id] * abs(t_in - etas[vessel_id])
        z4 = sum(tug_costs[k_] * tau_in_i for k_, _ in in_tugs) \
           + sum(tug_costs[k_] * tau_out_i for k_, _ in out_tugs)
        return float(z2 + z3 + z4)

    # ==================================================================
    # (4) Mutation
    # ==================================================================
    def apply_assignment(solution: dict, vessel_id: int, berth_id: int,
                         t_b: int, in_tugs: list, out_tugs: list) -> dict:
        """Return a NEW solution dict with `vessel_id` assigned. Pure
        function — does NOT modify `solution`. The new dict still has all
        n vessel_ids as keys (strict shape).
        """
        va = dict(solution["vessel_assignments"])
        ib = dict(solution["inbound_tugboats"])
        ob = dict(solution["outbound_tugboats"])
        va[vessel_id] = [int(berth_id), int(t_b)]
        ib[vessel_id] = [[int(k_), int(t_)] for k_, t_ in in_tugs]
        ob[vessel_id] = [[int(k_), int(t_)] for k_, t_ in out_tugs]
        return {
            "vessel_assignments": va,
            "inbound_tugboats":   ib,
            "outbound_tugboats":  ob,
        }

    return {
        "compatible_berths":         compatible_berths,
        "vessel_time_window":        vessel_time_window,
        "unassigned_vessels":        unassigned_vessels,
        "is_berth_available":        is_berth_available,
        "is_tugboat_available":      is_tugboat_available,
        "find_tugboat_combination":  find_tugboat_combination,
        "find_feasible_assignment":  find_feasible_assignment,
        "assignment_cost":           assignment_cost,
        "apply_assignment":          apply_assignment,
    }


EXTRA_TOOLS_DESCRIPTION = [
    # ─── Queries ───────────────────────────────────────────────────────
    {
        "name": "compatible_berths",
        "input": "vessel_id: int",
        "output": "list[int]",
        "purpose": (
            "Berths j with capacity Cⱼ ≥ Sᵢ (C2). Precomputed once per "
            "instance — O(1) lookup. Use to short-circuit the berth loop."
        ),
    },
    {
        "name": "vessel_time_window",
        "input": "vessel_id: int",
        "output": "(int, int)",
        "purpose": (
            "Returns (earliest_t_in, latest_t_in) per C13, already clamped "
            "to [0, T). Inbound services for vessel i MUST start at some "
            "t_in within this interval."
        ),
    },
    {
        "name": "unassigned_vessels",
        "input": "solution: dict",
        "output": "list[int]",
        "purpose": (
            "Vessel ids i where solution['vessel_assignments'][i] is None. "
            "Useful as the outer loop of a construction heuristic."
        ),
    },
    # ─── Feasibility primitives ────────────────────────────────────────
    {
        "name": "is_berth_available",
        "input": "berth_id: int, t_b: int, vessel_id: int, solution: dict",
        "output": "bool",
        "purpose": (
            "Would placing `vessel_id` at (berth_id, t_b) violate C11 against "
            "the other vessels already in `solution`? Excludes vessel_id's own "
            "current slot so you can move it without self-conflict."
        ),
    },
    {
        "name": "is_tugboat_available",
        "input": "tug_id: int, t: int, kind: str, vessel_id: int, solution: dict",
        "output": "bool",
        "purpose": (
            "Would using `tug_id` for a `kind` ∈ {'inbound', 'outbound'} "
            "service of `vessel_id` starting at t violate C12 (including prep "
            "time) against the other vessels in `solution`? Excludes the "
            "vessel's own current slot."
        ),
    },
    # ─── Construction ──────────────────────────────────────────────────
    {
        "name": "find_tugboat_combination",
        "input": "vessel_id: int, t: int, kind: str, solution: dict",
        "output": "(list[[int, int]] | None, float)",
        "purpose": (
            "Find a set of ≤ H_max tugs whose HP sum ≥ vessel's requirement "
            "(C5/C7 or C6/C8), ALL available at time t (via "
            "is_tugboat_available), all starting at t (C9/C10 satisfied by "
            "construction). Greedy by HP descending; falls back to exhaustive "
            "combinations if greedy fails. Returns (tugs_list, cost_estimate) "
            "or (None, 0) if no combination works."
        ),
    },
    {
        "name": "find_feasible_assignment",
        "input": "vessel_id: int, solution: dict",
        "output": "dict | None",
        "purpose": (
            "Find ONE feasible complete assignment for vessel_id against the "
            "current `solution`. Searches t_in candidates in proximity-to-ETA "
            "order, uses tight scheduling (t_b = t_in + τⁱⁿ, t_out = t_b + D — "
            "zero slack, well within ε_time). Returns dict with keys "
            "{berth_id, berth_start, inbound_tugs, outbound_tugs} or None if "
            "no feasible triple exists. Pass directly into apply_assignment."
        ),
    },
    {
        "name": "assignment_cost",
        "input": "vessel_id: int, berth_id: int, t_b: int, in_tugs: list, out_tugs: list",
        "output": "float",
        "purpose": (
            "Per-vessel Z₂+Z₃+Z₄ estimate (NOT λ-weighted). Compare against "
            "M·αᵢ to decide if serving the vessel beats skipping it."
        ),
    },
    # ─── Mutation ──────────────────────────────────────────────────────
    {
        "name": "apply_assignment",
        "input": "solution: dict, vessel_id: int, berth_id: int, t_b: int, in_tugs: list, out_tugs: list",
        "output": "dict",
        "purpose": (
            "Return a NEW solution dict with vessel_id assigned. Pure "
            "function — does NOT mutate `solution`. The returned dict still "
            "has all n vessel_ids as keys (strict-shape contract)."
        ),
    },
]
