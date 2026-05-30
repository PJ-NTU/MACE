"""Step-by-step is_feasible reference for Port Scheduling Problem.

Read by spec.py and prepended to feasibility_doc. Gives the LLM a compact,
clearly-labeled view of every constraint (without the cost computation), in
the TSP-style early-return pattern. Empirically (see OVERNIGHT_LOG.md) this
raises first-shot LLM bootstrap success notably over relying on eval_func
source alone — the constraint codes (C2..C15) make it scannable.

All `is_feasible` references below assume the instance kwargs are in scope
(vessel_num, berth_capacities, vessel_sizes, ... — see solve() docstring).
"""

FEASIBILITY_STEPS_PY = r'''def is_feasible(solution):
    # ────── Top-level shape ──────────────────────────────────────────────
    if not isinstance(solution, dict):
        return False, f"solution must be dict, got {type(solution).__name__}"
    for key in ("vessel_assignments", "inbound_tugboats", "outbound_tugboats"):
        if key not in solution:
            return False, f"solution missing '{key}' key"

    va = solution["vessel_assignments"]
    ib = solution["inbound_tugboats"]
    ob = solution["outbound_tugboats"]
    n = vessel_num
    J = berth_num
    K = tugboat_num
    T = time_periods
    H_max = max_tugboats_per_service
    eps_time = time_constraint_tolerance

    expected_keys = set(range(n))
    if not isinstance(va, dict) or set(va.keys()) != expected_keys:
        return False, f"vessel_assignments must be dict with keys exactly {{0..{n-1}}}"
    if not isinstance(ib, dict) or set(ib.keys()) != expected_keys:
        return False, f"inbound_tugboats must be dict with keys exactly {{0..{n-1}}}"
    if not isinstance(ob, dict) or set(ob.keys()) != expected_keys:
        return False, f"outbound_tugboats must be dict with keys exactly {{0..{n-1}}}"

    # ────── Per-vessel checks ────────────────────────────────────────────
    for i in range(n):
        assignment = va[i]
        in_tugs    = ib[i]
        out_tugs   = ob[i]

        # Unassigned: no services allowed (C3/C4 contrapositive)
        if assignment is None:
            if in_tugs:
                return False, f"vessel {i} unassigned but inbound_tugboats[{i}]={in_tugs}"
            if out_tugs:
                return False, f"vessel {i} unassigned but outbound_tugboats[{i}]={out_tugs}"
            continue

        # Assigned: validate (berth_id, t_b) shape
        if not (isinstance(assignment, list) and len(assignment) == 2):
            return False, f"vessel_assignments[{i}] must be None or [berth_id, t_b], got {assignment!r}"
        berth_id, t_b = assignment
        if not isinstance(berth_id, int) or not (0 <= berth_id < J):
            return False, f"vessel {i} berth_id={berth_id} not in [0, {J})"
        if not isinstance(t_b, int) or not (0 <= t_b < T):
            return False, f"vessel {i} berth_start_time={t_b} not in [0, {T})"

        # (C2) vessel size ≤ berth capacity
        if berth_capacities[berth_id] < vessel_sizes[i]:
            return False, (f"vessel {i} size={vessel_sizes[i]} > berth {berth_id} "
                           f"capacity={berth_capacities[berth_id]} (C2)")

        # (C3/C4) assigned ⇒ both tug services exist
        if not (isinstance(in_tugs, list) and len(in_tugs) >= 1):
            return False, f"vessel {i} assigned but no inbound tug service (C3)"
        if not (isinstance(out_tugs, list) and len(out_tugs) >= 1):
            return False, f"vessel {i} assigned but no outbound tug service (C4)"

        # (C7/C8) ≤ H_max tugs per service
        if len(in_tugs) > H_max:
            return False, f"vessel {i} uses {len(in_tugs)} inbound tugs > H_max={H_max} (C7)"
        if len(out_tugs) > H_max:
            return False, f"vessel {i} uses {len(out_tugs)} outbound tugs > H_max={H_max} (C8)"

        # Tug pair shape + index range
        for pairs, kind in ((in_tugs, "inbound"), (out_tugs, "outbound")):
            for p in pairs:
                if not (isinstance(p, list) and len(p) == 2):
                    return False, f"vessel {i} {kind}_tugboats entry must be [tug_id, t], got {p!r}"
                k_, t_ = p
                if not isinstance(k_, int) or not (0 <= k_ < K):
                    return False, f"vessel {i} {kind} tug_id={k_} not in [0, {K})"
                if not isinstance(t_, int) or not (0 <= t_ < T):
                    return False, f"vessel {i} {kind} t={t_} not in [0, {T})"

        # (C9/C10) all tugs in one service share start time
        in_starts = {t_ for _, t_ in in_tugs}
        if len(in_starts) > 1:
            return False, f"vessel {i} inbound tugs have inconsistent starts {in_starts} (C9)"
        out_starts = {t_ for _, t_ in out_tugs}
        if len(out_starts) > 1:
            return False, f"vessel {i} outbound tugs have inconsistent starts {out_starts} (C10)"
        t_in  = in_tugs[0][1]
        t_out = out_tugs[0][1]

        # (C5) inbound HP sum ≥ requirement
        hp_in = sum(tugboat_horsepower[k_] for k_, _ in in_tugs)
        if hp_in < vessel_horsepower_requirements[i]:
            return False, (f"vessel {i} inbound HP={hp_in} < req "
                           f"{vessel_horsepower_requirements[i]} (C5)")
        # (C6) outbound HP sum ≥ requirement
        hp_out = sum(tugboat_horsepower[k_] for k_, _ in out_tugs)
        if hp_out < vessel_horsepower_requirements[i]:
            return False, (f"vessel {i} outbound HP={hp_out} < req "
                           f"{vessel_horsepower_requirements[i]} (C6)")

        # (C13) inbound time window
        eta = vessel_etas[i]
        early = vessel_early_limits[i]
        late = vessel_late_limits[i]
        if not (eta - early <= t_in <= eta + late):
            return False, (f"vessel {i} t_in={t_in} outside ETA window "
                           f"[{eta - early}, {eta + late}] (C13)")

        # (C14) berth_start in [inbound_end, inbound_end + ε_time]
        tau_in = vessel_inbound_service_times[i]
        gap_b = t_b - (t_in + tau_in)
        if gap_b < 0:
            return False, f"vessel {i} berth_start={t_b} < inbound_end={t_in + tau_in} (C14)"
        if gap_b > eps_time:
            return False, f"vessel {i} berth_start - inbound_end={gap_b} > ε_time={eps_time} (C14)"

        # (C15) outbound_start in [berth_end, berth_end + ε_time]
        D_i = vessel_durations[i]
        gap_o = t_out - (t_b + D_i)
        if gap_o < 0:
            return False, f"vessel {i} outbound_start={t_out} < berth_end={t_b + D_i} (C15)"
        if gap_o > eps_time:
            return False, f"vessel {i} outbound_start - berth_end={gap_o} > ε_time={eps_time} (C15)"

        # Outbound must complete before T
        tau_out = vessel_outbound_service_times[i]
        if t_out + tau_out > T:
            return False, f"vessel {i} outbound_end={t_out + tau_out} exceeds T={T}"

    # ────── Cross-vessel resource non-overlap ─────────────────────────────

    # (C11) berth non-overlap across vessels
    berth_occ = {j: [] for j in range(J)}
    for i in range(n):
        if va[i] is None:
            continue
        bj, t_b = va[i]
        berth_occ[bj].append((t_b, t_b + vessel_durations[i], i))
    for bj, occs in berth_occ.items():
        occs.sort()
        for u in range(len(occs) - 1):
            if occs[u][1] > occs[u + 1][0]:
                return False, (f"berth {bj} conflict: vessel {occs[u][2]} ends at "
                               f"{occs[u][1]} but vessel {occs[u + 1][2]} starts at "
                               f"{occs[u + 1][0]} (C11)")

    # (C12) tugboat non-overlap including prep time
    rho_in  = inbound_preparation_time
    rho_out = outbound_preparation_time
    tug_occ = {k_: [] for k_ in range(K)}
    for i in range(n):
        if va[i] is None:
            continue
        tau_in  = vessel_inbound_service_times[i]
        tau_out = vessel_outbound_service_times[i]
        for k_, t_ in ib[i]:
            tug_occ[k_].append((t_, t_ + tau_in + rho_in, i, "in"))
        for k_, t_ in ob[i]:
            tug_occ[k_].append((t_, t_ + tau_out + rho_out, i, "out"))
    for k_, occs in tug_occ.items():
        occs.sort()
        for u in range(len(occs) - 1):
            if occs[u][1] > occs[u + 1][0]:
                return False, (f"tugboat {k_} conflict: vessel {occs[u][2]} "
                               f"({occs[u][3]}) busy until {occs[u][1]} but vessel "
                               f"{occs[u + 1][2]} ({occs[u + 1][3]}) starts at "
                               f"{occs[u + 1][0]} (C12)")

    return True, None
'''
