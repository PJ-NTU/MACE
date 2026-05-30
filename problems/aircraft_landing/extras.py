"""Per-problem extras for CO-Bench Aircraft Landing Scheduling Problem (ALSP).

Provides building blocks so the LLM can compose construction + local-search +
exact methods for the Aircraft Landing Problem (Cluster A: scheduling with
time windows + pairwise separation on multiple runways, earliness/lateness
penalties).

Tool groups:
  (1) Queries:         n_planes, n_runways, plane_window, plane_penalty_early,
                       plane_penalty_late, separation
  (2) Feasibility:     validate_schedule, penalty_of_schedule, runway_load
  (3) Construction /
      improvement:     greedy_target_time_construct,
                       construct_by_appearance_order,
                       apply_swap_landings,
                       apply_reassign_runway
  (4) Heavy (MILP):    ilp_aircraft_landing

A schedule is the same dict CO-Bench's eval_func expects:
    {plane_id (1-indexed): {"landing_time": float, "runway": int}, ...}
"""
from __future__ import annotations
import time
from typing import Optional


def extra_tools(instance: dict) -> dict:
    """Factory: returns ALSP-specific tool callables given the loaded instance.

    Instance schema (from CO-Bench Aircraft landing load_data, one case):
      - num_planes:  int
      - num_runways: int
      - freeze_time: float (unused for scheduling decisions)
      - planes:      list[dict] with keys
                       appearance, earliest, target, latest,
                       penalty_early, penalty_late
      - separation:  list[list[float]] -- separation[i][j] is the required
                     gap after plane i lands before plane j can land
                     when they share a runway. (0-indexed.)
    """
    N = int(instance["num_planes"])
    R = int(instance["num_runways"])
    planes = instance["planes"]
    sep = instance["separation"]

    # Cached per-plane fields (1-indexed access for the public API).
    earliest = [float(p["earliest"]) for p in planes]
    target = [float(p["target"]) for p in planes]
    latest = [float(p["latest"]) for p in planes]
    pe = [float(p["penalty_early"]) for p in planes]
    pl = [float(p["penalty_late"]) for p in planes]
    appearance = [float(p["appearance"]) for p in planes]

    EPS = 1e-9

    def _check_id(i: int) -> int:
        ii = int(i)
        if not (1 <= ii <= N):
            raise ValueError(f"plane id {i} out of range [1, {N}]")
        return ii

    # ==================================================================
    # (1) Queries
    # ==================================================================
    def n_planes() -> int:
        return N

    def n_runways() -> int:
        return R

    def plane_window(i: int) -> tuple:
        ii = _check_id(i)
        return (earliest[ii - 1], target[ii - 1], latest[ii - 1])

    def plane_penalty_early(i: int) -> float:
        ii = _check_id(i)
        return pe[ii - 1]

    def plane_penalty_late(i: int) -> float:
        ii = _check_id(i)
        return pl[ii - 1]

    def separation(i: int, j: int) -> float:
        ii = _check_id(i)
        jj = _check_id(j)
        return float(sep[ii - 1][jj - 1])

    # ==================================================================
    # (2) Feasibility primitives
    # ==================================================================
    def _normalize_entry(entry):
        if not isinstance(entry, dict):
            return None, None
        if "landing_time" not in entry or "runway" not in entry:
            return None, None
        try:
            t = float(entry["landing_time"])
            r = int(entry["runway"])
        except Exception:
            return None, None
        return t, r

    def validate_schedule(schedule: dict) -> tuple:
        """Return (True, None) if `schedule` satisfies all constraints
        (window + runway range + pairwise separation on same runway),
        else (False, error_message). Mirrors CO-Bench eval_func's checks."""
        if not isinstance(schedule, dict):
            return False, f"schedule must be dict, got {type(schedule).__name__}"
        if len(schedule) != N:
            return False, f"schedule must have exactly {N} entries, got {len(schedule)}"
        for pid in range(1, N + 1):
            if pid not in schedule:
                return False, f"plane {pid} missing from schedule"
            t, r = _normalize_entry(schedule[pid])
            if t is None:
                return False, f"plane {pid} entry malformed: {schedule[pid]!r}"
            if not (1 <= r <= R):
                return False, f"plane {pid} runway {r} not in [1, {R}]"
            if t < earliest[pid - 1] - EPS or t > latest[pid - 1] + EPS:
                return (False,
                        f"plane {pid} time {t} outside window "
                        f"[{earliest[pid - 1]}, {latest[pid - 1]}]")
        # Pairwise separation on shared runway.
        for i in range(1, N + 1):
            ti, ri = _normalize_entry(schedule[i])
            for j in range(1, N + 1):
                if i == j:
                    continue
                tj, rj = _normalize_entry(schedule[j])
                if ri != rj:
                    continue
                if ti <= tj:
                    need = float(sep[i - 1][j - 1])
                    if (tj - ti) < need - EPS:
                        return (False,
                                f"separation violation runway {ri}: plane {i}@{ti} -> "
                                f"plane {j}@{tj} (need >= {need})")
        return True, None

    def penalty_of_schedule(schedule: dict) -> float:
        """Total earliness+lateness penalty for a schedule.

        NOTE: does NOT check feasibility -- combine with validate_schedule if
        you need both. Returns float (>= 0)."""
        total = 0.0
        for pid in range(1, N + 1):
            if pid not in schedule:
                continue
            t, _ = _normalize_entry(schedule[pid])
            if t is None:
                continue
            tgt = target[pid - 1]
            if t < tgt:
                total += (tgt - t) * pe[pid - 1]
            elif t > tgt:
                total += (t - tgt) * pl[pid - 1]
        return total

    def runway_load(r: int, schedule: dict) -> list:
        """List of (plane_id, landing_time) on runway r, sorted by landing_time."""
        rr = int(r)
        if not (1 <= rr <= R):
            raise ValueError(f"runway {r} not in [1, {R}]")
        out = []
        for pid in range(1, N + 1):
            if pid not in schedule:
                continue
            t, ri = _normalize_entry(schedule[pid])
            if t is not None and ri == rr:
                out.append((pid, t))
        out.sort(key=lambda x: x[1])
        return out

    # ==================================================================
    # (3) Construction / improvement
    # ==================================================================
    def _round_robin_runway(plane_idx: int) -> int:
        # 1-indexed runway in [1, R]
        return (plane_idx % R) + 1

    def greedy_target_time_construct() -> dict:
        """Greedy construction. Sort planes by target time, place each at target;
        if a separation violation against an already-placed plane on the same
        runway is found, push the new plane to the earliest feasible time
        (still within its [earliest, latest] window). Try each runway, pick the
        runway giving the smallest induced penalty. If no runway is feasible at
        all (i.e., even pushing past latest), drop runway choice that needs
        smallest violation; in that case the returned schedule may still be
        infeasible -- caller should validate."""
        # planes sorted by target time, keeping 1-indexed ids
        order = sorted(range(1, N + 1), key=lambda i: target[i - 1])
        # per-runway sorted list of (landing_time, plane_id)
        runways: list[list[tuple]] = [[] for _ in range(R)]
        schedule: dict = {}

        for pid in order:
            tgt = target[pid - 1]
            elo = earliest[pid - 1]
            ehi = latest[pid - 1]
            best_choice = None  # (penalty_delta, runway, time)
            for r_idx in range(R):
                # earliest start considering separations from planes already on r_idx
                t_cand = max(tgt, elo)
                # repeatedly push if separation broken
                changed = True
                while changed:
                    changed = False
                    for (other_t, other_pid) in runways[r_idx]:
                        if other_t <= t_cand:
                            need = sep[other_pid - 1][pid - 1]
                            if t_cand - other_t < need - EPS:
                                t_cand = other_t + need
                                changed = True
                        else:
                            # `pid` lands before `other_t`: separation pid->other
                            need = sep[pid - 1][other_pid - 1]
                            if other_t - t_cand < need - EPS:
                                # conflict: we'd violate constraint after `other`
                                # push past other_t
                                # but we also need to satisfy other_pid->pid sep
                                need2 = sep[other_pid - 1][pid - 1]
                                t_cand = max(t_cand, other_t + need2)
                                changed = True
                if t_cand > ehi + EPS:
                    continue  # infeasible on this runway
                if t_cand < elo - EPS:
                    t_cand = elo
                pen = ((tgt - t_cand) * pe[pid - 1] if t_cand < tgt
                       else (t_cand - tgt) * pl[pid - 1])
                if best_choice is None or pen < best_choice[0]:
                    best_choice = (pen, r_idx + 1, t_cand)
            if best_choice is None:
                # last resort: place at target on the round-robin runway
                # (may be infeasible -- caller must validate)
                r_pick = _round_robin_runway(pid - 1)
                t_pick = max(elo, min(ehi, tgt))
                schedule[pid] = {"landing_time": float(t_pick), "runway": int(r_pick)}
                runways[r_pick - 1].append((float(t_pick), pid))
                runways[r_pick - 1].sort()
            else:
                _, r_pick, t_pick = best_choice
                schedule[pid] = {"landing_time": float(t_pick), "runway": int(r_pick)}
                runways[r_pick - 1].append((float(t_pick), pid))
                runways[r_pick - 1].sort()
        return schedule

    def construct_by_appearance_order() -> dict:
        """Simpler construction: take planes in order of appearance, assign each
        to the runway whose latest current landing time leaves the most room.
        Each plane lands at max(target, earliest feasible time on chosen runway),
        clipped into [earliest, latest]. May be infeasible if windows are
        tight -- caller must validate."""
        order = sorted(range(1, N + 1), key=lambda i: appearance[i - 1])
        runways: list[list[tuple]] = [[] for _ in range(R)]
        schedule: dict = {}
        for pid in order:
            elo = earliest[pid - 1]
            ehi = latest[pid - 1]
            tgt = target[pid - 1]
            best = None  # (time, runway)
            for r_idx in range(R):
                t_cand = max(tgt, elo)
                changed = True
                while changed:
                    changed = False
                    for (other_t, other_pid) in runways[r_idx]:
                        if other_t <= t_cand:
                            need = sep[other_pid - 1][pid - 1]
                            if t_cand - other_t < need - EPS:
                                t_cand = other_t + need
                                changed = True
                        else:
                            need2 = sep[other_pid - 1][pid - 1]
                            t_cand = max(t_cand, other_t + need2)
                            changed = True
                if t_cand > ehi + EPS:
                    continue
                if t_cand < elo:
                    t_cand = elo
                pen = ((tgt - t_cand) * pe[pid - 1] if t_cand < tgt
                       else (t_cand - tgt) * pl[pid - 1])
                key = (pen, t_cand)
                if best is None or key < best[0]:
                    best = (key, r_idx + 1, t_cand)
            if best is None:
                r_pick = _round_robin_runway(pid - 1)
                t_pick = max(elo, min(ehi, tgt))
                schedule[pid] = {"landing_time": float(t_pick), "runway": int(r_pick)}
                runways[r_pick - 1].append((float(t_pick), pid))
                runways[r_pick - 1].sort()
            else:
                _, r_pick, t_pick = best
                schedule[pid] = {"landing_time": float(t_pick), "runway": int(r_pick)}
                runways[r_pick - 1].append((float(t_pick), pid))
                runways[r_pick - 1].sort()
        return schedule

    def _resolve_runway_times(schedule: dict) -> Optional[dict]:
        """Given runway assignments (read from schedule), recompute earliest
        feasible landing times within windows using sequential push:
        on each runway, sort by current landing_time, then push each plane up
        to max(earliest, prev_landing + sep[prev][cur]). Returns a new schedule
        if all windows respected, else None."""
        new_sched: dict = {}
        for r in range(1, R + 1):
            seq = sorted(
                ((pid, schedule[pid]) for pid in schedule
                 if schedule[pid]["runway"] == r),
                key=lambda kv: kv[1]["landing_time"],
            )
            prev_t = None
            prev_pid = None
            for pid, entry in seq:
                t_low = earliest[pid - 1]
                if prev_pid is not None:
                    t_low = max(t_low, prev_t + sep[prev_pid - 1][pid - 1])
                tgt = target[pid - 1]
                # land as close to target as possible without going below t_low
                t_pick = max(t_low, min(latest[pid - 1], tgt))
                if t_pick < t_low - EPS:
                    t_pick = t_low
                if t_pick > latest[pid - 1] + EPS:
                    return None
                new_sched[pid] = {"landing_time": float(t_pick), "runway": int(r)}
                prev_pid, prev_t = pid, t_pick
        if len(new_sched) != N:
            return None
        return new_sched

    def apply_swap_landings(schedule: dict, t_limit: float = 2.0) -> dict:
        """Local search: try swapping each pair of (plane_a, plane_b) on the
        same runway (swap their order in the runway sequence) and on different
        runways (swap their runway assignments). Recompute optimal landing
        times via _resolve_runway_times. Keep the swap if it lowers total
        penalty. First-improvement, restarts when one improvement is found.
        Returns the improved schedule (does not mutate input)."""
        cur = {k: dict(v) for k, v in schedule.items()}
        ok, _ = validate_schedule(cur)
        if not ok:
            # try resolving first
            resolved = _resolve_runway_times(cur)
            if resolved is not None:
                cur = resolved
            else:
                return cur
        best_pen = penalty_of_schedule(cur)
        t0 = time.time()
        improved = True
        while improved and (time.time() - t0) < t_limit - 0.02:
            improved = False
            ids = list(range(1, N + 1))
            for a in ids:
                if (time.time() - t0) >= t_limit - 0.02:
                    break
                for b in ids:
                    if b <= a:
                        continue
                    # Try swapping the runway of a and b
                    cand = {k: dict(v) for k, v in cur.items()}
                    ra = cand[a]["runway"]
                    rb = cand[b]["runway"]
                    cand[a]["runway"] = rb
                    cand[b]["runway"] = ra
                    resolved = _resolve_runway_times(cand)
                    if resolved is not None:
                        pen = penalty_of_schedule(resolved)
                        ok2, _ = validate_schedule(resolved)
                        if ok2 and pen < best_pen - 1e-9:
                            cur = resolved
                            best_pen = pen
                            improved = True
                            break
                if improved:
                    break
        return cur

    def apply_reassign_runway(schedule: dict, plane: int, runway: int) -> dict:
        """Reassign `plane` to `runway` and recompute feasible landing times
        for every plane via _resolve_runway_times. Returns the new schedule
        if feasible, else returns the original schedule unchanged."""
        pid = _check_id(plane)
        rr = int(runway)
        if not (1 <= rr <= R):
            raise ValueError(f"runway {runway} not in [1, {R}]")
        cand = {k: dict(v) for k, v in schedule.items()}
        if pid not in cand:
            return schedule
        cand[pid]["runway"] = rr
        resolved = _resolve_runway_times(cand)
        if resolved is None:
            return schedule
        ok, _ = validate_schedule(resolved)
        if not ok:
            return schedule
        return resolved

    # ==================================================================
    # (4) MILP (mixed-integer linear program, exact for small N)
    # ==================================================================
    def ilp_aircraft_landing(time_limit_s: float = 30.0) -> Optional[dict]:
        """Exact MILP for ALSP with continuous landing times + binary runway
        assignment + big-M ordering on each runway. Uses CBC via the `mip`
        package. Returns the best schedule found within time_limit_s
        (could be optimal), or None if no feasible solution found or `mip`
        is unavailable.

        Note: scales reasonably up to ~30-50 planes; beyond that it becomes
        impractical. For larger instances, prefer the greedy +
        apply_swap_landings combination."""
        try:
            from mip import (Model, BINARY, CONTINUOUS, MINIMIZE, xsum,
                             OptimizationStatus)
        except Exception:
            return None

        m = Model(sense=MINIMIZE)
        m.verbose = 0
        m.max_seconds = float(time_limit_s)

        ids = list(range(N))  # 0-indexed internally
        # Continuous: landing time of plane i.
        t = [m.add_var(name=f"t[{i}]", var_type=CONTINUOUS,
                       lb=earliest[i], ub=latest[i]) for i in ids]
        # Earliness alpha, lateness beta (continuous, >= 0).
        alpha = [m.add_var(name=f"a[{i}]", var_type=CONTINUOUS, lb=0.0,
                           ub=max(0.0, target[i] - earliest[i])) for i in ids]
        beta = [m.add_var(name=f"b[{i}]", var_type=CONTINUOUS, lb=0.0,
                          ub=max(0.0, latest[i] - target[i])) for i in ids]
        # Runway assignment: y[i][r] = 1 if plane i on runway r.
        y = [[m.add_var(name=f"y[{i},{r}]", var_type=BINARY)
              for r in range(R)] for i in ids]
        # Same-runway indicator: z[i][j] = 1 if i, j share a runway (i < j).
        z = {}
        for i in ids:
            for j in ids:
                if i < j:
                    z[(i, j)] = m.add_var(name=f"z[{i},{j}]", var_type=BINARY)
        # Ordering: delta[i][j] = 1 if plane i lands before plane j (i != j).
        delta = {}
        for i in ids:
            for j in ids:
                if i != j:
                    delta[(i, j)] = m.add_var(name=f"d[{i},{j}]", var_type=BINARY)

        # Each plane on exactly one runway.
        for i in ids:
            m += xsum(y[i][r] for r in range(R)) == 1, f"one_rwy_{i}"

        # Earliness/lateness linkage: t[i] = target - alpha + beta
        for i in ids:
            m += t[i] - target[i] + alpha[i] - beta[i] == 0, f"link_{i}"

        # Ordering: delta[i,j] + delta[j,i] = 1 for i < j
        for i in ids:
            for j in ids:
                if i < j:
                    m += delta[(i, j)] + delta[(j, i)] == 1, f"ord_{i}_{j}"

        # Same-runway linking with z:
        #   z[i,j] <= y[i][r] + y[j][r]'s ... using sum of products:
        # z[i,j] = sum_r y[i][r]*y[j][r] -- linearize:
        # We use sum_r w[i,j,r] = z[i,j], with w[i,j,r] <= y[i][r],
        # w[i,j,r] <= y[j][r], w[i,j,r] >= y[i][r] + y[j][r] - 1.
        w = {}
        for i in ids:
            for j in ids:
                if i < j:
                    for r in range(R):
                        w[(i, j, r)] = m.add_var(name=f"w[{i},{j},{r}]",
                                                 var_type=BINARY)
                        m += w[(i, j, r)] <= y[i][r], f"w_le_yi_{i}_{j}_{r}"
                        m += w[(i, j, r)] <= y[j][r], f"w_le_yj_{i}_{j}_{r}"
                        m += w[(i, j, r)] >= y[i][r] + y[j][r] - 1, \
                            f"w_ge_{i}_{j}_{r}"
                    m += z[(i, j)] == xsum(w[(i, j, r)] for r in range(R)), \
                        f"z_def_{i}_{j}"

        # Separation big-M:
        # If z[i,j] = 1 (same runway) and delta[i,j] = 1 (i before j):
        #   t[j] >= t[i] + sep[i][j]
        # Big-M: t[j] - t[i] >= sep[i][j] - M*(1 - z[i,j]) - M*(1 - delta[i,j])
        # We pick M large enough.
        for i in ids:
            for j in ids:
                if i == j:
                    continue
                ii_lo, ii_hi = earliest[i], latest[i]
                jj_lo, jj_hi = earliest[j], latest[j]
                # worst-case: t[j] = jj_lo, t[i] = ii_hi, separation positive
                # Choose M = (ii_hi - jj_lo) + sep[i][j] + 1
                bigM = abs(ii_hi - jj_lo) + abs(sep[i][j]) + 1.0
                # Map (i, j) to z key (sorted)
                ik, jk = (i, j) if i < j else (j, i)
                m += (t[j] - t[i] >= sep[i][j]
                      - bigM * (1 - z[(ik, jk)])
                      - bigM * (1 - delta[(i, j)])), f"sep_{i}_{j}"

        # Objective.
        m.objective = xsum(pe[i] * alpha[i] + pl[i] * beta[i] for i in ids)

        status = m.optimize()
        if status not in (OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE):
            return None
        if m.num_solutions < 1:
            return None
        sched = {}
        for i in ids:
            ti = float(t[i].x) if t[i].x is not None else float(target[i])
            # CBC sometimes returns landing_time = target + 1e-14 due to
            # floating-point creep; round to 4 decimal places to keep the
            # separation constraint numerically satisfied.
            ti = round(ti, 4)
            r_pick = 1
            for r in range(R):
                if y[i][r].x is not None and y[i][r].x > 0.5:
                    r_pick = r + 1
                    break
            sched[i + 1] = {"landing_time": ti, "runway": int(r_pick)}
        return sched

    return {
        # queries
        "n_planes": n_planes,
        "n_runways": n_runways,
        "plane_window": plane_window,
        "plane_penalty_early": plane_penalty_early,
        "plane_penalty_late": plane_penalty_late,
        "separation": separation,
        # feasibility
        "validate_schedule": validate_schedule,
        "penalty_of_schedule": penalty_of_schedule,
        "runway_load": runway_load,
        # construction / improvement
        "greedy_target_time_construct": greedy_target_time_construct,
        "construct_by_appearance_order": construct_by_appearance_order,
        "apply_swap_landings": apply_swap_landings,
        "apply_reassign_runway": apply_reassign_runway,
        # heavy
        "ilp_aircraft_landing": ilp_aircraft_landing,
    }


EXTRA_TOOLS_DESCRIPTION = [
    # ----- Queries -----
    {
        "name": "n_planes",
        "input": "(no args)",
        "output": "int",
        "purpose": "Number of planes N in the instance (1-indexed externally).",
    },
    {
        "name": "n_runways",
        "input": "(no args)",
        "output": "int",
        "purpose": "Number of available runways R in the instance (runway ids 1..R).",
    },
    {
        "name": "plane_window",
        "input": "i: int (1-indexed)",
        "output": "(earliest: float, target: float, latest: float)",
        "purpose": (
            "Time window for plane i. The landing time must lie in "
            "[earliest, latest]; target is the zero-penalty point."
        ),
    },
    {
        "name": "plane_penalty_early",
        "input": "i: int (1-indexed)",
        "output": "float",
        "purpose": (
            "Per-unit-time earliness penalty for plane i (applies when "
            "landing_time < target)."
        ),
    },
    {
        "name": "plane_penalty_late",
        "input": "i: int (1-indexed)",
        "output": "float",
        "purpose": (
            "Per-unit-time lateness penalty for plane i (applies when "
            "landing_time > target)."
        ),
    },
    {
        "name": "separation",
        "input": "i: int, j: int (both 1-indexed)",
        "output": "float",
        "purpose": (
            "Required minimum gap after plane i lands before plane j can "
            "land, when they share a runway. (Indexed sep[i-1][j-1] in the "
            "raw instance.) Matrix is generally asymmetric."
        ),
    },
    # ----- Feasibility primitives -----
    {
        "name": "validate_schedule",
        "input": "schedule: dict",
        "output": "(bool, str | None)",
        "purpose": (
            "Local re-implementation of CO-Bench's eval_func feasibility "
            "checks (windows + runway range + pairwise separation on shared "
            "runway). Cheaper than tools['is_feasible'] for tight inner "
            "loops; returns the first violation reason if any."
        ),
    },
    {
        "name": "penalty_of_schedule",
        "input": "schedule: dict",
        "output": "float",
        "purpose": (
            "Total earliness+lateness penalty cost. Does NOT verify "
            "feasibility -- combine with validate_schedule when needed."
        ),
    },
    {
        "name": "runway_load",
        "input": "r: int, schedule: dict",
        "output": "list[(plane_id, landing_time)] sorted by landing_time",
        "purpose": (
            "List of all planes assigned to runway r in the schedule, in "
            "landing order. Useful for inspecting/repairing a runway's "
            "sequence."
        ),
    },
    # ----- Construction / improvement -----
    {
        "name": "greedy_target_time_construct",
        "input": "(no args)",
        "output": "dict (schedule)",
        "purpose": (
            "Greedy construction. Sorts planes by target time; for each "
            "plane, tries every runway, places at target (push later to "
            "satisfy separation, but within [earliest, latest]) and keeps "
            "the runway that yields the smallest induced penalty. May "
            "return an infeasible schedule if no runway fits -- caller "
            "MUST validate. Good warm-start; pair with apply_swap_landings."
        ),
    },
    {
        "name": "construct_by_appearance_order",
        "input": "(no args)",
        "output": "dict (schedule)",
        "purpose": (
            "Alternative greedy: schedules planes in order of appearance "
            "time. Same per-runway push-to-feasible logic as the target-"
            "time greedy. Diversification for restarts -- run both "
            "constructors and keep the better feasible result."
        ),
    },
    {
        "name": "apply_swap_landings",
        "input": "schedule: dict, t_limit: float = 2.0",
        "output": "dict (schedule)",
        "purpose": (
            "Local search: iteratively swap the runway assignments of two "
            "planes (i, j), resolve each runway's landing times by tight "
            "sequential packing (each plane lands as close to its target "
            "as separation allows), and keep the swap if it strictly "
            "lowers total penalty. First-improvement, returns improved "
            "schedule (does not mutate input)."
        ),
    },
    {
        "name": "apply_reassign_runway",
        "input": "schedule: dict, plane: int (1-indexed), runway: int (1-indexed)",
        "output": "dict (schedule)",
        "purpose": (
            "Move `plane` to `runway` and re-pack every runway's landing "
            "times (each plane lands as close to its target as the "
            "separation constraints allow, within its window). Returns the "
            "new feasible schedule, or the input unchanged if the move "
            "would force any plane out of its window."
        ),
    },
    # ----- Heavy / MILP -----
    {
        "name": "ilp_aircraft_landing",
        "input": "time_limit_s: float = 30.0",
        "output": "dict (schedule) | None",
        "purpose": (
            "Exact MILP via CBC (open-source, through the `mip` package): "
            "continuous landing times + binary runway assignment + binary "
            "ordering + big-M separation constraints. Returns the best "
            "schedule found within the time limit (may be optimal), or None "
            "if no feasible solution / CBC unavailable. Practical up to "
            "~30-50 planes; beyond that, prefer greedy + apply_swap_landings."
        ),
    },
]
