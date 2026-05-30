"""Per-problem extras for CO-Bench Period Vehicle Routing Problem (PVRP).

PVRP is a two-layer problem:
  (i) Schedule choice: for each customer, pick exactly one candidate schedule
      -- a binary vector of length period_length saying on which days that
      customer must be visited.
  (ii) Daily CVRP: for every day d, the customers whose chosen schedule has
       a 1 in position d-1 must be partitioned into vehicle tours from the
       depot, each tour respecting vehicle_capacity, and at most
       vehicles_per_day[d-1] tours are allowed.

The objective is total Euclidean distance over all daily tours.

These extras give the LLM building blocks across four tiers so it doesn't
have to reimplement nearest-neighbor / 2-opt / capacity-bin-packing / a
schedule-assignment ILP from scratch.

Tool groups:
  (1) Queries:        customer_demand, customer_visit_count,
                      customer_allowed_periods, customer_candidate_schedules,
                      vehicle_capacity, num_periods, distance, customer_ids
  (2) Feasibility:    period_required_customers, period_demand,
                      period_routes_valid, unassigned_visits
  (3) Construction:   nn_route, apply_2opt_route, assign_schedules_greedy,
                      solve_period_routing
  (4) Heavy:          ilp_schedule_assignment

The LLM may use any subset, or write everything from scratch.

Coordinate conventions
----------------------
"schedule" here always means a `dict[customer_id -> binary_vector]` -- the
same shape as the solution field `selected_schedules`. A "period" is a
1-indexed day in [1, period_length]. A "tour" is a list of vertex ids that
starts and ends at depot id 0, with no intermediate depot.
"""
from __future__ import annotations
import math
import random
import time
from typing import Iterable, Optional

try:
    from mip import Model, BINARY, MINIMIZE, xsum, OptimizationStatus
    _HAS_MIP = True
except ImportError:  # pragma: no cover - fallback if mip not installed
    _HAS_MIP = False


def extra_tools(instance: dict) -> dict:
    """Factory: returns PVRP-specific tool callables given the loaded instance.

    Instance schema (from CO-Bench Vehicle routing_ period routing load_data):
      - period_length: int
      - vehicles_per_day: list[int] (len == period_length)
      - vehicle_capacity: float
      - depot:     {"id": 0, "x": float, "y": float}
      - customers: list[{"id": int, "x": float, "y": float,
                         "demand": float,
                         "schedules": list[list[int]]}]
    """
    depot = instance["depot"]
    customers = instance["customers"]
    period_length = int(instance["period_length"])
    vehicles_per_day = list(instance["vehicles_per_day"])
    veh_cap = float(instance["vehicle_capacity"])

    # Build id -> customer dict (depot included at id 0).
    cust_by_id = {c["id"]: c for c in customers}
    if depot["id"] not in cust_by_id:
        # store depot under id 0 so distance() can use it uniformly
        cust_by_id[depot["id"]] = depot
    customer_id_list = sorted(c["id"] for c in customers)

    # Precompute pairwise Euclidean distances on demand via memo
    _dist_cache: dict[tuple[int, int], float] = {}

    def _dist(i: int, j: int) -> float:
        if i == j:
            return 0.0
        key = (i, j) if i < j else (j, i)
        d = _dist_cache.get(key)
        if d is not None:
            return d
        a = cust_by_id[i]
        b = cust_by_id[j]
        d = math.sqrt((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2)
        _dist_cache[key] = d
        return d

    def _tour_length(tour: list) -> float:
        if len(tour) < 2:
            return 0.0
        total = 0.0
        for k in range(len(tour) - 1):
            total += _dist(tour[k], tour[k + 1])
        return total

    # ==================================================================
    # (1) Queries
    # ==================================================================
    def customer_ids() -> list:
        """All customer ids (depot excluded), sorted."""
        return list(customer_id_list)

    def customer_demand(c: int) -> float:
        """Demand of customer c. Depot (id 0) returns 0.0."""
        if c == depot["id"]:
            return 0.0
        cu = cust_by_id.get(int(c))
        if cu is None:
            raise KeyError(f"unknown customer id: {c}")
        return float(cu["demand"])

    def customer_candidate_schedules(c: int) -> list:
        """List of candidate schedules for customer c (each a binary list of
        length period_length). Useful before picking one in
        selected_schedules[c]."""
        cu = cust_by_id.get(int(c))
        if cu is None or c == depot["id"]:
            raise KeyError(f"unknown customer id: {c}")
        return [list(s) for s in cu["schedules"]]

    def customer_visit_count(c: int, schedule: Optional[list] = None) -> int:
        """Number of days customer c is visited.
        If `schedule` is given, returns sum(schedule).
        Else if all candidate schedules have the same total, returns that
        common count; otherwise raises ValueError (caller must pick one)."""
        if schedule is not None:
            return int(sum(int(v) for v in schedule))
        cu = cust_by_id.get(int(c))
        if cu is None or c == depot["id"]:
            raise KeyError(f"unknown customer id: {c}")
        counts = {sum(int(v) for v in s) for s in cu["schedules"]}
        if len(counts) == 1:
            return int(next(iter(counts)))
        raise ValueError(
            f"customer {c} has candidate schedules with different visit "
            f"counts {sorted(counts)}; pass `schedule=` to disambiguate"
        )

    def customer_allowed_periods(c: int) -> list:
        """1-indexed days on which customer c CAN be visited under at least
        one of its candidate schedules. The union of the supports of all
        candidate schedules."""
        cu = cust_by_id.get(int(c))
        if cu is None or c == depot["id"]:
            raise KeyError(f"unknown customer id: {c}")
        s_union = set()
        for sched in cu["schedules"]:
            for d, v in enumerate(sched, start=1):
                if int(v) == 1:
                    s_union.add(d)
        return sorted(s_union)

    def vehicle_capacity() -> float:
        return float(veh_cap)

    def num_periods() -> int:
        return int(period_length)

    def vehicles_on_period(period: int) -> int:
        """Number of vehicles available on day `period` (1-indexed)."""
        if not (1 <= period <= period_length):
            raise ValueError(f"period {period} out of [1, {period_length}]")
        return int(vehicles_per_day[period - 1])

    def distance(i: int, j: int) -> float:
        """Euclidean distance between vertices i and j. Depot id is 0."""
        return _dist(int(i), int(j))

    # ==================================================================
    # (2) Feasibility primitives
    # ==================================================================
    def period_required_customers(period: int, schedule: dict) -> list:
        """Customers whose chosen schedule says they must be visited on
        day `period` (1-indexed). `schedule` must be a dict cust_id -> binary
        vector (same shape as solution['selected_schedules'])."""
        if not (1 <= period <= period_length):
            raise ValueError(f"period {period} out of [1, {period_length}]")
        out = []
        for cid, sched in schedule.items():
            if cid == depot["id"]:
                continue
            if sched is None:
                continue
            if int(sched[period - 1]) == 1:
                out.append(int(cid))
        return sorted(out)

    def period_demand(period: int, schedule: dict) -> float:
        """Total demand of all customers required on day `period` by
        `schedule`. Useful sanity check: must be <= vehicles_on_period(p) *
        vehicle_capacity for feasibility."""
        return sum(customer_demand(c) for c in period_required_customers(period, schedule))

    def period_routes_valid(period: int, tours_day: list) -> tuple:
        """Validate the routes for a single day.
        Returns (ok: bool, reason: str | None).
        Checks: (a) at most vehicles_on_period tours, (b) each tour begins
        and ends at depot 0 with no mid-depot, (c) per-tour demand <=
        vehicle_capacity, (d) no customer visited twice across the day."""
        if not (1 <= period <= period_length):
            return False, f"period {period} out of [1, {period_length}]"
        max_v = vehicles_on_period(period)
        if len(tours_day) > max_v:
            return False, f"day {period}: {len(tours_day)} tours > {max_v} vehicles"
        seen = set()
        for k, tour in enumerate(tours_day):
            if len(tour) < 2 or tour[0] != 0 or tour[-1] != 0:
                return False, f"day {period} tour {k}: must start and end at depot 0"
            if 0 in tour[1:-1]:
                return False, f"day {period} tour {k}: depot appears mid-tour"
            load = 0.0
            for v in tour[1:-1]:
                if v in seen:
                    return False, f"day {period}: customer {v} visited twice"
                seen.add(v)
                load += customer_demand(v)
            if load > veh_cap + 1e-9:
                return False, f"day {period} tour {k}: load {load} > capacity {veh_cap}"
        return True, None

    def unassigned_visits(schedule: dict) -> list:
        """Customer ids whose chosen schedule is missing or empty (no day
        marked 1). Returns sorted list."""
        out = []
        for cid in customer_id_list:
            s = schedule.get(cid)
            if s is None or len(s) != period_length or sum(int(v) for v in s) == 0:
                out.append(cid)
        return out

    # ==================================================================
    # (3) Construction heuristics
    # ==================================================================
    def nn_route(visit_list: Iterable[int]) -> list:
        """Nearest-neighbor TSP route over a subset of customers, returning
        a tour [0, v1, v2, ..., vk, 0]. Starts from the depot. Does NOT
        respect vehicle capacity -- use solve_period_routing for the
        capacity-aware version."""
        remaining = set(int(v) for v in visit_list)
        if not remaining:
            return [0, 0]  # empty route; caller may filter these out
        tour = [0]
        cur = 0
        while remaining:
            nxt = min(remaining, key=lambda u: _dist(cur, u))
            tour.append(int(nxt))
            remaining.discard(nxt)
            cur = nxt
        tour.append(0)
        return tour

    def apply_2opt_route(route: list, time_limit_s: float = 2.0) -> list:
        """2-opt local search on a single CVRP route [0, ..., 0]. Reverses
        sub-segments to remove crossings. The depot is pinned at both ends
        (indices 0 and -1). Pure Python, O(k^2) per pass. Returns a new
        route (the input is not mutated)."""
        r = list(route)
        n = len(r)
        if n < 5:
            return r  # nothing to 2-opt on
        t0 = time.time()
        safety = 0.02
        improved = True
        while improved and (time.time() - t0) < time_limit_s - safety:
            improved = False
            # i, j range over INTERIOR indices [1, n-2]. Reversing r[i:j+1]
            # changes edges (r[i-1], r[i]) and (r[j], r[j+1]).
            for i in range(1, n - 2):
                if (time.time() - t0) >= time_limit_s - safety:
                    break
                a, b = r[i - 1], r[i]
                for j in range(i + 1, n - 1):
                    c, d = r[j], r[j + 1]
                    delta = (_dist(a, c) + _dist(b, d)
                             - _dist(a, b) - _dist(c, d))
                    if delta < -1e-10:
                        r[i:j + 1] = r[i:j + 1][::-1]
                        improved = True
                        break
                if improved:
                    break
        return r

    def solve_period_routing(period: int,
                             customers_visiting: Iterable[int],
                             time_limit_s: float = 3.0) -> list:
        """Solve the CVRP for one day: given which customers must be visited
        today, return a list of capacity-feasible tours (each [0, ..., 0]).

        Heuristic:
          1. Sweep-like seed: order customers by polar angle around depot.
          2. Pack into bins by capacity (first-fit, in angle order). This
             gives a small number of tours that respect vehicle_capacity.
          3. Apply nearest-neighbor TSP within each bin, then 2-opt.

        If the resulting number of tours exceeds vehicles_on_period, this
        function STILL returns them all -- the LLM must decide whether to
        re-route or to switch to a different schedule. The returned list may
        therefore be infeasible w.r.t. the vehicle count constraint, but
        every individual route is capacity-feasible.

        Empty input -> empty list (zero tours; allowed by eval_func)."""
        custs = [int(c) for c in customers_visiting]
        if not custs:
            return []
        # Total demand quick check
        total_d = sum(customer_demand(c) for c in custs)
        if any(customer_demand(c) > veh_cap + 1e-9 for c in custs):
            # infeasible at the instance level; still try to route others
            pass

        # Step 1: angle ordering around the depot
        dx0, dy0 = float(depot["x"]), float(depot["y"])

        def _angle(c: int) -> float:
            cu = cust_by_id[c]
            return math.atan2(cu["y"] - dy0, cu["x"] - dx0)

        ordered = sorted(custs, key=_angle)

        # Step 2: first-fit by capacity in angle order -> bins
        bins: list[list[int]] = []
        loads: list[float] = []
        for c in ordered:
            d = customer_demand(c)
            placed = False
            for bi in range(len(bins)):
                if loads[bi] + d <= veh_cap + 1e-9:
                    bins[bi].append(c)
                    loads[bi] += d
                    placed = True
                    break
            if not placed:
                bins.append([c])
                loads.append(d)

        # Step 3: NN + 2-opt within each bin. Budget time across bins.
        if not bins:
            return []
        per_bin_budget = max(0.05, (time_limit_s - 0.05) / len(bins))
        tours_out = []
        for bi, members in enumerate(bins):
            if not members:
                continue
            tour = nn_route(members)
            tour = apply_2opt_route(tour, time_limit_s=per_bin_budget)
            tours_out.append(tour)
        return tours_out

    def assign_schedules_greedy(seed: Optional[int] = None) -> dict:
        """Heuristic: for each customer, pick the candidate schedule that
        best balances per-day total demand. Returns a dict
        cust_id -> chosen_schedule (binary list of length period_length).

        Processing order: customers with highest demand first (these are
        the hardest to fit), then ties broken by a random tiebreaker (use
        `seed` to reproduce). Among that customer's candidate schedules,
        pick the one minimizing max-loaded-day after the customer is
        added."""
        rng = random.Random(seed if seed is not None else 0)
        order = sorted(
            customer_id_list,
            key=lambda c: (-cust_by_id[c]["demand"], rng.random()),
        )
        day_load = [0.0] * period_length
        assignment: dict[int, list] = {}
        for cid in order:
            cu = cust_by_id[cid]
            best_sched = None
            best_metric = None
            for sched in cu["schedules"]:
                # tentative max-day-load if we add this schedule
                tentative_max = 0.0
                d = float(cu["demand"])
                for k, v in enumerate(sched):
                    extra = d if int(v) == 1 else 0.0
                    tentative_max = max(tentative_max, day_load[k] + extra)
                if best_metric is None or tentative_max < best_metric:
                    best_metric = tentative_max
                    best_sched = list(sched)
            assignment[cid] = best_sched
            for k, v in enumerate(best_sched):
                if int(v) == 1:
                    day_load[k] += float(cu["demand"])
        return assignment

    # ==================================================================
    # (4) Heavy: ILP for schedule assignment
    # ==================================================================
    def ilp_schedule_assignment(time_limit_s: float = 10.0,
                                objective: str = "balance",
                                capacity_factor: float = 1.0) -> Optional[dict]:
        """Solve a Set-Partitioning-style ILP that picks exactly one
        candidate schedule per customer. Per-day CVRP routing is NOT solved
        by the ILP -- only the schedule choice. The LLM should call
        `solve_period_routing` afterward on each day.

        Decision variables: x[c, s] in {0, 1}, "customer c uses schedule s".
        Per customer: sum_s x[c, s] == 1.
        Per day d: a CAPACITY relaxation
            sum_{c, s: schedule s has 1 on day d} demand_c * x[c, s]
              <= vehicles_per_day[d-1] * vehicle_capacity
        This is necessary (not sufficient) for daily routing feasibility.

        objective:
          - "balance": minimize the per-day load excess above the period
            mean (helps spread demand evenly). Models a proxy for routing
            cost.
          - "min_max_load": minimize the maximum daily load.
          - "min_total_demand_days": minimize sum over (c, s) of (chosen
            visit count) * demand_c. Equal across feasible assignments when
            visit counts are fixed -- mostly useful when candidate
            schedules have different visit counts.

        Returns a dict cust_id -> chosen_schedule, or None if infeasible /
        mip unavailable."""
        if not _HAS_MIP:
            return None
        m = Model(sense=MINIMIZE)
        m.verbose = 0
        m.max_seconds = float(time_limit_s)

        # x[cid, sidx] -- sidx = index into customer's schedule list
        x: dict[tuple[int, int], object] = {}
        for cu in customers:
            cid = cu["id"]
            for si, _sched in enumerate(cu["schedules"]):
                x[(cid, si)] = m.add_var(var_type=BINARY, name=f"x_{cid}_{si}")

        # Pick exactly one schedule per customer
        for cu in customers:
            cid = cu["id"]
            m += xsum(x[(cid, si)] for si in range(len(cu["schedules"]))) == 1, \
                 f"choose_one_{cid}"

        # Per-day load expressions
        day_load_expr = []
        for d in range(period_length):
            terms = []
            for cu in customers:
                cid = cu["id"]
                dem = float(cu["demand"])
                for si, sched in enumerate(cu["schedules"]):
                    if int(sched[d]) == 1:
                        terms.append(dem * x[(cid, si)])
            day_load_expr.append(xsum(terms) if terms else 0)

        # Per-day capacity constraint (capacity_factor tightens the bound
        # below the nominal vehicles*capacity to leave headroom for bin
        # packing inefficiency in the daily routing step).
        cf = max(0.1, min(1.0, float(capacity_factor)))
        for d in range(period_length):
            cap_d = cf * float(vehicles_per_day[d]) * float(veh_cap)
            m += day_load_expr[d] <= cap_d, f"cap_day_{d + 1}"

        if objective == "min_max_load":
            L = m.add_var(name="L", lb=0.0)
            for d in range(period_length):
                m += day_load_expr[d] <= L, f"maxload_{d + 1}"
            m.objective = L
        elif objective == "balance":
            # minimize sum of |load_d - mean|; linearized as 2 * sum of
            # one-sided slack above the mean.
            total_demand_per_visit = sum(
                float(cu["demand"]) * min(sum(s) for s in cu["schedules"])
                for cu in customers
            )
            # Approx mean using a loose bound (true mean is solution-
            # dependent; we use the proxy: total demand if every customer
            # visited once, divided across days).
            proxy_mean = total_demand_per_visit / max(1, period_length)
            slack = [m.add_var(name=f"sl_{d + 1}", lb=0.0)
                     for d in range(period_length)]
            for d in range(period_length):
                m += day_load_expr[d] - proxy_mean <= slack[d], f"bal_{d + 1}"
            m.objective = xsum(slack)
        elif objective == "min_total_demand_days":
            terms = []
            for cu in customers:
                cid = cu["id"]
                dem = float(cu["demand"])
                for si, sched in enumerate(cu["schedules"]):
                    terms.append(dem * sum(int(v) for v in sched) * x[(cid, si)])
            m.objective = xsum(terms)
        else:
            raise ValueError(f"unknown objective: {objective!r}")

        status = m.optimize()
        if status not in (OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE):
            return None
        if m.num_solutions < 1:
            return None

        out: dict[int, list] = {}
        for cu in customers:
            cid = cu["id"]
            chosen = None
            for si, sched in enumerate(cu["schedules"]):
                v = x[(cid, si)].x
                if v is not None and v > 0.5:
                    chosen = list(sched)
                    break
            if chosen is None:
                return None  # solver returned malformed solution
            out[cid] = chosen
        return out

    # ==================================================================
    # End-to-end solver
    # ==================================================================
    def make_solution(selected_schedules: dict, tours: dict) -> dict:
        """Wrap the schedule + tours into the EXACT dict shape eval_func
        expects: {'selected_schedules': {cust_id: binary_vec},
                  'tours': {day_1idx: list_of_tours}}.
        Each tour is a list [0, ..., 0] (starts and ends at depot 0)."""
        return {
            "selected_schedules": {int(k): list(v) for k, v in (selected_schedules or {}).items()},
            "tours": {int(d): [list(t) for t in (tours_d or [])]
                      for d, tours_d in (tours or {}).items()},
        }

    def _rebalance_schedule(sched: dict, target_factor: float = 0.92) -> dict:
        """Shift customers between days to reduce peak day_load below
        target_factor * day_capacity. Walks heaviest-day customers (sorted
        by demand descending) and tries each alternative candidate schedule
        that lightens the heavy day. Accepts a swap only if no other day's
        load exceeds its day_capacity. Returns a NEW schedule dict; does
        not mutate input."""
        if not sched:
            return sched
        sched = {int(k): list(v) for k, v in sched.items()}
        # Track best (lowest max-excess) state to avoid oscillation.
        best_sched = {k: list(v) for k, v in sched.items()}
        best_max_excess = float("inf")
        for _outer in range(100):
            day_loads = [0.0] * period_length
            for cid, sc in sched.items():
                dem = customer_demand(cid)
                for d_idx, v in enumerate(sc):
                    if int(v) == 1:
                        day_loads[d_idx] += dem
            # Find day with highest excess above target.
            day_caps = [float(vehicles_per_day[d]) * veh_cap
                        for d in range(period_length)]
            excesses = [(day_loads[d] - target_factor * day_caps[d], d)
                        for d in range(period_length)]
            excesses.sort(reverse=True)
            worst_excess, worst_d = excesses[0]
            # Maintain best-so-far snapshot to defeat oscillation.
            if worst_excess < best_max_excess - 1e-6:
                best_max_excess = worst_excess
                best_sched = {k: list(v) for k, v in sched.items()}
            if worst_excess <= 0:
                return sched
            # Try shifting some customer off worst_d.
            # Order: heaviest demand first (bigger relief per shift).
            on_d = [cid for cid in sched
                    if int(sched[cid][worst_d]) == 1
                    and cid != depot["id"]]
            on_d.sort(key=lambda c: -customer_demand(c))
            shifted = False
            for cid in on_d:
                cu = cust_by_id[cid]
                cur_sched = sched[cid]
                for alt in cu["schedules"]:
                    if list(alt) == cur_sched:
                        continue
                    if int(alt[worst_d]) == 1:
                        continue  # still on worst day -- no relief
                    # Check that adopting alt doesn't push any other day
                    # above its cap.
                    new_loads = list(day_loads)
                    dem = customer_demand(cid)
                    for d_idx in range(period_length):
                        new_loads[d_idx] += dem * (int(alt[d_idx]) - int(cur_sched[d_idx]))
                    if all(new_loads[d_idx] <= day_caps[d_idx] + 1e-9
                           for d_idx in range(period_length)):
                        sched[cid] = list(alt)
                        shifted = True
                        break
                if shifted:
                    break
            if not shifted:
                return best_sched  # no further legal shift
        return best_sched

    def _bin_pack_two_via_subset_sum(custs: list) -> Optional[list]:
        """For the k=2 case, use subset-sum DP to find a 2-bin partition
        whose larger bin is as small as possible. Returns the 2 bins (each
        a list of customer ids), or None if either bin would exceed
        veh_cap. Integer demands assumed (we round up to avoid float DP).
        O(n * total_demand) which is fine for typical instances."""
        if not custs:
            return [[], []]
        custs = list(custs)
        demands = [customer_demand(c) for c in custs]
        # Convert to int weights (multiply by a small factor if non-integer).
        if all(abs(d - round(d)) < 1e-6 for d in demands):
            w = [int(round(d)) for d in demands]
            scale = 1
        else:
            scale = 100
            w = [int(round(d * scale)) for d in demands]
        cap_int = int(veh_cap * scale + 1e-9)
        total = sum(w)
        if total > 2 * cap_int + 1e-9:
            return None
        # We want subset S whose weight is close to total/2 but <= cap_int.
        # Run knapsack DP: max subset weight <= cap_int.
        reachable = {0: -1}  # weight -> last cust idx used (for backtrack)
        prev_choice = {}  # (weight) -> previous weight, choice taken
        for i, wi in enumerate(w):
            new_reachable = dict(reachable)
            for wt, last in reachable.items():
                nw = wt + wi
                if nw <= cap_int and nw not in new_reachable:
                    new_reachable[nw] = i
                    prev_choice[nw] = (wt, i)
            reachable = new_reachable
        # Pick the largest reachable weight that, together with the
        # complement (total - weight), keeps the complement <= cap_int too.
        best = None
        for wt in sorted(reachable.keys(), reverse=True):
            if (total - wt) <= cap_int:
                best = wt
                break
        if best is None:
            return None
        # Backtrack to recover items in bin A.
        in_a = [False] * len(custs)
        cur = best
        while cur > 0 and cur in prev_choice:
            prev, idx = prev_choice[cur]
            in_a[idx] = True
            cur = prev
        bin_a = [custs[i] for i in range(len(custs)) if in_a[i]]
        bin_b = [custs[i] for i in range(len(custs)) if not in_a[i]]
        return [bin_a, bin_b]

    def _bin_pack_into_k_bins_bfd(custs: list, k_max: int) -> Optional[list]:
        """Best-fit decreasing bin-pack: try to pack `custs` into AT MOST
        k_max bins, each <= veh_cap. Returns the bins (list of customer-id
        lists) or None if no such packing exists. O(n * k) per pass."""
        if not custs:
            return []
        items = sorted(custs, key=lambda c: -customer_demand(c))
        # Quick infeasibility check: total demand cap.
        total = sum(customer_demand(c) for c in items)
        if total > k_max * veh_cap + 1e-9:
            return None
        bins: list[list[int]] = []
        loads: list[float] = []
        for c in items:
            d = customer_demand(c)
            # Best-fit: tightest bin that still fits.
            best_b = -1
            best_slack = float("inf")
            for bi in range(len(bins)):
                slack = veh_cap - loads[bi] - d
                if slack >= -1e-9 and slack < best_slack:
                    best_slack = slack
                    best_b = bi
            if best_b >= 0:
                bins[best_b].append(c)
                loads[best_b] += d
            else:
                if len(bins) >= k_max:
                    return None  # no room for new bin
                bins.append([c])
                loads.append(d)
        return bins

    def _route_day_tight(period: int, custs: list, time_limit_s: float = 3.0) -> list:
        """Stronger daily router that enforces vehicles_on_period(period).
        Tries solve_period_routing first; if that yields > max_v tours,
        falls back to a best-fit-decreasing bin pack into exactly max_v
        bins, then nn + 2-opt per bin. Returns a list of capacity-feasible
        tours; if even BFD can't fit within max_v bins, returns whatever
        BFD gave (more bins than allowed -- caller's problem)."""
        import time as _t
        t0 = _t.time()
        max_v = vehicles_on_period(period)
        # Attempt 1: existing geometric heuristic.
        tours = solve_period_routing(period, custs, time_limit_s=time_limit_s * 0.5)
        if len(tours) <= max_v:
            return tours
        # Attempt 2: BFD into exactly max_v bins.
        bins = _bin_pack_into_k_bins_bfd(list(custs), max_v)
        # Attempt 3 (k=2 only): subset-sum DP for the tight case where BFD
        # fails or is borderline. This finds exact partitions when they exist.
        if max_v == 2:
            ss_bins = _bin_pack_two_via_subset_sum(list(custs))
            if ss_bins is not None:
                if bins is None or len(bins) > 2:
                    bins = ss_bins
        if bins is None:
            # Demand exceeds capacity -- just return the geometric result.
            return tours
        out: list = []
        remaining = max(0.2, time_limit_s - (_t.time() - t0))
        per_bin = remaining / max(1, len(bins))
        for members in bins:
            if not members:
                continue
            tour = nn_route(members)
            tour = apply_2opt_route(tour, time_limit_s=per_bin)
            out.append(tour)
        return out

    def _merge_into_k_tours(tours_in: list, k_max: int,
                            time_limit_s: float = 1.0) -> list:
        """Force a list of tours [0,...,0] down to at most k_max tours by
        repeatedly merging the two lightest-load tours whose combined load
        stays <= veh_cap. Each merged tour is re-sequenced via nn over its
        customers and lightly polished with 2-opt. If no further legal
        merge exists, returns the current state (which may still exceed
        k_max -- the caller should treat as infeasible for that day)."""
        if k_max <= 0:
            return []
        tours = [list(t) for t in tours_in if len(t) > 2]
        while len(tours) > k_max:
            # Score each tour by demand load.
            loads = []
            for idx, t in enumerate(tours):
                d = sum(customer_demand(v) for v in t[1:-1])
                loads.append((d, idx))
            loads.sort()
            # Try every (lightest, next) pair until one merges legally.
            merged = False
            for a_pos in range(len(loads)):
                for b_pos in range(a_pos + 1, len(loads)):
                    la, ia = loads[a_pos]
                    lb, ib = loads[b_pos]
                    if la + lb > veh_cap + 1e-9:
                        continue
                    custs = (tours[ia][1:-1] + tours[ib][1:-1])
                    new_tour = nn_route(custs)
                    new_tour = apply_2opt_route(new_tour, time_limit_s=time_limit_s / max(1, len(tours)))
                    tours = [t for k, t in enumerate(tours)
                             if k != ia and k != ib]
                    tours.append(new_tour)
                    merged = True
                    break
                if merged:
                    break
            if not merged:
                break  # no further legal merge
        return tours

    def solve_pvrp(time_limit_s: float = 30.0,
                   schedule_time_s: float = 8.0) -> dict:
        """ONE-SHOT END-TO-END SOLVER for the Period VRP. Returns the
        complete solution dict {'selected_schedules': ..., 'tours': ...}
        ready to return directly.

        Strategy (two layers, both handled here so the LLM doesn't have to):
          1. Schedule choice. Try ilp_schedule_assignment for up to
             schedule_time_s seconds (balances per-day load via an ILP
             with a per-day capacity relaxation). Falls back to
             assign_schedules_greedy if the ILP is unavailable.
          2. Daily CVRP. For each day d, call solve_period_routing on
             the customers whose chosen schedule has a 1 on day d.
             Budgets the remaining time evenly across days.

        Returns a feasible solution PROVIDED the chosen schedule is
        actually routable under the daily vehicle / capacity limits.
        If not, the framework's is_feasible will flag the offending day --
        in which case retry with a tighter schedule_time_s and a smaller
        vehicles_per_day relaxation.

        Use as the FIRST thing your solve() function calls. ONE LINE:
            return tools['solve_pvrp'](time_limit_s=30)
        """
        import time as _time
        t0 = _time.time()

        def _attempt(cap_factor: float, sched_budget: float) -> tuple:
            sched_local = None
            if _HAS_MIP:
                try:
                    sched_local = ilp_schedule_assignment(
                        time_limit_s=sched_budget, objective="balance",
                        capacity_factor=cap_factor,
                    )
                except Exception:
                    sched_local = None
            if sched_local is None:
                sched_local = assign_schedules_greedy(seed=0)
            tours_local: dict[int, list] = {}
            remaining_local = time_limit_s - (_time.time() - t0)
            per_day = max(0.5, remaining_local / max(1, period_length * 2))
            ok_local = True
            for d in range(1, period_length + 1):
                custs_today = period_required_customers(d, sched_local)
                if not custs_today:
                    tours_local[d] = []
                    continue
                day_tours = solve_period_routing(
                    d, custs_today,
                    time_limit_s=per_day,
                )
                max_v = vehicles_on_period(d)
                if len(day_tours) > max_v:
                    day_tours = _merge_into_k_tours(
                        day_tours, max_v,
                        time_limit_s=min(2.0, per_day / 2),
                    )
                if len(day_tours) > max_v:
                    ok_local = False
                tours_local[d] = day_tours
            return sched_local, tours_local, ok_local

        # First attempt: nominal capacity_factor=1.0.
        budget_per_try = max(2.0, schedule_time_s)
        sched, tours, ok = _attempt(1.0, budget_per_try)
        if ok:
            return make_solution(sched, tours)
        # Retry: rebalance the schedule (shift customers off overloaded
        # days to days with slack) and re-route.
        best = (sched, tours)
        for tf in (0.95, 0.90, 0.85, 0.80, 0.75):
            if _time.time() - t0 > time_limit_s - 1.5:
                break
            sched_r = _rebalance_schedule(sched, target_factor=tf)
            # Re-route with the rebalanced schedule.
            tours_r: dict[int, list] = {}
            ok_r = True
            per_day = max(0.5, (time_limit_s - (_time.time() - t0))
                          / max(1, period_length * 2))
            for d in range(1, period_length + 1):
                custs_today = period_required_customers(d, sched_r)
                if not custs_today:
                    tours_r[d] = []
                    continue
                # Tighter router that enforces vehicle count via BFD.
                day_tours = _route_day_tight(d, custs_today, time_limit_s=per_day)
                max_v = vehicles_on_period(d)
                if len(day_tours) > max_v:
                    day_tours = _merge_into_k_tours(
                        day_tours, max_v,
                        time_limit_s=min(2.0, per_day / 2))
                if len(day_tours) > max_v:
                    ok_r = False
                tours_r[d] = day_tours
            if ok_r:
                return make_solution(sched_r, tours_r)
            best = (sched_r, tours_r)
        return make_solution(best[0], best[1])

    return {
        # one-shot end-to-end + builder (CALL FIRST)
        "solve_pvrp": solve_pvrp,
        "solve_default": solve_pvrp,  # alias for consistency across problems
        "make_solution": make_solution,
        # Heavy
        "ilp_schedule_assignment": ilp_schedule_assignment,
        # Construction heuristics
        "assign_schedules_greedy": assign_schedules_greedy,
        "solve_period_routing": solve_period_routing,
        "nn_route": nn_route,
        "apply_2opt_route": apply_2opt_route,
        # Feasibility primitives
        "period_required_customers": period_required_customers,
        "period_demand": period_demand,
        "period_routes_valid": period_routes_valid,
        "unassigned_visits": unassigned_visits,
        # Queries
        "customer_ids": customer_ids,
        "customer_demand": customer_demand,
        "customer_candidate_schedules": customer_candidate_schedules,
        "customer_visit_count": customer_visit_count,
        "customer_allowed_periods": customer_allowed_periods,
        "vehicle_capacity": vehicle_capacity,
        "num_periods": num_periods,
        "vehicles_on_period": vehicles_on_period,
        "distance": distance,
    }


EXTRA_TOOLS_DESCRIPTION = [
    # ----- ONE-SHOT END-TO-END SOLVER (call this first!) -----
    {
        "name": "solve_pvrp",
        "input": "time_limit_s: float = 30.0, schedule_time_s: float = 8.0",
        "output": "dict {'selected_schedules': ..., 'tours': ...}",
        "purpose": (
            "RECOMMENDED START: returns a complete solution dict ready to return "
            "directly. Handles BOTH layers of the Period VRP end-to-end: (1) "
            "picks per-customer schedules via ilp_schedule_assignment (load-"
            "balanced ILP, falls back to assign_schedules_greedy); (2) for each "
            "day, runs solve_period_routing on that day's required customers. "
            "Budgets the remaining time evenly across days. ONE LINE: "
            "`return tools['solve_pvrp'](time_limit_s=30)`. Also exposed as "
            "`solve_default` for consistency."
        ),
    },
    {
        "name": "solve_default",
        "input": "time_limit_s: float = 30.0, schedule_time_s: float = 8.0",
        "output": "dict {'selected_schedules': ..., 'tours': ...}",
        "purpose": "Alias for solve_pvrp. Use either name interchangeably.",
    },
    {
        "name": "make_solution",
        "input": "selected_schedules: dict, tours: dict",
        "output": "dict {'selected_schedules', 'tours'}",
        "purpose": (
            "Build the EXACT solution dict shape eval_func wants: "
            "{'selected_schedules': {cust_id: binary_vec}, 'tours': {day_1idx: "
            "[tours]}}. Use to assemble the final dict from "
            "assign_schedules_greedy() + solve_period_routing() outputs."
        ),
    },
    # ----- Queries -----
    {
        "name": "customer_ids",
        "input": "(no args)",
        "output": "list[int]",
        "purpose": (
            "Sorted list of all customer ids in the instance (depot 0 excluded). "
            "Iterate over this when building selected_schedules."
        ),
    },
    {
        "name": "customer_demand",
        "input": "c: int",
        "output": "float",
        "purpose": (
            "Demand of customer c. Depot (id 0) returns 0.0. Raises KeyError for "
            "unknown ids."
        ),
    },
    {
        "name": "customer_candidate_schedules",
        "input": "c: int",
        "output": "list[list[int]]",
        "purpose": (
            "Candidate schedules for customer c -- each is a binary list of length "
            "period_length. You MUST select exactly one of these per customer for "
            "the final solution['selected_schedules'][c]."
        ),
    },
    {
        "name": "customer_visit_count",
        "input": "c: int, schedule: list[int] = None",
        "output": "int",
        "purpose": (
            "Number of days customer c is visited. If `schedule` is given, returns "
            "sum(schedule). Otherwise returns the common visit count across all "
            "candidate schedules (and raises ValueError if candidates differ)."
        ),
    },
    {
        "name": "customer_allowed_periods",
        "input": "c: int",
        "output": "list[int]",
        "purpose": (
            "1-indexed days on which customer c COULD be visited under at least one "
            "of its candidate schedules (the union of supports). Use to prune ILPs / "
            "neighborhoods."
        ),
    },
    {
        "name": "vehicle_capacity",
        "input": "(no args)",
        "output": "float",
        "purpose": "Per-vehicle capacity (same for every vehicle and every day).",
    },
    {
        "name": "num_periods",
        "input": "(no args)",
        "output": "int",
        "purpose": "Number of days in the planning horizon (== period_length).",
    },
    {
        "name": "vehicles_on_period",
        "input": "period: int",
        "output": "int",
        "purpose": (
            "Number of vehicles available on day `period` (1-indexed). The number "
            "of tours you return for that day MUST NOT exceed this."
        ),
    },
    {
        "name": "distance",
        "input": "i: int, j: int",
        "output": "float",
        "purpose": (
            "Euclidean distance between vertices i and j (depot id is 0). Cached. "
            "Use to score moves without recomputing whole routes."
        ),
    },
    # ----- Feasibility primitives -----
    {
        "name": "period_required_customers",
        "input": "period: int, schedule: dict[int, list[int]]",
        "output": "list[int]",
        "purpose": (
            "Customers that MUST be visited on day `period` (1-indexed) given a "
            "schedule assignment dict (cust_id -> binary vector, same shape as "
            "solution['selected_schedules']). Sorted."
        ),
    },
    {
        "name": "period_demand",
        "input": "period: int, schedule: dict[int, list[int]]",
        "output": "float",
        "purpose": (
            "Total demand on day `period` under `schedule`. Quick infeasibility "
            "check: must be <= vehicles_on_period(p) * vehicle_capacity."
        ),
    },
    {
        "name": "period_routes_valid",
        "input": "period: int, tours_day: list[list[int]]",
        "output": "(bool, str | None)",
        "purpose": (
            "Validate one day's tours: (a) tour count <= vehicles_on_period, "
            "(b) each tour begins / ends at depot with no mid-depot, (c) each "
            "tour load <= vehicle_capacity, (d) no customer visited twice. "
            "Returns (True, None) on success, else (False, reason)."
        ),
    },
    {
        "name": "unassigned_visits",
        "input": "schedule: dict[int, list[int]]",
        "output": "list[int]",
        "purpose": (
            "Customer ids in the instance whose entry in `schedule` is missing, "
            "wrong-length, or all-zero. Sorted. Must be empty for a valid "
            "selected_schedules."
        ),
    },
    # ----- Construction heuristics -----
    {
        "name": "nn_route",
        "input": "visit_list: Iterable[int]",
        "output": "list[int]",
        "purpose": (
            "Nearest-neighbor TSP route starting and ending at the depot over the "
            "given customer subset. Returns [0, v1, v2, ..., vk, 0]. Does NOT "
            "respect vehicle capacity -- use solve_period_routing for that."
        ),
    },
    {
        "name": "apply_2opt_route",
        "input": "route: list[int], time_limit_s: float = 2.0",
        "output": "list[int]",
        "purpose": (
            "2-opt local search on a single CVRP route [0, ..., 0]. Pins the depot "
            "at both ends. Returns an improved route (input not mutated). O(k^2) "
            "per pass; usually fast for k <= 50."
        ),
    },
    {
        "name": "solve_period_routing",
        "input": "period: int, customers_visiting: Iterable[int], time_limit_s: float = 3.0",
        "output": "list[list[int]]",
        "purpose": (
            "Solve one day's CVRP heuristically: angle-sweep into capacity bins, "
            "nearest-neighbor TSP per bin, then 2-opt. Returns a list of "
            "capacity-feasible tours. WARNING: the number of tours returned can "
            "exceed vehicles_on_period(period) when demand is heavy -- check with "
            "period_routes_valid before accepting."
        ),
    },
    {
        "name": "assign_schedules_greedy",
        "input": "seed: int = None",
        "output": "dict[int, list[int]]",
        "purpose": (
            "Greedy schedule assignment: order customers by descending demand, "
            "pick the candidate schedule that minimizes the resulting max daily "
            "load. Returns a dict cust_id -> chosen schedule. Fast warm start; "
            "feed into solve_period_routing day by day."
        ),
    },
    # ----- Heavy: ILP -----
    {
        "name": "ilp_schedule_assignment",
        "input": "time_limit_s: float = 10.0, objective: str = 'balance'",
        "output": "dict[int, list[int]] | None",
        "purpose": (
            "ILP that picks ONE candidate schedule per customer subject to a "
            "per-day capacity relaxation (sum demand <= num_vehicles * capacity). "
            "Does NOT solve the routing -- combine with solve_period_routing. "
            "`objective` options: 'balance' (minimize sum of per-day loads "
            "exceeding the mean), 'min_max_load' (minimize the busiest day), "
            "'min_total_demand_days' (minimize sum of demand*visits, only "
            "differentiates when candidate schedules have different visit counts). "
            "Returns None if mip is unavailable or no feasible assignment exists."
        ),
    },
]
