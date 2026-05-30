"""Per-problem extras for the Multi-Demand Multidimensional Knapsack Problem
(MDMKP).

MDMKP extends the classical Multidimensional Knapsack (MKP) with additional
LOWER-bound "demand" constraints. Given n binary items with profits c_j,
m capacity constraints (A_leq x <= b_leq), and q demand constraints
(A_geq x >= b_geq), maximise sum c_j x_j.

The CO-Bench solution dict is
    {'x': [0|1, ..., 0|1] (length n), 'optimal_value': <int|float>}.
All tools below accept and return that same 0/1 vector convention so values
flow directly into tools['is_feasible'] / tools['objective'].

Tool groups:
  (1) Queries:        item_profit, item_resource_cap, item_resource_demand,
                      capacity, demand, n_items, n_caps, n_demands
  (2) Feasibility:    current_cap_usage, current_demand_satisfaction,
                      is_caps_satisfied, is_demands_satisfied, missing_demands
  (3) Construction /
      improvement:    greedy_profit_within_caps, greedy_for_demand_then_profit,
                      repair_for_demands, apply_swap_in_out
  (4) Exact / heavy:  ilp_solve_mdmkp
"""
from __future__ import annotations
import time
from typing import Iterable, List, Optional

from mip import Model, BINARY, MAXIMIZE, xsum, OptimizationStatus


def extra_tools(instance: dict) -> dict:
    """Factory: returns MDMKP-specific tool callables given the loaded instance.

    Instance schema (from CO-Bench MDMKP load_data, one variant):
      - n:           int                                 number of items
      - m:           int                                 number of <= constraints
      - q:           int                                 number of active >= constraints
      - A_leq:       list[list[int]] shape (m, n)        <= coefficients
      - b_leq:       list[int] length m                  <= RHS
      - A_geq:       list[list[int]] shape (q, n)        >= coefficients
      - b_geq:       list[int] length q                  >= RHS
      - cost_vector: list[int] length n                  objective coefficients
      - cost_type:   str                                  "positive" or "mixed"
    """
    n: int = int(instance["n"])
    m: int = int(instance["m"])
    q: int = int(instance["q"])
    A_leq = instance["A_leq"]          # m x n
    b_leq = instance["b_leq"]          # length m
    A_geq = instance["A_geq"]          # q x n
    b_geq = instance["b_geq"]          # length q
    cost_vector = instance["cost_vector"]  # length n

    def _check_x(x: Iterable[int]) -> List[int]:
        xl = list(x)
        if len(xl) != n:
            raise ValueError(f"x has length {len(xl)}, expected {n}")
        return xl

    # ==================================================================
    # (1) Queries
    # ==================================================================
    def item_profit(i: int) -> float:
        """cost_vector[i] for item i (0-indexed). May be negative when
        cost_type == 'mixed'."""
        return float(cost_vector[int(i)])

    def item_resource_cap(i: int, k: int) -> float:
        """A_leq[k][i]: usage of <= resource k by item i (0-indexed)."""
        return float(A_leq[int(k)][int(i)])

    def item_resource_demand(i: int, k: int) -> float:
        """A_geq[k][i]: contribution of item i to >= demand k (0-indexed)."""
        return float(A_geq[int(k)][int(i)])

    def capacity(k: int) -> float:
        """b_leq[k]: RHS of the k-th <= constraint (0-indexed)."""
        return float(b_leq[int(k)])

    def demand(k: int) -> float:
        """b_geq[k]: RHS of the k-th >= demand constraint (0-indexed)."""
        return float(b_geq[int(k)])

    def n_items() -> int:
        return n

    def n_caps() -> int:
        return m

    def n_demands() -> int:
        return q

    # ==================================================================
    # (2) Feasibility primitives
    # ==================================================================
    def current_cap_usage(x: Iterable[int]) -> List[float]:
        """Length-m list: usage[k] = sum_j A_leq[k][j] * x[j]. Feasible iff
        usage[k] <= b_leq[k] for all k."""
        xl = _check_x(x)
        return [sum(A_leq[k][j] * xl[j] for j in range(n)) for k in range(m)]

    def current_demand_satisfaction(x: Iterable[int]) -> List[float]:
        """Length-q list: sat[k] = sum_j A_geq[k][j] * x[j]. Feasible iff
        sat[k] >= b_geq[k] for all k."""
        xl = _check_x(x)
        return [sum(A_geq[k][j] * xl[j] for j in range(n)) for k in range(q)]

    def is_caps_satisfied(x: Iterable[int]) -> bool:
        """True iff every <= constraint holds for the given 0/1 vector."""
        xl = _check_x(x)
        for k in range(m):
            if sum(A_leq[k][j] * xl[j] for j in range(n)) > b_leq[k]:
                return False
        return True

    def is_demands_satisfied(x: Iterable[int]) -> bool:
        """True iff every >= demand constraint holds for the given 0/1 vector."""
        xl = _check_x(x)
        for k in range(q):
            if sum(A_geq[k][j] * xl[j] for j in range(n)) < b_geq[k]:
                return False
        return True

    def missing_demands(x: Iterable[int]) -> List[int]:
        """0-indexed indices of demand constraints k for which
        sum_j A_geq[k][j] * x[j] < b_geq[k]. Empty => all demands met."""
        xl = _check_x(x)
        out = []
        for k in range(q):
            if sum(A_geq[k][j] * xl[j] for j in range(n)) < b_geq[k]:
                out.append(k)
        return out

    # ==================================================================
    # (3) Construction / improvement
    # ==================================================================
    def _cap_slack(usage: List[float]) -> List[float]:
        return [float(b_leq[k]) - usage[k] for k in range(m)]

    def _fits(j: int, usage: List[float]) -> bool:
        for k in range(m):
            if usage[k] + A_leq[k][j] > b_leq[k]:
                return False
        return True

    def greedy_profit_within_caps() -> List[int]:
        """Pure MKP-style greedy: ignore demand constraints, add items in
        decreasing 'efficiency' (profit / total resource use) order while every
        <= capacity still has room. Items with non-positive profit are not
        added. Returns a 0/1 vector of length n. The result respects all <=
        constraints but may still leave some >= demands unmet -- pipe through
        repair_for_demands to fix that."""
        # efficiency = profit / sum_k A_leq[k][j] (with eps to avoid div0).
        scores = []
        for j in range(n):
            total = sum(A_leq[k][j] for k in range(m))
            scores.append((cost_vector[j] / (total + 1e-9), j))
        scores.sort(reverse=True)
        x = [0] * n
        usage = [0.0] * m
        for _, j in scores:
            if cost_vector[j] <= 0:
                continue
            if _fits(j, usage):
                x[j] = 1
                for k in range(m):
                    usage[k] += A_leq[k][j]
        return x

    def greedy_for_demand_then_profit() -> List[int]:
        """Two-phase greedy tailored to MDMKP:

        Phase A (cover demands): while any demand k is unmet, add the item j
        (not yet selected) that maximises (unmet demand it satisfies) /
        (capacity usage it incurs). Items that would violate any <= cap are
        skipped. Stops when all demands are met or no candidate is feasible.

        Phase B (maximise profit): among items not yet chosen, greedily pack by
        profit / cap-usage ratio (only items with positive profit, only if they
        fit). Items with non-positive profit are also considered when their
        addition reduces demand violations -- but in phase B all demands are
        already met, so they are skipped.

        Returns a 0/1 vector of length n. Typically a strong warm start; check
        is_demands_satisfied / is_caps_satisfied to confirm.
        """
        x = [0] * n
        usage = [0.0] * m
        sat = [0.0] * q  # current demand coverage
        chosen = set()

        # Phase A: cover demands cheaply.
        while True:
            unmet = [k for k in range(q) if sat[k] < b_geq[k]]
            if not unmet:
                break
            best_j = None
            best_score = -float("inf")
            for j in range(n):
                if j in chosen:
                    continue
                if not _fits(j, usage):
                    continue
                # contribution to unmet demands (only the still-unmet part counts).
                contrib = 0.0
                for k in unmet:
                    remaining = b_geq[k] - sat[k]
                    if remaining <= 0:
                        continue
                    add = A_geq[k][j]
                    if add <= 0:
                        continue
                    contrib += min(add, remaining)
                if contrib <= 0:
                    continue
                cap_use = sum(A_leq[k][j] for k in range(m))
                # Tie-break by profit so we don't sacrifice it unnecessarily.
                score = contrib / (cap_use + 1e-9) + 1e-6 * cost_vector[j]
                if score > best_score:
                    best_score = score
                    best_j = j
            if best_j is None:
                # Cannot cover remaining demands without violating <= -- give up.
                break
            x[best_j] = 1
            chosen.add(best_j)
            for k in range(m):
                usage[k] += A_leq[k][best_j]
            for k in range(q):
                sat[k] += A_geq[k][best_j]

        # Phase B: pack remaining capacity by profit/cap-usage.
        scores = []
        for j in range(n):
            if j in chosen:
                continue
            if cost_vector[j] <= 0:
                continue
            total = sum(A_leq[k][j] for k in range(m))
            scores.append((cost_vector[j] / (total + 1e-9), j))
        scores.sort(reverse=True)
        for _, j in scores:
            if _fits(j, usage):
                x[j] = 1
                chosen.add(j)
                for k in range(m):
                    usage[k] += A_leq[k][j]
        return x

    def repair_for_demands(x: Iterable[int]) -> List[int]:
        """Given a 0/1 vector that satisfies all <= constraints but may miss
        some >= demand constraints, greedily ADD items to cover the missing
        demands. Each step picks the unselected item that:
          - still fits in every <= capacity, and
          - maximises (its contribution to currently-missing demands) /
            (capacity it consumes) -- breaking ties by larger profit.
        Stops when all demands are met or no item helps. Returns a NEW list
        (never mutates input). If demands still missing afterwards the
        instance is too tight for a pure addition repair -- consider
        apply_swap_in_out or ilp_solve_mdmkp."""
        xl = _check_x(x)
        out = list(xl)
        usage = [sum(A_leq[k][j] * out[j] for j in range(n)) for k in range(m)]
        sat = [sum(A_geq[k][j] * out[j] for j in range(n)) for k in range(q)]
        while True:
            unmet = [k for k in range(q) if sat[k] < b_geq[k]]
            if not unmet:
                break
            best_j = None
            best_score = -float("inf")
            for j in range(n):
                if out[j] == 1:
                    continue
                # must fit every <= cap
                ok = True
                for k in range(m):
                    if usage[k] + A_leq[k][j] > b_leq[k]:
                        ok = False
                        break
                if not ok:
                    continue
                contrib = 0.0
                for k in unmet:
                    remaining = b_geq[k] - sat[k]
                    if remaining <= 0:
                        continue
                    add = A_geq[k][j]
                    if add <= 0:
                        continue
                    contrib += min(add, remaining)
                if contrib <= 0:
                    continue
                cap_use = sum(A_leq[k][j] for k in range(m))
                score = contrib / (cap_use + 1e-9) + 1e-6 * cost_vector[j]
                if score > best_score:
                    best_score = score
                    best_j = j
            if best_j is None:
                break
            out[best_j] = 1
            for k in range(m):
                usage[k] += A_leq[k][best_j]
            for k in range(q):
                sat[k] += A_geq[k][best_j]
        return out

    def apply_swap_in_out(x: Iterable[int],
                          t_limit: float = 1.0) -> List[int]:
        """Local search around `x` using two move types:
          - 1-flip: toggle a single bit (in or out) if it keeps both <= and >=
            satisfied and increases profit.
          - 1-in / 1-out swap: simultaneously add one currently-zero item and
            remove one currently-one item, accept if feasible and profit
            improves.
        Iterates until no improving move is found OR the wall-clock budget
        `t_limit` (seconds) is exhausted. Returns a NEW list (input unchanged).
        Best used to polish a heuristic solution; if `x` is infeasible the
        function will only accept moves that are themselves feasible, so a
        feasibility-respecting warm start (e.g. greedy_for_demand_then_profit
        possibly followed by repair_for_demands) is recommended."""
        xl = _check_x(x)
        out = list(xl)
        start = time.time()

        def feasible(vec):
            for k in range(m):
                if sum(A_leq[k][j] * vec[j] for j in range(n)) > b_leq[k]:
                    return False
            for k in range(q):
                if sum(A_geq[k][j] * vec[j] for j in range(n)) < b_geq[k]:
                    return False
            return True

        def profit(vec):
            return sum(cost_vector[j] * vec[j] for j in range(n))

        # incremental bookkeeping: maintain usage[], sat[], cur_profit
        usage = [sum(A_leq[k][j] * out[j] for j in range(n)) for k in range(m)]
        sat = [sum(A_geq[k][j] * out[j] for j in range(n)) for k in range(q)]
        cur_profit = profit(out)

        improved = True
        while improved:
            if time.time() - start > t_limit:
                break
            improved = False

            # ---- 1-flip moves ----
            for j in range(n):
                if time.time() - start > t_limit:
                    break
                if out[j] == 0:
                    # try adding j
                    ok = True
                    for k in range(m):
                        if usage[k] + A_leq[k][j] > b_leq[k]:
                            ok = False
                            break
                    if not ok:
                        continue
                    delta = cost_vector[j]
                    if delta <= 0:
                        continue
                    # adding only HELPS >= constraints -- still feasible.
                    out[j] = 1
                    for k in range(m):
                        usage[k] += A_leq[k][j]
                    for k in range(q):
                        sat[k] += A_geq[k][j]
                    cur_profit += delta
                    improved = True
                else:
                    # try removing j
                    delta = -cost_vector[j]
                    if delta <= 0:
                        continue
                    ok = True
                    for k in range(q):
                        if sat[k] - A_geq[k][j] < b_geq[k]:
                            ok = False
                            break
                    if not ok:
                        continue
                    out[j] = 0
                    for k in range(m):
                        usage[k] -= A_leq[k][j]
                    for k in range(q):
                        sat[k] -= A_geq[k][j]
                    cur_profit += delta
                    improved = True

            # ---- 1-in / 1-out swaps ----
            if time.time() - start > t_limit:
                break
            for jin in range(n):
                if out[jin] != 0:
                    continue
                if time.time() - start > t_limit:
                    break
                for jout in range(n):
                    if out[jout] != 1:
                        continue
                    if jin == jout:
                        continue
                    delta = cost_vector[jin] - cost_vector[jout]
                    if delta <= 0:
                        continue
                    # check <= after swap
                    ok = True
                    for k in range(m):
                        new_u = usage[k] + A_leq[k][jin] - A_leq[k][jout]
                        if new_u > b_leq[k]:
                            ok = False
                            break
                    if not ok:
                        continue
                    # check >= after swap
                    for k in range(q):
                        new_s = sat[k] + A_geq[k][jin] - A_geq[k][jout]
                        if new_s < b_geq[k]:
                            ok = False
                            break
                    if not ok:
                        continue
                    out[jin] = 1
                    out[jout] = 0
                    for k in range(m):
                        usage[k] += A_leq[k][jin] - A_leq[k][jout]
                    for k in range(q):
                        sat[k] += A_geq[k][jin] - A_geq[k][jout]
                    cur_profit += delta
                    improved = True
                    break  # restart outer scan on improvement

        # Final safety: if our incremental moves drifted, recompute feasibility.
        if not feasible(out):
            return list(xl)
        return out

    # ==================================================================
    # (4) Exact / heavy: ILP
    # ==================================================================
    def ilp_solve_mdmkp(time_limit_s: float = 10.0) -> Optional[List[int]]:
        """Solve the MDMKP exactly via CBC (open-source MILP, via python-mip).

        Variables: x[j] in {0,1}.
        Objective: maximise sum_j cost_vector[j] * x[j].
        Constraints:
          - sum_j A_leq[k][j] * x[j] <= b_leq[k]  for each k in 0..m-1
          - sum_j A_geq[k][j] * x[j] >= b_geq[k]  for each k in 0..q-1

        Returns a 0/1 list of length n (the assignment), or None if the
        solver did not find any feasible solution within `time_limit_s`.
        Primary tool when the instance is small enough; can also be used in
        an LNS loop by warm-starting from a heuristic, fixing a subset of
        bits, and re-running with a short budget."""
        model = Model(sense=MAXIMIZE)
        model.verbose = 0
        model.max_seconds = float(time_limit_s)
        x = [model.add_var(var_type=BINARY, name=f"x_{j}") for j in range(n)]
        model.objective = xsum(float(cost_vector[j]) * x[j] for j in range(n))
        for k in range(m):
            model += (xsum(float(A_leq[k][j]) * x[j] for j in range(n))
                      <= float(b_leq[k])), f"leq_{k}"
        for k in range(q):
            model += (xsum(float(A_geq[k][j]) * x[j] for j in range(n))
                      >= float(b_geq[k])), f"geq_{k}"
        status = model.optimize()
        if status not in (OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE):
            return None
        if model.num_solutions < 1:
            return None
        sol = [0] * n
        for j in range(n):
            val = x[j].x
            if val is not None and val > 0.5:
                sol[j] = 1
        return sol

    return {
        # (1) queries
        "item_profit": item_profit,
        "item_resource_cap": item_resource_cap,
        "item_resource_demand": item_resource_demand,
        "capacity": capacity,
        "demand": demand,
        "n_items": n_items,
        "n_caps": n_caps,
        "n_demands": n_demands,
        # (2) feasibility primitives
        "current_cap_usage": current_cap_usage,
        "current_demand_satisfaction": current_demand_satisfaction,
        "is_caps_satisfied": is_caps_satisfied,
        "is_demands_satisfied": is_demands_satisfied,
        "missing_demands": missing_demands,
        # (3) construction / improvement
        "greedy_profit_within_caps": greedy_profit_within_caps,
        "greedy_for_demand_then_profit": greedy_for_demand_then_profit,
        "repair_for_demands": repair_for_demands,
        "apply_swap_in_out": apply_swap_in_out,
        # (4) exact
        "ilp_solve_mdmkp": ilp_solve_mdmkp,
    }


EXTRA_TOOLS_DESCRIPTION = [
    # ----- (1) Queries -----
    {
        "name": "item_profit",
        "input": "i: int (0-indexed item)",
        "output": "float",
        "purpose": "cost_vector[i]: profit gained by setting x[i] = 1. May be negative when cost_type == 'mixed'.",
    },
    {
        "name": "item_resource_cap",
        "input": "i: int (0-indexed item), k: int (0-indexed <= constraint)",
        "output": "float",
        "purpose": "A_leq[k][i]: how much of the k-th <= resource item i consumes.",
    },
    {
        "name": "item_resource_demand",
        "input": "i: int (0-indexed item), k: int (0-indexed >= constraint)",
        "output": "float",
        "purpose": "A_geq[k][i]: how much item i contributes to the k-th >= demand.",
    },
    {
        "name": "capacity",
        "input": "k: int (0-indexed <= constraint)",
        "output": "float",
        "purpose": "b_leq[k]: RHS (upper bound) of the k-th <= constraint.",
    },
    {
        "name": "demand",
        "input": "k: int (0-indexed >= constraint)",
        "output": "float",
        "purpose": "b_geq[k]: RHS (lower bound) of the k-th >= demand constraint.",
    },
    {
        "name": "n_items",
        "input": "(no args)",
        "output": "int",
        "purpose": "Number of binary decision variables n.",
    },
    {
        "name": "n_caps",
        "input": "(no args)",
        "output": "int",
        "purpose": "Number of <= (capacity) constraints m.",
    },
    {
        "name": "n_demands",
        "input": "(no args)",
        "output": "int",
        "purpose": "Number of active >= (demand) constraints q.",
    },
    # ----- (2) Feasibility primitives -----
    {
        "name": "current_cap_usage",
        "input": "x: list[int] (length n, entries 0/1)",
        "output": "list[float] (length m)",
        "purpose": (
            "Returns usage[k] = sum_j A_leq[k][j] * x[j] for each <= "
            "constraint. The solution is cap-feasible iff usage[k] <= b_leq[k] "
            "for every k. Useful in local search to maintain incremental "
            "bookkeeping."
        ),
    },
    {
        "name": "current_demand_satisfaction",
        "input": "x: list[int] (length n, entries 0/1)",
        "output": "list[float] (length q)",
        "purpose": (
            "Returns sat[k] = sum_j A_geq[k][j] * x[j] for each >= demand. "
            "The solution is demand-feasible iff sat[k] >= b_geq[k] for every "
            "k. Companion to current_cap_usage."
        ),
    },
    {
        "name": "is_caps_satisfied",
        "input": "x: list[int] (length n, entries 0/1)",
        "output": "bool",
        "purpose": (
            "Cheap precheck: True iff every <= constraint is met. Faster than "
            "going through tools['is_feasible'] when you only care about the "
            "capacity side (e.g. inside a greedy that builds up demands "
            "separately)."
        ),
    },
    {
        "name": "is_demands_satisfied",
        "input": "x: list[int] (length n, entries 0/1)",
        "output": "bool",
        "purpose": (
            "Cheap precheck: True iff every >= demand constraint is met. "
            "Pair with is_caps_satisfied for a fast full feasibility check."
        ),
    },
    {
        "name": "missing_demands",
        "input": "x: list[int] (length n, entries 0/1)",
        "output": "list[int]",
        "purpose": (
            "0-indexed indices of the >= constraints that are still not "
            "satisfied. Drives demand-coverage repair / construction loops; "
            "an empty list means all demands are met."
        ),
    },
    # ----- (3) Construction / improvement -----
    {
        "name": "greedy_profit_within_caps",
        "input": "(no args)",
        "output": "list[int] (length n)",
        "purpose": (
            "Pure MKP-style greedy: ignore >= demands and add items in "
            "decreasing profit-per-total-resource ratio while every <= "
            "capacity has room. Items with non-positive profit are skipped. "
            "The result always satisfies all <= constraints but may leave "
            ">= demands unmet -- pipe through repair_for_demands."
        ),
    },
    {
        "name": "greedy_for_demand_then_profit",
        "input": "(no args)",
        "output": "list[int] (length n)",
        "purpose": (
            "Two-phase MDMKP-aware construction: first cover the >= demand "
            "constraints by adding items that maximise "
            "(demand-contribution / cap-usage), then pack remaining capacity "
            "by profit-per-cap-usage. Strong default warm start. Verify with "
            "is_caps_satisfied + is_demands_satisfied; if tight, refine with "
            "apply_swap_in_out or repair_for_demands."
        ),
    },
    {
        "name": "repair_for_demands",
        "input": "x: list[int] (length n)",
        "output": "list[int] (length n)",
        "purpose": (
            "Greedy demand repair: keeps adding items to `x` (one at a time, "
            "preferring those with high demand-contribution per cap-usage) "
            "until either all >= demands are met or no addition is feasible. "
            "Pure function -- input is unchanged. Combine with "
            "greedy_profit_within_caps to upgrade an MKP solution to an "
            "MDMKP-feasible one."
        ),
    },
    {
        "name": "apply_swap_in_out",
        "input": "x: list[int] (length n), t_limit: float = 1.0",
        "output": "list[int] (length n)",
        "purpose": (
            "Best-improvement local search around `x` using 1-flip (add or "
            "remove a single item) and 1-in / 1-out swap moves; only accepts "
            "moves that remain feasible for BOTH <= and >= constraints AND "
            "strictly increase total profit. Runs until no improving move "
            "remains or the wall-clock budget `t_limit` (seconds) is reached. "
            "Falls back to the input if it cannot maintain feasibility. "
            "Pure function -- input is never mutated."
        ),
    },
    # ----- (4) Exact / heavy -----
    {
        "name": "ilp_solve_mdmkp",
        "input": "time_limit_s: float = 10.0",
        "output": "list[int] | None",
        "purpose": (
            "Solve the MDMKP exactly via CBC (open-source MILP through "
            "python-mip) with a wall-clock budget. Variables x[j] in {0,1}, "
            "maximises sum cost_vector[j]*x[j] subject to every <= and >= "
            "constraint. Returns a 0/1 list of length n, or None if no "
            "feasible solution was found within the budget. Primary tool when "
            "the instance fits the budget; can also be used in LNS by "
            "warm-starting from a heuristic, fixing some bits, and re-solving."
        ),
    },
]
