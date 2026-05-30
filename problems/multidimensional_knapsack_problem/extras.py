"""Per-problem extras for Multidimensional Knapsack Problem (MKP).

Provides primitive building blocks so the LLM can compose construction +
repair + local-search heuristics instead of falling back to only the
"all-ILP or all-greedy" extremes. Tools fall in 4 tiers:

  (1) Queries (item / instance metadata):
        item_profit, item_resource, capacity, n_items, n_dims

  (2) Feasibility primitives (work on a 0-indexed selection of item ids):
        current_usage, remaining_capacity, is_within_all_capacities,
        profit_of_selection

  (3) Construction / improvement:
        greedy_by_profit_density, greedy_by_efficiency,
        repair_capacity_violation, apply_local_swap_in_out

  (4) Heavy (exact):
        ilp_solve_mkp

CONVENTIONS:
  - Items are referenced by 0-indexed integer ids in [0, n).
  - A "selection" is any Iterable[int] of item ids (a list, set, or generator
    of indices that the user wants to include). Order is irrelevant.
  - The final solution dict the framework expects is {'x': [0/1] * n}; these
    tools work in selection-set form because that is what most heuristics
    naturally manipulate. Convert with `to_x(selection)` -- not provided here
    since `[1 if i in S else 0 for i in range(n)]` is one line of Python.

NOTE ON CAPACITY DIRECTION:
  MKP is maximization with `<=` resource constraints (resource consumption
  must not exceed capacity). All tools here treat infeasibility as
  "some dim has usage > capacity"; the ILP uses `<=` constraints accordingly.
"""
from __future__ import annotations

import time
from typing import Iterable, Optional

from mip import BINARY, MAXIMIZE, Model, OptimizationStatus, xsum


def extra_tools(instance: dict) -> dict:
    """Factory: returns problem-specific tool callables given the loaded instance.

    Instance schema (from CO-Bench MKP load_data):
      - n: int, number of items (decision variables)
      - m: int, number of resource constraints (knapsack dimensions)
      - p: list[float] of length n, profit coefficients
      - r: list[list[float]] of shape (m, n), resource consumption: r[dim][item]
      - b: list[float] of length m, capacity per resource dimension
    """
    n = int(instance["n"])
    m = int(instance["m"])
    p = [float(x) for x in instance["p"]]
    r = [[float(x) for x in row] for row in instance["r"]]
    b = [float(x) for x in instance["b"]]

    # Precompute per-item "scarcity" score used for greedy_by_profit_density:
    # density_i = p_i / max_m (r[m][i] / b[m]). Items consuming a large share
    # of any tight dim are penalized. Zero-resource items get +inf density.
    def _max_relative_load(i: int) -> float:
        best = 0.0
        for dim in range(m):
            cap = b[dim]
            if cap <= 0:
                # zero-capacity dim: any positive consumption is fatal
                if r[dim][i] > 0:
                    return float("inf")
                continue
            best = max(best, r[dim][i] / cap)
        return best

    _density_score = []
    for i in range(n):
        load = _max_relative_load(i)
        if load <= 0:
            _density_score.append(float("inf") if p[i] > 0 else 0.0)
        else:
            _density_score.append(p[i] / load)

    # Precompute per-item total raw resource consumption across all dims.
    _total_resource = [sum(r[dim][i] for dim in range(m)) for i in range(n)]
    _efficiency_score = []
    for i in range(n):
        tot = _total_resource[i]
        if tot <= 0:
            _efficiency_score.append(float("inf") if p[i] > 0 else 0.0)
        else:
            _efficiency_score.append(p[i] / tot)

    # ==================================================================
    # (1) Queries
    # ==================================================================
    def item_profit(i: int) -> Optional[float]:
        """O(1). Profit p_i of item `i` (0-indexed), or None if out of range."""
        if 0 <= int(i) < n:
            return float(p[int(i)])
        return None

    def item_resource(i: int, dim: int) -> Optional[float]:
        """O(1). Resource consumption r[dim][i] of item `i` on dimension
        `dim`. Both 0-indexed. Returns None if either index is out of range."""
        if 0 <= int(i) < n and 0 <= int(dim) < m:
            return float(r[int(dim)][int(i)])
        return None

    def capacity(dim: int) -> Optional[float]:
        """O(1). Capacity b[dim] of resource dimension `dim` (0-indexed),
        or None if out of range."""
        if 0 <= int(dim) < m:
            return float(b[int(dim)])
        return None

    def n_items() -> int:
        """Number of items (length of `x`). Same as instance['n']."""
        return n

    def n_dims() -> int:
        """Number of resource constraint dimensions. Same as instance['m']."""
        return m

    # ==================================================================
    # (2) Feasibility primitives
    # ==================================================================
    def _normalize_selection(selected: Iterable[int]) -> list:
        out = []
        seen = set()
        for i in selected:
            ii = int(i)
            if 0 <= ii < n and ii not in seen:
                seen.add(ii)
                out.append(ii)
        return out

    def current_usage(selected: Iterable[int], dim: int) -> Optional[float]:
        """Total resource consumption on dimension `dim` (0-indexed) for the
        items in `selected`. Returns None if `dim` is out of range."""
        if not (0 <= int(dim) < m):
            return None
        d = int(dim)
        sel = _normalize_selection(selected)
        return float(sum(r[d][i] for i in sel))

    def remaining_capacity(selected: Iterable[int], dim: int) -> Optional[float]:
        """Remaining capacity on dimension `dim` after taking `selected`:
        b[dim] - sum_{i in selected} r[dim][i]. May be negative if the
        selection already over-uses the dimension. Returns None if `dim`
        is out of range."""
        if not (0 <= int(dim) < m):
            return None
        d = int(dim)
        sel = _normalize_selection(selected)
        return float(b[d] - sum(r[d][i] for i in sel))

    def is_within_all_capacities(selected: Iterable[int], tol: float = 1e-6) -> bool:
        """True iff for every dimension d, sum_{i in selected} r[d][i] <= b[d] + tol.
        This is exactly the feasibility test CO-Bench's eval_func performs
        (with the same default tolerance)."""
        sel = _normalize_selection(selected)
        for d in range(m):
            usage = sum(r[d][i] for i in sel)
            if usage - b[d] > tol:
                return False
        return True

    def profit_of_selection(selected: Iterable[int]) -> float:
        """Total profit sum(p[i] for i in selected). NO feasibility check --
        the value is just the sum; pair with `is_within_all_capacities` if
        you also need to know whether it's a valid solution."""
        sel = _normalize_selection(selected)
        return float(sum(p[i] for i in sel))

    # ==================================================================
    # (3) Construction / improvement heuristics
    # ==================================================================
    def greedy_by_profit_density() -> list:
        """Greedy construction: sort items by p_i / max_m(r[m][i] / b[m])
        DESCENDING (items giving the most profit per unit of the tightest
        dim get picked first), then take each item if it still fits in
        EVERY dimension. Returns a sorted list of 0-indexed item ids
        forming a feasible selection. O(n log n + n*m)."""
        order = sorted(range(n), key=lambda i: -_density_score[i])
        used = [0.0] * m
        chosen = []
        for i in order:
            ok = True
            for d in range(m):
                if used[d] + r[d][i] > b[d] + 1e-9:
                    ok = False
                    break
            if ok:
                chosen.append(i)
                for d in range(m):
                    used[d] += r[d][i]
        return sorted(chosen)

    def greedy_by_efficiency() -> list:
        """Greedy construction: sort items by p_i / sum_m r[m][i] DESCENDING
        (profit per unit of total resource used across all dims), then take
        each item if it still fits in EVERY dimension. Complementary to
        `greedy_by_profit_density` -- they often disagree on tight instances,
        so running both and keeping the better is cheap diversification."""
        order = sorted(range(n), key=lambda i: -_efficiency_score[i])
        used = [0.0] * m
        chosen = []
        for i in order:
            ok = True
            for d in range(m):
                if used[d] + r[d][i] > b[d] + 1e-9:
                    ok = False
                    break
            if ok:
                chosen.append(i)
                for d in range(m):
                    used[d] += r[d][i]
        return sorted(chosen)

    def repair_capacity_violation(selected: Iterable[int]) -> list:
        """Make `selected` feasible by repeatedly removing the item with the
        LOWEST efficiency score (p_i / sum_m r[m][i]) until every dimension
        is within capacity. Returns a new sorted list of 0-indexed ids.
        Useful after a perturbation that may over-commit some dimension.
        If the input is already feasible, returns a deduplicated copy."""
        sel = _normalize_selection(selected)
        # Sort once by efficiency ASCENDING -- worst-first removal.
        sel.sort(key=lambda i: _efficiency_score[i])
        used = [0.0] * m
        for i in sel:
            for d in range(m):
                used[d] += r[d][i]
        idx = 0
        while idx < len(sel):
            # Check if any dim is over capacity.
            over = False
            for d in range(m):
                if used[d] - b[d] > 1e-6:
                    over = True
                    break
            if not over:
                break
            victim = sel[idx]
            for d in range(m):
                used[d] -= r[d][victim]
            sel[idx] = None  # mark removed
            idx += 1
        return sorted(i for i in sel if i is not None)

    def apply_local_swap_in_out(
        selected: Iterable[int],
        time_limit_s: float = 2.0,
    ) -> list:
        """Pairwise swap local search: try replacing one IN item with one OUT
        item (1-out / 1-in swap) whenever the swap is feasible AND strictly
        improves total profit. First-improvement; restarts from scratch after
        each accepted swap; stops at local optimum or when `time_limit_s`
        elapses. Also tries pure ADDITION (add an out-item that still fits)
        as a 0-out / 1-in special case. Returns the new selection (sorted).

        Input must be FEASIBLE (no capacity violation) -- otherwise the
        swap test cannot be evaluated correctly. Call
        `repair_capacity_violation` first if unsure."""
        sel_set = set(_normalize_selection(selected))
        used = [0.0] * m
        for i in sel_set:
            for d in range(m):
                used[d] += r[d][i]
        # Sanity: if input is infeasible, bail out with a best-effort repair
        # so callers don't get stuck in undefined behavior.
        for d in range(m):
            if used[d] - b[d] > 1e-6:
                return repair_capacity_violation(sel_set)

        t0 = time.time()
        safety = 0.02
        deadline = float(time_limit_s) - safety

        improved = True
        while improved:
            improved = False
            if time.time() - t0 >= deadline:
                break

            outside = [i for i in range(n) if i not in sel_set]

            # First: pure additions (0-out / 1-in).
            for i in outside:
                if p[i] <= 0:
                    continue
                if all(used[d] + r[d][i] <= b[d] + 1e-9 for d in range(m)):
                    sel_set.add(i)
                    for d in range(m):
                        used[d] += r[d][i]
                    improved = True
                    break
            if improved:
                continue

            # Then: 1-out / 1-in swaps with positive net profit.
            done = False
            inside = list(sel_set)
            for j in inside:
                if time.time() - t0 >= deadline:
                    done = True
                    break
                pj = p[j]
                for i in outside:
                    if i in sel_set:  # may have been added in this pass
                        continue
                    if p[i] - pj <= 1e-12:
                        continue
                    feasible = True
                    for d in range(m):
                        if used[d] - r[d][j] + r[d][i] > b[d] + 1e-9:
                            feasible = False
                            break
                    if feasible:
                        sel_set.discard(j)
                        sel_set.add(i)
                        for d in range(m):
                            used[d] += r[d][i] - r[d][j]
                        improved = True
                        done = True
                        break
                if done:
                    break
        return sorted(sel_set)

    # ==================================================================
    # (4) Heavy: exact ILP
    # ==================================================================
    def ilp_solve_mkp(
        must_include: Optional[Iterable[int]] = None,
        must_exclude: Optional[Iterable[int]] = None,
        time_limit_s: float = 10.0,
    ):
        """Solve the FULL Multidimensional Knapsack ILP with CBC under a wall
        clock budget. Returns the best feasible selection found as a sorted
        list of 0-indexed item ids (optimal if CBC finished in time) or None
        on failure / no feasible solution. Use `must_include` / `must_exclude`
        to fix variables for LNS-style refinement around a warm start."""
        try:
            tl = max(0.5, float(time_limit_s))
        except Exception:
            tl = 10.0
        mi = {int(i) for i in (must_include or []) if 0 <= int(i) < n}
        me = {int(i) for i in (must_exclude or []) if 0 <= int(i) < n}

        mdl = Model(sense=MAXIMIZE)
        mdl.verbose = 0
        mdl.max_seconds = tl

        x = {i: mdl.add_var(var_type=BINARY, name=f"x[{i}]") for i in range(n)}
        mdl.objective = xsum(p[i] * x[i] for i in range(n))

        for d in range(m):
            mdl += xsum(r[d][i] * x[i] for i in range(n)) <= b[d], f"cap_{d}"

        for i in mi:
            mdl += x[i] == 1, f"force_in_{i}"
        for i in me:
            mdl += x[i] == 0, f"force_out_{i}"

        status = mdl.optimize()
        if status not in (
            OptimizationStatus.OPTIMAL,
            OptimizationStatus.FEASIBLE,
        ):
            return None
        if mdl.num_solutions < 1:
            return None
        return sorted(i for i in x if x[i].x is not None and x[i].x > 0.5)

    return {
        # (1) queries
        "item_profit": item_profit,
        "item_resource": item_resource,
        "capacity": capacity,
        "n_items": n_items,
        "n_dims": n_dims,
        # (2) feasibility primitives
        "current_usage": current_usage,
        "remaining_capacity": remaining_capacity,
        "is_within_all_capacities": is_within_all_capacities,
        "profit_of_selection": profit_of_selection,
        # (3) construction / improvement
        "greedy_by_profit_density": greedy_by_profit_density,
        "greedy_by_efficiency": greedy_by_efficiency,
        "repair_capacity_violation": repair_capacity_violation,
        "apply_local_swap_in_out": apply_local_swap_in_out,
        # (4) heavy
        "ilp_solve_mkp": ilp_solve_mkp,
    }


EXTRA_TOOLS_DESCRIPTION = [
    # ----- (1) Queries -----
    {
        "name": "item_profit",
        "input": "i: int",
        "output": "float | None",
        "purpose": (
            "O(1). Profit p_i of item `i` (0-indexed in [0, n)). Returns None "
            "if `i` is out of range. Use to score candidate moves without "
            "indexing instance['p'] yourself."
        ),
    },
    {
        "name": "item_resource",
        "input": "i: int, dim: int",
        "output": "float | None",
        "purpose": (
            "O(1). Resource consumption r[dim][i] of item `i` on dimension "
            "`dim` (both 0-indexed). Returns None if either is out of range."
        ),
    },
    {
        "name": "capacity",
        "input": "dim: int",
        "output": "float | None",
        "purpose": (
            "O(1). Capacity b[dim] of resource dimension `dim` (0-indexed), "
            "or None if out of range."
        ),
    },
    {
        "name": "n_items",
        "input": "(no args)",
        "output": "int",
        "purpose": "Number of items (length the final `x` list must have).",
    },
    {
        "name": "n_dims",
        "input": "(no args)",
        "output": "int",
        "purpose": "Number of resource constraint dimensions m.",
    },
    # ----- (2) Feasibility primitives -----
    {
        "name": "current_usage",
        "input": "selected: Iterable[int], dim: int",
        "output": "float | None",
        "purpose": (
            "Total resource consumption on dimension `dim` for the items in "
            "`selected` (0-indexed item ids). Returns None if `dim` is out of "
            "range. Use to check headroom on a single dim cheaply."
        ),
    },
    {
        "name": "remaining_capacity",
        "input": "selected: Iterable[int], dim: int",
        "output": "float | None",
        "purpose": (
            "b[dim] - current_usage(selected, dim). May be negative if the "
            "selection over-uses the dim. Returns None if `dim` is out of "
            "range. Useful as a 'can I still add an item that costs X here?' "
            "test."
        ),
    },
    {
        "name": "is_within_all_capacities",
        "input": "selected: Iterable[int], tol: float = 1e-6",
        "output": "bool",
        "purpose": (
            "True iff every dim's usage is <= its capacity (within `tol`). "
            "Mirrors CO-Bench's eval_func feasibility test exactly. Cheaper "
            "than calling tools['is_feasible'] because it skips dict packing."
        ),
    },
    {
        "name": "profit_of_selection",
        "input": "selected: Iterable[int]",
        "output": "float",
        "purpose": (
            "Sum of p_i over items in `selected`. NO feasibility check; "
            "pair with `is_within_all_capacities` if you need both."
        ),
    },
    # ----- (3) Construction / improvement -----
    {
        "name": "greedy_by_profit_density",
        "input": "(no args)",
        "output": "list[int]",
        "purpose": (
            "Greedy: sort items by p_i / max_d(r[d][i] / b[d]) descending "
            "(profit per unit of the TIGHTEST dim), then take each if it "
            "fits in every dim. Returns a feasible selection (0-indexed ids). "
            "Strong on tight instances where one dim dominates. O(n log n + n*m)."
        ),
    },
    {
        "name": "greedy_by_efficiency",
        "input": "(no args)",
        "output": "list[int]",
        "purpose": (
            "Greedy: sort items by p_i / sum_d r[d][i] descending (profit per "
            "unit of TOTAL resource), then take each if it fits in every dim. "
            "Complementary to `greedy_by_profit_density` -- run both and keep "
            "the better selection as a warm start."
        ),
    },
    {
        "name": "repair_capacity_violation",
        "input": "selected: Iterable[int]",
        "output": "list[int]",
        "purpose": (
            "Make `selected` feasible by removing items in ASCENDING order of "
            "efficiency (p_i / sum_d r[d][i]) until every dim is within "
            "capacity. Idempotent on feasible inputs. Use after a perturbation "
            "/ destroy step in LNS."
        ),
    },
    {
        "name": "apply_local_swap_in_out",
        "input": "selected: Iterable[int], time_limit_s: float = 2.0",
        "output": "list[int]",
        "purpose": (
            "Local search: greedy ADDITIONS (add a fitting out-item with "
            "positive profit) plus 1-out / 1-in SWAPS that strictly improve "
            "profit while remaining feasible. First-improvement; loops until "
            "local optimum or `time_limit_s` elapses. Input MUST be feasible "
            "(else the function auto-repairs and returns). Returns the new "
            "selection sorted."
        ),
    },
    # ----- (4) Heavy -----
    {
        "name": "ilp_solve_mkp",
        "input": (
            "must_include: Iterable[int] = None, "
            "must_exclude: Iterable[int] = None, "
            "time_limit_s: float = 10.0"
        ),
        "output": "list[int] | None",
        "purpose": (
            "Solve the FULL MKP ILP exactly with CBC under a wall-clock "
            "budget. Returns the best feasible selection found as a sorted "
            "list of 0-indexed item ids (optimal if it finished in time) "
            "or None on failure. Use `must_include` / `must_exclude` to fix "
            "variables for LNS-style refinement around a warm start."
        ),
    },
]
