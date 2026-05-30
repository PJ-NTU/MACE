"""Per-problem extras for CO-Bench Bin Packing (one-dimensional).

Provides primitive building blocks so the LLM can compose construction +
repair heuristics for 1-D bin packing without re-deriving FFD/BFD/WFD from
scratch. Tools fall in 4 tiers:

  (1) Queries:        item_size, bin_capacity, num_items, lower_bound
  (2) Feasibility:    bin_load, remaining_capacity, is_bin_feasible,
                      count_bins_used, is_feasible_solution
  (3) Construction /
      improvement:    first_fit_decreasing, best_fit_decreasing,
                      worst_fit_decreasing, find_smallest_bin_that_fits,
                      apply_move_item
  (4) Heavy:          ilp_bin_packing  (exact ILP via CBC)

CO-Bench solution schema (1-BASED item indices!):
    {"num_bins": int, "bins": [[i, j, ...], ...]}
Tools that return a "bins" assignment use the same 1-based convention so
the LLM can plug results straight into the solution dict.

All tools are immutable: they return new lists/dicts and do not mutate
their inputs.
"""
from __future__ import annotations

import time
from typing import Iterable, Optional

from mip import Model, BINARY, MINIMIZE, xsum, OptimizationStatus


# Numerical tolerance for capacity comparisons. Item sizes and capacity can be
# floats in CO-Bench, so an item that "exactly fills" a bin should still be
# accepted. CO-Bench's eval_func uses strict `>` for the overflow check, so any
# small positive epsilon is safe.
_EPS = 1e-9


def extra_tools(instance: dict) -> dict:
    """Factory: returns Bin-Packing tool callables for the loaded instance.

    Instance schema (from CO-Bench load_data):
      - id:           str
      - bin_capacity: float   (capacity C of every bin)
      - num_items:    int     (n)
      - items:        list[float]  (size of each item; 0-indexed in the list,
                                    but CO-Bench solutions use 1-based indices)
      - best_known:   int     (reference value; not for solve)
    """
    capacity = float(instance["bin_capacity"])
    n = int(instance["num_items"])
    sizes = [float(s) for s in instance["items"]]
    if len(sizes) != n:
        # tolerate but trust num_items
        n = len(sizes)

    # Precompute: indices sorted by size desc (FFD/BFD/WFD all want this).
    # We work in 1-based item ids throughout so the output drops straight
    # into CO-Bench's solution dict.
    order_desc = sorted(range(1, n + 1), key=lambda i: -sizes[i - 1])

    # Trivial continuous lower bound on number of bins.
    total_demand = sum(sizes)
    lb_continuous = int(-(-total_demand // capacity)) if capacity > 0 else n  # ceil

    # ==================================================================
    # Helpers (closure-private)
    # ==================================================================
    def _size(i: int) -> float:
        # 1-based access; raises IndexError on bad input which is acceptable
        # since these tools are LLM-facing and bugs should be loud.
        if not (1 <= int(i) <= n):
            raise ValueError(f"item index {i} out of range [1, {n}]")
        return sizes[int(i) - 1]

    def _bin_total(items_in_bin: Iterable[int]) -> float:
        return sum(_size(i) for i in items_in_bin)

    # ==================================================================
    # (1) Queries
    # ==================================================================
    def item_size(i: int) -> float:
        """Size of item `i` (1-based)."""
        return _size(i)

    def bin_capacity() -> float:
        """Capacity C of every bin."""
        return capacity

    def num_items() -> int:
        """Total number of items n."""
        return n

    def lower_bound() -> int:
        """Continuous lower bound on the optimal number of bins:
        ceil(sum(items) / capacity). The true optimum is >= this."""
        return lb_continuous

    # ==================================================================
    # (2) Feasibility primitives
    # ==================================================================
    def bin_load(items_in_bin: Iterable[int]) -> float:
        """Total size of items currently in `items_in_bin` (sum of sizes)."""
        return _bin_total(items_in_bin)

    def remaining_capacity(items_in_bin: Iterable[int]) -> float:
        """capacity - bin_load(items_in_bin). May be negative if the bin is
        infeasible. Use to decide if another item fits."""
        return capacity - _bin_total(items_in_bin)

    def is_bin_feasible(items_in_bin: Iterable[int]) -> bool:
        """True iff the bin's total size does not exceed capacity (with eps).
        Mirrors CO-Bench's strict `>` overflow rule."""
        return _bin_total(items_in_bin) <= capacity + _EPS

    def count_bins_used(solution: dict) -> int:
        """Number of NON-EMPTY bins in `solution['bins']`. Use this to set
        `solution['num_bins']` consistently with the bins list -- CO-Bench
        requires the two to match exactly."""
        bins = solution.get("bins", []) if isinstance(solution, dict) else []
        return sum(1 for b in bins if len(b) > 0)

    def is_feasible_solution(solution: dict) -> tuple[bool, Optional[str]]:
        """Local feasibility check that mirrors CO-Bench's eval_func without
        the framework round-trip. Returns (True, None) or (False, reason).
        Faster than tools['is_feasible'] for tight inner loops."""
        if not isinstance(solution, dict):
            return False, f"solution must be dict, got {type(solution).__name__}"
        bins = solution.get("bins")
        if not isinstance(bins, list):
            return False, "solution['bins'] must be a list"
        nb = solution.get("num_bins", len(bins))
        if nb != len(bins):
            return False, f"num_bins={nb} but len(bins)={len(bins)}"
        counts = [0] * (n + 1)
        for k, b in enumerate(bins, start=1):
            if not isinstance(b, list):
                return False, f"bin {k} is not a list"
            tot = 0.0
            for it in b:
                if not isinstance(it, int) or it < 1 or it > n:
                    return False, f"bin {k} has invalid item index {it}"
                counts[it] += 1
                tot += sizes[it - 1]
            if tot > capacity + _EPS:
                return False, f"bin {k} overflows: {tot} > {capacity}"
        for i in range(1, n + 1):
            if counts[i] != 1:
                return False, f"item {i} appears {counts[i]} times (need 1)"
        return True, None

    # ==================================================================
    # (3) Construction / improvement
    # ==================================================================
    def _pack(order: list[int], rule: str) -> list[list[int]]:
        """Generic packer: iterate `order` (a list of 1-based item ids) and
        place each item into a bin according to `rule`:
          - 'first':  first bin where it fits      (First-Fit)
          - 'best':   bin with smallest remaining cap that still fits  (Best-Fit)
          - 'worst':  bin with largest remaining cap that still fits   (Worst-Fit)
        If no open bin can hold it, open a new one. Items that exceed
        capacity on their own go into singleton bins (CO-Bench will then
        flag the solution infeasible, but that's data-fault, not ours)."""
        bins_items: list[list[int]] = []
        bins_rem: list[float] = []
        for it in order:
            s = sizes[it - 1]
            chosen = -1
            if rule == "first":
                for k, rem in enumerate(bins_rem):
                    if s <= rem + _EPS:
                        chosen = k
                        break
            elif rule == "best":
                best_rem = float("inf")
                for k, rem in enumerate(bins_rem):
                    if s <= rem + _EPS and rem < best_rem:
                        best_rem = rem
                        chosen = k
            elif rule == "worst":
                worst_rem = -1.0
                for k, rem in enumerate(bins_rem):
                    if s <= rem + _EPS and rem > worst_rem:
                        worst_rem = rem
                        chosen = k
            else:
                raise ValueError(f"unknown rule {rule!r}")
            if chosen == -1:
                bins_items.append([it])
                bins_rem.append(capacity - s)
            else:
                bins_items[chosen].append(it)
                bins_rem[chosen] -= s
        return bins_items

    def first_fit_decreasing() -> list[list[int]]:
        """Classic FFD: sort items by size desc, place each into the FIRST
        open bin where it fits, else open a new bin. Returns a list of bins
        (each a list of 1-based item ids). Result is at most ~11/9 * OPT + 1
        bins. Good default warm start. O(n^2)."""
        return _pack(list(order_desc), "first")

    def best_fit_decreasing() -> list[list[int]]:
        """BFD: items in size-desc order, each placed in the bin with the
        SMALLEST remaining capacity that still fits. Same asymptotic bound as
        FFD but often a touch tighter. O(n^2)."""
        return _pack(list(order_desc), "best")

    def worst_fit_decreasing() -> list[list[int]]:
        """WFD: items in size-desc order, each placed in the bin with the
        LARGEST remaining capacity. Spreads load evenly -- usually WORSE
        objective than FFD/BFD but useful for diversification / restarts."""
        return _pack(list(order_desc), "worst")

    def find_smallest_bin_that_fits(item: int, solution: dict) -> int:
        """Index of the open bin (0-based into `solution['bins']`) with the
        smallest remaining capacity that can still accept `item`. Returns -1
        if no current bin fits (caller should open a new bin). Useful for
        repair after a perturbation."""
        s = _size(item)
        bins = solution.get("bins", []) if isinstance(solution, dict) else []
        best = -1
        best_rem = float("inf")
        for k, b in enumerate(bins):
            rem = capacity - _bin_total(b)
            if s <= rem + _EPS and rem < best_rem:
                best_rem = rem
                best = k
        return best

    def apply_move_item(solution: dict, item: int, from_bin: int,
                        to_bin: int) -> Optional[dict]:
        """Move `item` (1-based) from bin index `from_bin` to bin index
        `to_bin` (both 0-based into `solution['bins']`). Returns a NEW
        solution dict if the move is feasible (item is in from_bin AND
        to_bin has room), else None.

        Empty bins are removed automatically and num_bins is updated.
        `to_bin` may equal len(bins) to open a fresh bin."""
        if not isinstance(solution, dict):
            return None
        bins = solution.get("bins")
        if not isinstance(bins, list):
            return None
        nb = len(bins)
        if not (0 <= from_bin < nb):
            return None
        if not (0 <= to_bin <= nb):  # == nb means open new bin
            return None
        src = list(bins[from_bin])
        if item not in src:
            return None
        # destination feasibility
        if to_bin == nb:
            dst_load = 0.0
        else:
            dst_load = _bin_total(bins[to_bin])
        if dst_load + _size(item) > capacity + _EPS:
            return None
        # build new bins list (immutable interface)
        new_bins: list[list[int]] = [list(b) for b in bins]
        new_bins[from_bin].remove(item)
        if to_bin == nb:
            new_bins.append([item])
        else:
            new_bins[to_bin].append(item)
        # drop any empty bins
        new_bins = [b for b in new_bins if len(b) > 0]
        return {"num_bins": len(new_bins), "bins": new_bins}

    # ==================================================================
    # (4) Heavy: exact ILP
    # ==================================================================
    def ilp_bin_packing(time_limit_s: float = 10.0,
                        upper_bound_bins: Optional[int] = None) -> Optional[dict]:
        """Solve the bin-packing ILP exactly (or to the best feasible found
        within `time_limit_s`) with CBC. Returns a solution dict
        {'num_bins': int, 'bins': list[list[int]]} on success, or None if no
        feasible solution was produced.

        Model:
          x[i,b] in {0,1}  item i in bin b      i = 1..n, b = 1..B
          y[b]   in {0,1}  bin b is used
          min  sum_b y[b]
          s.t. sum_b x[i,b] = 1                  for each i
               sum_i s_i * x[i,b] <= C * y[b]    for each b
        Symmetry-breaking: y[1] >= y[2] >= ... and require bin 1 to host
        item 1, so equivalent permutations are pruned.

        B (upper bound on number of bins) defaults to FFD's bin count
        (always feasible and almost always far below n). Heavy: O(n*B)
        variables. Recommend time_limit_s >= 5 for n > 60."""
        if n == 0:
            return {"num_bins": 0, "bins": []}
        # tight upper bound from FFD; gives the solver a feasible warm structure
        ffd_bins = first_fit_decreasing()
        B_default = len(ffd_bins)
        if upper_bound_bins is None:
            B = B_default
        else:
            B = max(1, min(int(upper_bound_bins), n))
            B = min(B, B_default) if B_default > 0 else B
        if B < 1:
            B = max(1, n)

        m = Model(sense=MINIMIZE)
        m.verbose = 0
        m.max_seconds = float(time_limit_s)

        x = {(i, b): m.add_var(var_type=BINARY, name=f"x[{i},{b}]")
             for i in range(1, n + 1) for b in range(1, B + 1)}
        y = {b: m.add_var(var_type=BINARY, name=f"y[{b}]")
             for b in range(1, B + 1)}

        m.objective = xsum(y[b] for b in range(1, B + 1))

        # each item assigned to exactly one bin
        for i in range(1, n + 1):
            m += xsum(x[i, b] for b in range(1, B + 1)) == 1, f"assign_{i}"
        # capacity coupled to y
        for b in range(1, B + 1):
            m += xsum(sizes[i - 1] * x[i, b]
                      for i in range(1, n + 1)) <= capacity * y[b], f"cap_{b}"
        # symmetry breaking: monotone y, and item 1 lives in bin 1
        for b in range(1, B):
            m += y[b] >= y[b + 1], f"sym_{b}"
        if n >= 1:
            m += x[1, 1] == 1, "fix_item_1_in_bin_1"

        status = m.optimize()
        if status not in (OptimizationStatus.OPTIMAL,
                          OptimizationStatus.FEASIBLE):
            return None
        if m.num_solutions < 1:
            return None
        # extract assignment
        result_bins: list[list[int]] = []
        for b in range(1, B + 1):
            members = [i for i in range(1, n + 1)
                       if x[i, b].x is not None and x[i, b].x > 0.5]
            if members:
                result_bins.append(members)
        return {"num_bins": len(result_bins), "bins": result_bins}

    return {
        # (1) queries
        "item_size": item_size,
        "bin_capacity": bin_capacity,
        "num_items": num_items,
        "lower_bound": lower_bound,
        # (2) feasibility primitives
        "bin_load": bin_load,
        "remaining_capacity": remaining_capacity,
        "is_bin_feasible": is_bin_feasible,
        "count_bins_used": count_bins_used,
        "is_feasible_solution": is_feasible_solution,
        # (3) construction / improvement
        "first_fit_decreasing": first_fit_decreasing,
        "best_fit_decreasing": best_fit_decreasing,
        "worst_fit_decreasing": worst_fit_decreasing,
        "find_smallest_bin_that_fits": find_smallest_bin_that_fits,
        "apply_move_item": apply_move_item,
        # (4) heavy
        "ilp_bin_packing": ilp_bin_packing,
    }


EXTRA_TOOLS_DESCRIPTION = [
    # ----- (1) Queries -----
    {
        "name": "item_size",
        "input": "i: int  (1-based item id)",
        "output": "float",
        "purpose": (
            "Size of item `i`. Item indices are 1-BASED throughout this "
            "problem (matches CO-Bench's solution schema). O(1)."
        ),
    },
    {
        "name": "bin_capacity",
        "input": "(no args)",
        "output": "float",
        "purpose": "Capacity C shared by every bin. O(1).",
    },
    {
        "name": "num_items",
        "input": "(no args)",
        "output": "int",
        "purpose": "Total number of items n. O(1).",
    },
    {
        "name": "lower_bound",
        "input": "(no args)",
        "output": "int",
        "purpose": (
            "Continuous lower bound on the optimum: ceil(sum(items)/capacity). "
            "Use to decide whether to invest more compute -- if your current "
            "solution already matches this bound, it is OPTIMAL."
        ),
    },
    # ----- (2) Feasibility primitives -----
    {
        "name": "bin_load",
        "input": "items_in_bin: Iterable[int]  (1-based ids)",
        "output": "float",
        "purpose": "Sum of sizes of items currently in this bin. O(k).",
    },
    {
        "name": "remaining_capacity",
        "input": "items_in_bin: Iterable[int]",
        "output": "float",
        "purpose": (
            "capacity - bin_load. May be negative if the bin is over-full. "
            "Use to decide whether another item still fits."
        ),
    },
    {
        "name": "is_bin_feasible",
        "input": "items_in_bin: Iterable[int]",
        "output": "bool",
        "purpose": (
            "True iff this bin's total size <= capacity (with float tolerance). "
            "Mirrors CO-Bench's strict `>` overflow rule."
        ),
    },
    {
        "name": "count_bins_used",
        "input": "solution: dict",
        "output": "int",
        "purpose": (
            "Number of NON-EMPTY bins in solution['bins']. Use to set "
            "solution['num_bins'] consistently -- CO-Bench requires "
            "num_bins == len(bins) exactly, so always recompute this before "
            "returning a solution."
        ),
    },
    {
        "name": "is_feasible_solution",
        "input": "solution: dict",
        "output": "(bool, str | None)",
        "purpose": (
            "Local feasibility check (same rules as CO-Bench's eval_func but "
            "without the framework round-trip). Faster than tools['is_feasible'] "
            "for tight neighborhood-search loops."
        ),
    },
    # ----- (3) Construction / improvement -----
    {
        "name": "first_fit_decreasing",
        "input": "(no args)",
        "output": "list[list[int]]  (bins, each a list of 1-based item ids)",
        "purpose": (
            "Classic FFD: items sorted by size desc, each placed into the "
            "first bin where it fits, else a new bin opens. <= 11/9 * OPT + 1 "
            "bins. Excellent default warm start. O(n^2). To use as a full "
            "solution: bins = first_fit_decreasing(); "
            "return {'num_bins': len(bins), 'bins': bins}."
        ),
    },
    {
        "name": "best_fit_decreasing",
        "input": "(no args)",
        "output": "list[list[int]]",
        "purpose": (
            "BFD: like FFD but places each item in the bin with the SMALLEST "
            "remaining capacity that still fits. Often slightly better than "
            "FFD on average. O(n^2)."
        ),
    },
    {
        "name": "worst_fit_decreasing",
        "input": "(no args)",
        "output": "list[list[int]]",
        "purpose": (
            "WFD: places each item in the bin with the LARGEST remaining "
            "capacity. Spreads load evenly; usually WORSE than FFD/BFD on "
            "objective but useful for diversification in multi-start / "
            "perturbation-based metaheuristics."
        ),
    },
    {
        "name": "find_smallest_bin_that_fits",
        "input": "item: int, solution: dict",
        "output": "int  (bin index 0-based, or -1 if none fits)",
        "purpose": (
            "Returns the 0-based index of the open bin with the smallest "
            "remaining capacity that can still accept `item`, or -1 if no "
            "open bin can hold it (caller should open a new one). Useful for "
            "repair after removing items in an LNS / ruin-and-recreate step."
        ),
    },
    {
        "name": "apply_move_item",
        "input": ("solution: dict, item: int, from_bin: int (0-based), "
                  "to_bin: int (0-based; equal to len(bins) to open a new bin)"),
        "output": "dict | None",
        "purpose": (
            "Move `item` from `from_bin` to `to_bin` IF the destination still "
            "fits. Returns a NEW solution dict (immutable interface) with "
            "empty bins removed and num_bins refreshed, or None if the move "
            "is infeasible. Use for hill-climbing / VND."
        ),
    },
    # ----- (4) Heavy -----
    {
        "name": "ilp_bin_packing",
        "input": "time_limit_s: float = 10.0, upper_bound_bins: int | None = None",
        "output": "dict | None  ({'num_bins', 'bins'})",
        "purpose": (
            "Solve the bin-packing ILP exactly with CBC, with symmetry "
            "breaking (monotone y[b], item 1 fixed in bin 1) and an upper "
            "bound seeded from FFD. Returns the best feasible solution found "
            "within time_limit_s, or None on failure. EXPENSIVE: variables "
            "scale as O(n*B) where B is the FFD bin count. Recommended only "
            "for n <= ~80 with time_limit_s >= 5; otherwise stick to "
            "first_fit_decreasing / best_fit_decreasing."
        ),
    },
]
