"""Per-problem extras for Set Covering.

Provides building-block tools so the LLM can compose construction / repair /
LNS heuristics instead of only the "all-ILP or all-greedy" extremes. Tools
fall in 4 tiers:

  (1) Queries:      column_cost, column_covers, columns_covering_row
  (2) Feasibility:  covered_rows, uncovered_rows, cost_of_selection,
                    is_full_cover
  (3) Construction/improvement:
                    greedy_cover_by_cost_ratio, remove_redundant,
                    cheapest_column_covering_row
  (4) Heavy:        ilp_solve_cover

KEY DIFFERENCE vs Set Partitioning:
  Row coverage constraint is `>= 1` (at-least-once), NOT `== 1`. Over-cover
  is allowed and frequently useful. This means:
    * No conflict_rows / is_conflict_free notion -- nothing conflicts.
    * No complete_partial_via_ilp -- any partial can always be extended
      (assuming the instance itself is coverable).
    * `remove_redundant` (column removal local search) is meaningful: a
      column whose every row is also covered by another selected column
      can be dropped for free.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Optional

from mip import BINARY, MINIMIZE, Model, OptimizationStatus, xsum


def extra_tools(instance: dict) -> dict:
    """Factory: returns problem-specific tool callables for a loaded instance.

    Instance schema (from CO-Bench Set Covering load_data):
      - m:         int, number of rows
      - n:         int, number of columns
      - costs:     list[int] of length n, costs[j] = cost of column (j+1)
      - row_cover: list[list[int]] of length m, row_cover[i] = 1-indexed
                   column ids that cover row (i+1)
    """
    m = int(instance["m"])
    n = int(instance["n"])
    costs = list(instance["costs"])
    row_cover = instance["row_cover"]

    # --- Precompute column -> set of rows it covers (1-indexed both sides) ---
    col_to_rows: dict[int, set[int]] = defaultdict(set)
    for i, cols in enumerate(row_cover):
        r = i + 1
        for c in cols:
            col_to_rows[int(c)].add(r)

    # row_to_cols is the inverse map (already given by `row_cover`, but cached
    # in dict form for O(1) lookup by 1-indexed row id).
    row_to_cols: dict[int, list[int]] = {
        i + 1: list(row_cover[i]) for i in range(m)
    }

    all_col_ids: list[int] = sorted(set(col_to_rows.keys()) | set(range(1, n + 1)))

    def _col_cost(c: int) -> Optional[int]:
        if 1 <= int(c) <= n:
            return int(costs[int(c) - 1])
        return None

    # ==================================================================
    # (1) Queries
    # ==================================================================
    def column_cost(c: int):
        """O(1). Cost of column `c` (1-indexed) or None if out of range."""
        return _col_cost(c)

    def column_covers(c: int) -> set:
        """O(1). Set of rows (1-indexed) covered by column `c`. Empty if
        `c` is out of range or covers no rows."""
        return set(col_to_rows.get(int(c), set()))

    def columns_covering_row(r: int) -> list:
        """O(1). List of 1-indexed column ids that cover row `r`."""
        if not (1 <= int(r) <= m):
            return []
        return list(row_to_cols.get(int(r), []))

    # ==================================================================
    # (2) Feasibility primitives
    # ==================================================================
    def _coverage_counts(selected: Iterable[int]) -> dict:
        cnt: dict[int, int] = defaultdict(int)
        for c in selected:
            for r in col_to_rows.get(int(c), ()):  # noqa: PERF102
                cnt[r] += 1
        return cnt

    def covered_rows(selected: Iterable[int]) -> set:
        """O(sum |col|). Rows (1-indexed) covered AT LEAST ONCE by the
        selection. Over-cover is fine in set covering -- this returns the
        full coverage set, not the 'exactly once' set."""
        cnt = _coverage_counts(selected)
        return {r for r, k in cnt.items() if k >= 1}

    def uncovered_rows(selected: Iterable[int]) -> set:
        """O(sum |col| + m). Rows (1-indexed) NOT covered by `selected`.
        Empty iff the selection is feasible."""
        cnt = _coverage_counts(selected)
        return {r for r in range(1, m + 1) if cnt.get(r, 0) == 0}

    def cost_of_selection(selected: Iterable[int]) -> int:
        """O(|selected|). Total cost of the selection. NO feasibility check
        -- the value is just the sum; pair with `is_full_cover` if you
        also need to know whether it's a valid solution."""
        total = 0
        for c in selected:
            v = _col_cost(c)
            if v is not None:
                total += v
        return int(total)

    def is_full_cover(selected: Iterable[int]) -> bool:
        """O(sum |col| + m). True iff every row 1..m is covered by at least
        one column in `selected`."""
        return len(uncovered_rows(selected)) == 0

    # ==================================================================
    # (3) Construction / improvement heuristics
    # ==================================================================
    def greedy_cover_by_cost_ratio() -> list:
        """Cost-effective greedy (Chvatal): iteratively pick the column
        with the smallest cost / new-rows-covered. Returns a FULL cover
        (a list of 1-indexed column ids) if one exists. Over-cover is
        permitted -- no column is dropped just because some of its rows
        are already covered. O(n * m) in the worst case.

        Use as a warm start for local search or as an upper-bound seed
        for `ilp_solve_cover`."""
        uncovered: set[int] = set(range(1, m + 1))
        chosen: list[int] = []
        candidates: set[int] = {c for c in all_col_ids if col_to_rows.get(c)}
        while uncovered and candidates:
            best = None
            best_ratio = float("inf")
            best_new = 0
            for c in candidates:
                new = len(col_to_rows[c] & uncovered)
                if new == 0:
                    continue
                cost = _col_cost(c)
                if cost is None:
                    continue
                # Cost 0 column that covers anything is always best.
                ratio = (cost / new) if cost > 0 else -1.0
                if ratio < best_ratio or (
                    ratio == best_ratio and new > best_new
                ):
                    best_ratio = ratio
                    best = c
                    best_new = new
            if best is None:
                break
            chosen.append(int(best))
            uncovered -= col_to_rows[best]
            candidates.discard(best)
        return sorted(chosen)

    def remove_redundant(selected: Iterable[int]) -> list:
        """Local improvement specific to set covering: drop any column
        whose rows are ALL still covered by the remaining selected
        columns. Processes columns from most-expensive to cheapest so
        big wins are realized first. Returns a new sorted list of
        1-indexed columns; never raises. If the input wasn't a full
        cover to begin with, the output won't be either (but no
        previously-covered row becomes uncovered)."""
        sel = [int(c) for c in selected if int(c) in col_to_rows or 1 <= int(c) <= n]
        # Unique + sort by cost descending so we try to drop expensive ones first.
        sel = sorted(set(sel), key=lambda c: (-(_col_cost(c) or 0), c))
        kept: list[int] = list(sel)
        # Coverage count from current `kept`.
        cnt = _coverage_counts(kept)
        for c in sel:
            rows_c = col_to_rows.get(c, set())
            # Removing c is safe iff every row in rows_c still has count >= 2.
            if rows_c and all(cnt.get(r, 0) >= 2 for r in rows_c):
                kept.remove(c)
                for r in rows_c:
                    cnt[r] -= 1
            elif not rows_c:
                # column covers nothing -- always safe to drop
                kept.remove(c)
        return sorted(kept)

    def cheapest_column_covering_row(
        row: int,
        exclude: Optional[Iterable[int]] = None,
    ):
        """Among columns covering `row` (excluding ids in `exclude`),
        return the one with the smallest cost. Returns None if no such
        column exists. Useful for repair: 'row r is uncovered -- find
        me the cheapest column to add'."""
        if not (1 <= int(row) <= m):
            return None
        ex = set(int(c) for c in exclude) if exclude else set()
        best = None
        best_cost = float("inf")
        for c in row_to_cols.get(int(row), []):
            if c in ex:
                continue
            cost = _col_cost(c)
            if cost is None:
                continue
            if cost < best_cost:
                best_cost = cost
                best = c
        return best

    # ==================================================================
    # (4) Heavy: exact ILP
    # ==================================================================
    def ilp_solve_cover(
        must_include: Optional[Iterable[int]] = None,
        must_exclude: Optional[Iterable[int]] = None,
        time_limit_s: float = 10.0,
    ):
        """Solve the FULL Set Covering ILP exactly with CBC (open-source).
        Returns `selected_columns` (sorted list of 1-indexed ids) for the
        best solution found within `time_limit_s`, or None if no feasible
        solution was obtained.

        Use `must_include` / `must_exclude` to fix variables for LNS-style
        refinement around a known good solution."""
        try:
            tl = max(0.5, float(time_limit_s))
        except Exception:
            tl = 10.0
        mi = {int(c) for c in (must_include or [])}
        me = {int(c) for c in (must_exclude or [])}

        mdl = Model(sense=MINIMIZE)
        mdl.verbose = 0
        mdl.max_seconds = tl

        col_ids = [c for c in range(1, n + 1)]
        x = {c: mdl.add_var(var_type=BINARY, name=f"x[{c}]") for c in col_ids}
        mdl.objective = xsum(int(costs[c - 1]) * x[c] for c in col_ids)

        for r in range(1, m + 1):
            covers = row_to_cols.get(r, [])
            if not covers:
                # row that no column covers -> instance is infeasible
                return None
            mdl += xsum(x[c] for c in covers) >= 1, f"row_{r}"

        for c in mi:
            if c in x:
                mdl += x[c] == 1, f"force_in_{c}"
        for c in me:
            if c in x:
                mdl += x[c] == 0, f"force_out_{c}"

        status = mdl.optimize()
        if status not in (
            OptimizationStatus.OPTIMAL,
            OptimizationStatus.FEASIBLE,
        ):
            return None
        if mdl.num_solutions < 1:
            return None
        return sorted(c for c in x if x[c].x is not None and x[c].x > 0.5)

    # ==================================================================
    # (5) Solution-dict builder + one-shot solver
    # ==================================================================
    def make_solution(selected_columns: Iterable[int]) -> dict:
        """Wrap a list of 1-indexed column ids into the EXACT dict shape
        eval_func expects: {'selected_columns': list[int]}. Use this on the
        output of ilp_solve_cover() / greedy_cover_by_cost_ratio() /
        remove_redundant() so you never return the wrong dict shape."""
        cols = sorted({int(c) for c in selected_columns if 1 <= int(c) <= n})
        return {"selected_columns": cols}

    def solve_default(time_limit_s: float = 10.0) -> dict:
        """ONE-SHOT STRONG SOLVER. Returns the complete solution dict
        {'selected_columns': list[int]} ready to return directly.

        Strategy: first try ilp_solve_cover (exact CBC). If the ILP fails
        within time_limit_s, fall back to greedy_cover_by_cost_ratio +
        remove_redundant. Always returns a feasible full cover (assuming
        the instance is coverable).

        Use as the FIRST thing your solve() function calls. ONE LINE:
            return tools['solve_default'](time_limit_s=10)
        """
        cols = ilp_solve_cover(time_limit_s=time_limit_s)
        if cols is None or not cols:
            cols = greedy_cover_by_cost_ratio()
            cols = remove_redundant(cols)
        return make_solution(cols)

    return {
        # (5) one-shot + builder (CALL FIRST)
        "solve_default": solve_default,
        "make_solution": make_solution,
        # (4) heavy
        "ilp_solve_cover": ilp_solve_cover,
        # (3) construction / improvement
        "greedy_cover_by_cost_ratio": greedy_cover_by_cost_ratio,
        "remove_redundant": remove_redundant,
        "cheapest_column_covering_row": cheapest_column_covering_row,
        # (2) feasibility primitives
        "covered_rows": covered_rows,
        "uncovered_rows": uncovered_rows,
        "cost_of_selection": cost_of_selection,
        "is_full_cover": is_full_cover,
        # (1) queries
        "column_cost": column_cost,
        "column_covers": column_covers,
        "columns_covering_row": columns_covering_row,
    }


EXTRA_TOOLS_DESCRIPTION = [
    # ----- (5) ONE-SHOT STRONG SOLVER (call this first!) -----
    {
        "name": "solve_default",
        "input": "time_limit_s: float = 10.0",
        "output": "dict {'selected_columns': list[int]}",
        "purpose": (
            "RECOMMENDED START: returns a complete solution dict ready to return "
            "directly. Tries ilp_solve_cover first (exact CBC); on failure falls "
            "back to greedy_cover_by_cost_ratio + remove_redundant. Always yields "
            "a feasible full cover. ONE LINE: "
            "`return tools['solve_default'](time_limit_s=10)`."
        ),
    },
    {
        "name": "make_solution",
        "input": "selected_columns: Iterable[int] (1-indexed)",
        "output": "dict {'selected_columns': list[int]}",
        "purpose": (
            "Build the EXACT solution dict shape eval_func wants from a list of "
            "1-indexed column ids. Use on the output of ilp_solve_cover() / "
            "greedy_cover_by_cost_ratio() / remove_redundant() so you never "
            "return the wrong dict shape."
        ),
    },
    # ----- (4) Heavy: exact ILP -----
    {
        "name": "ilp_solve_cover",
        "input": (
            "must_include: Iterable[int] = None, "
            "must_exclude: Iterable[int] = None, "
            "time_limit_s: float = 10.0"
        ),
        "output": "list[int] | None",
        "purpose": (
            "Use as primary solver. Solves the FULL Set Covering ILP exactly with "
            "CBC under a wall-clock budget. Returns the optimal selected_columns "
            "(or best-found within time_limit_s), or None if no feasible solution "
            "exists. Wrap with make_solution() to get the ready-to-return dict, "
            "or just call solve_default() instead. Use `must_include` / "
            "`must_exclude` for LNS-style refinement around a known good solution."
        ),
    },
    # ----- (1) Queries -----
    {
        "name": "column_cost",
        "input": "c: int",
        "output": "int | None",
        "purpose": (
            "O(1). Cost of column `c` (1-indexed) or None if `c` is out of "
            "range. Use to compare candidates without iterating the cost list."
        ),
    },
    {
        "name": "column_covers",
        "input": "c: int",
        "output": "set[int]",
        "purpose": (
            "O(1). Set of rows (1-indexed) covered by column `c`. Empty if "
            "`c` is out of range or covers no rows."
        ),
    },
    {
        "name": "columns_covering_row",
        "input": "r: int",
        "output": "list[int]",
        "purpose": (
            "O(1). All column ids (1-indexed) that cover row `r`. "
            "Precomputed inverse map; cheaper than scanning `row_cover` "
            "in your code."
        ),
    },
    # ----- (2) Feasibility primitives -----
    {
        "name": "covered_rows",
        "input": "selected: Iterable[int]",
        "output": "set[int]",
        "purpose": (
            "O(sum |col|). Rows covered AT LEAST ONCE by the selection. "
            "Over-cover is allowed in set covering, so this is the proper "
            "'has been covered' set (not 'covered exactly once')."
        ),
    },
    {
        "name": "uncovered_rows",
        "input": "selected: Iterable[int]",
        "output": "set[int]",
        "purpose": (
            "O(sum |col| + m). Rows the selection has NOT covered. Empty "
            "iff the selection is feasible -- use this in repair loops."
        ),
    },
    {
        "name": "cost_of_selection",
        "input": "selected: Iterable[int]",
        "output": "int",
        "purpose": (
            "O(|selected|). Total cost of the columns in `selected`. No "
            "feasibility check; pair with `is_full_cover` when you need both."
        ),
    },
    {
        "name": "is_full_cover",
        "input": "selected: Iterable[int]",
        "output": "bool",
        "purpose": (
            "True iff every row 1..m is covered by at least one column in "
            "`selected`. Equivalent to len(uncovered_rows(selected)) == 0 "
            "but slightly cheaper -- short-circuits as soon as a hole is found."
        ),
    },
    # ----- (3) Construction / improvement -----
    {
        "name": "greedy_cover_by_cost_ratio",
        "input": "(no args)",
        "output": "list[int]",
        "purpose": (
            "Chvatal cost-effective greedy: repeatedly pick the column "
            "with the smallest cost / new-rows-covered until every row "
            "is covered. O(n*m) worst case. Returns a 1-indexed selection. "
            "Often within a small factor of optimal; great warm start for "
            "`remove_redundant` and `ilp_solve_cover`."
        ),
    },
    {
        "name": "remove_redundant",
        "input": "selected: Iterable[int]",
        "output": "list[int]",
        "purpose": (
            "Local improvement specific to set covering: drop any column "
            "whose rows are still covered by the remaining selection. "
            "Tries expensive columns first for bigger gains. Idempotent. "
            "Returns a new sorted list; never makes a covered row uncovered."
        ),
    },
    {
        "name": "cheapest_column_covering_row",
        "input": "row: int, exclude: Iterable[int] = None",
        "output": "int | None",
        "purpose": (
            "Cheapest column covering `row`, ignoring any ids in `exclude`. "
            "Returns None if no candidate. Standard repair move: when a row "
            "is uncovered, add the cheapest column that fixes it."
        ),
    },
]
