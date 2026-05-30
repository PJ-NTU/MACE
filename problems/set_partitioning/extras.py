"""Per-problem extras for Set Partitioning.

Provides a rich set of building-block tools so the LLM can compose
construction / repair / LNS heuristics rather than only the "all-ILP or
all-greedy" extremes. Tools fall in 4 groups:

  (1) Exact / heavy: ilp_solve_partition, complete_partial_via_ilp
  (2) Construction: greedy_cover_by_cost_ratio, cheapest_column_covering_row
  (3) Inspection:   columns_covering_row, column_rows, column_cost,
                    covered_rows, uncovered_rows, conflict_rows, cost_of_selection
  (4) Validation:   is_conflict_free, feasible_columns_given_partial

All are exposed under tools[...] and described in EXTRA_TOOLS_DESCRIPTION
for the LLM-facing prompt.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Optional, Iterable

from mip import Model, BINARY, MINIMIZE, xsum, OptimizationStatus


def extra_tools(instance: dict) -> dict:
    """Factory: returns problem-specific tool callables given the loaded instance."""
    num_rows = instance["num_rows"]
    num_columns = instance["num_columns"]
    columns_info = instance["columns_info"]

    # Precompute inverse map row -> [cols that cover it] once per instance
    row_to_cols: dict[int, list[int]] = defaultdict(list)
    for c, (_cost, rows) in columns_info.items():
        for r in rows:
            row_to_cols[r].append(c)

    # ==================================================================
    # (1) Exact / heavy
    # ==================================================================
    def _build_ilp(must_include: set, must_exclude: set, time_limit_s: float):
        m = Model(sense=MINIMIZE)
        m.verbose = 0
        m.max_seconds = float(time_limit_s)
        col_ids = sorted(columns_info.keys())
        x = {c: m.add_var(var_type=BINARY, name=f"x[{c}]") for c in col_ids}
        m.objective = xsum(int(columns_info[c][0]) * x[c] for c in col_ids)
        for r in range(1, num_rows + 1):
            covers = row_to_cols.get(r, [])
            if not covers:
                return None, None
            m += xsum(x[c] for c in covers) == 1, f"row_{r}"
        for c in must_include:
            if c in x:
                m += x[c] == 1, f"force_in_{c}"
        for c in must_exclude:
            if c in x:
                m += x[c] == 0, f"force_out_{c}"
        return m, x

    def ilp_solve_partition(
        must_include: Optional[Iterable[int]] = None,
        must_exclude: Optional[Iterable[int]] = None,
        time_limit_s: float = 10.0,
    ):
        mi = set(must_include) if must_include else set()
        me = set(must_exclude) if must_exclude else set()
        m, x = _build_ilp(mi, me, time_limit_s)
        if m is None:
            return None
        status = m.optimize()
        if status not in (OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE):
            return None
        if m.num_solutions < 1:
            return None
        return sorted(c for c in x if x[c].x is not None and x[c].x > 0.5)

    def complete_partial_via_ilp(
        partial: Iterable[int],
        time_limit_s: float = 10.0,
    ):
        """Take a conflict-free partial selection and extend it via ILP to a
        full feasible partition (if one exists with that prefix). Returns the
        complete selected_columns or None if no completion exists."""
        partial = list(partial)
        # ensure conflict-free
        seen = defaultdict(int)
        for c in partial:
            if c in columns_info:
                for r in columns_info[c][1]:
                    seen[r] += 1
        if any(k >= 2 for k in seen.values()):
            return None  # input itself has conflicts
        # locked-in: partial. Cannot use columns that overlap partial's rows.
        covered_so_far = {r for r, k in seen.items() if k >= 1}
        forbidden = {c for c, (_, rows) in columns_info.items()
                     if rows & covered_so_far and c not in partial}
        return ilp_solve_partition(
            must_include=partial,
            must_exclude=forbidden,
            time_limit_s=time_limit_s,
        )

    # ==================================================================
    # (2) Construction heuristics
    # ==================================================================
    def greedy_cover_by_cost_ratio():
        """Cost-effective greedy: iteratively pick the column with the smallest
        cost / new-rows-covered, dropping columns that would conflict with the
        partial selection. Returns a (possibly partial) selected_columns.
        Often a good warm start; if uncovered_rows is non-empty, refine with
        complete_partial_via_ilp."""
        uncovered: set[int] = set(range(1, num_rows + 1))
        chosen: list[int] = []
        candidates = set(columns_info.keys())
        while uncovered and candidates:
            best = None
            best_ratio = float("inf")
            for c in candidates:
                cost, rows = columns_info[c]
                new = len(rows & uncovered)
                if new == 0:
                    continue
                ratio = cost / new
                if ratio < best_ratio:
                    best_ratio = ratio
                    best = c
            if best is None:
                break
            cost, rows = columns_info[best]
            chosen.append(best)
            covered_now = set(range(1, num_rows + 1)) - uncovered  # before update
            uncovered -= rows
            candidates.discard(best)
            # remove columns that would conflict with any already-covered row
            # (including the rows just covered by `best`)
            covered_after = set(range(1, num_rows + 1)) - uncovered
            candidates = {c for c in candidates
                          if not (columns_info[c][1] & covered_after)}
        return sorted(chosen)

    def cheapest_column_covering_row(
        row: int,
        exclude: Optional[Iterable[int]] = None,
    ):
        """Among columns covering `row` (and not in `exclude`), return the one
        with the smallest cost. Returns None if no such column exists."""
        ex = set(exclude) if exclude else set()
        best = None
        best_cost = float("inf")
        for c in row_to_cols.get(int(row), []):
            if c in ex:
                continue
            cost = int(columns_info[c][0])
            if cost < best_cost:
                best_cost = cost
                best = c
        return best

    # ==================================================================
    # (3) Inspection
    # ==================================================================
    def columns_covering_row(row: int) -> list[int]:
        return list(row_to_cols.get(int(row), []))

    def column_rows(col_id: int) -> set[int]:
        return set(columns_info[col_id][1]) if col_id in columns_info else set()

    def column_cost(col_id: int) -> int:
        return int(columns_info[col_id][0]) if col_id in columns_info else None

    def covered_rows(selected: Iterable[int]) -> set[int]:
        cnt = defaultdict(int)
        for c in selected:
            if c in columns_info:
                for r in columns_info[c][1]:
                    cnt[r] += 1
        return {r for r, k in cnt.items() if k == 1}

    def uncovered_rows(selected: Iterable[int]) -> set[int]:
        cnt = defaultdict(int)
        for c in selected:
            if c in columns_info:
                for r in columns_info[c][1]:
                    cnt[r] += 1
        return {r for r in range(1, num_rows + 1) if cnt.get(r, 0) == 0}

    def conflict_rows(selected: Iterable[int]) -> set[int]:
        cnt = defaultdict(int)
        for c in selected:
            if c in columns_info:
                for r in columns_info[c][1]:
                    cnt[r] += 1
        return {r for r, k in cnt.items() if k >= 2}

    def cost_of_selection(selected: Iterable[int]) -> int:
        return sum(int(columns_info[c][0]) for c in selected if c in columns_info)

    # ==================================================================
    # (4) Validation
    # ==================================================================
    def is_conflict_free(selected: Iterable[int]) -> bool:
        """True iff no two columns in `selected` share any row.
        (Partial selections can be conflict-free without yet covering everything.)"""
        return len(conflict_rows(selected)) == 0

    def feasible_columns_given_partial(partial: Iterable[int]) -> list[int]:
        """Columns that could be ADDED to `partial` without introducing any
        row-cover conflict (i.e., whose row set is disjoint from rows already
        covered by partial). Useful for incremental construction."""
        partial = list(partial)
        already_covered: set[int] = set()
        for c in partial:
            if c in columns_info:
                already_covered |= columns_info[c][1]
        out = []
        for c, (_cost, rows) in columns_info.items():
            if c in partial:
                continue
            if not (rows & already_covered):
                out.append(c)
        return sorted(out)

    return {
        "ilp_solve_partition": ilp_solve_partition,
        "complete_partial_via_ilp": complete_partial_via_ilp,
        "greedy_cover_by_cost_ratio": greedy_cover_by_cost_ratio,
        "cheapest_column_covering_row": cheapest_column_covering_row,
        "columns_covering_row": columns_covering_row,
        "column_rows": column_rows,
        "column_cost": column_cost,
        "covered_rows": covered_rows,
        "uncovered_rows": uncovered_rows,
        "conflict_rows": conflict_rows,
        "cost_of_selection": cost_of_selection,
        "is_conflict_free": is_conflict_free,
        "feasible_columns_given_partial": feasible_columns_given_partial,
    }


EXTRA_TOOLS_DESCRIPTION = [
    # ----- Exact / heavy -----
    {
        "name": "ilp_solve_partition",
        "input": "must_include: Iterable[int] = None, must_exclude: Iterable[int] = None, time_limit_s: float = 10.0",
        "output": "list[int] | None",
        "purpose": (
            "Solve the FULL Set Partitioning ILP exactly with CBC (open-source). "
            "Returns optimal selected_columns satisfying 'each row covered exactly "
            "once', or None if infeasible. Use as primary solver, or with "
            "must_include/must_exclude for LNS-style refinement."
        ),
    },
    {
        "name": "complete_partial_via_ilp",
        "input": "partial: Iterable[int], time_limit_s: float = 10.0",
        "output": "list[int] | None",
        "purpose": (
            "Given a CONFLICT-FREE partial selection, extend it via ILP to a "
            "full feasible partition. Returns the complete selected_columns or "
            "None if no completion exists. Use after greedy construction leaves "
            "some rows uncovered."
        ),
    },
    # ----- Construction -----
    {
        "name": "greedy_cover_by_cost_ratio",
        "input": "(no args)",
        "output": "list[int]",
        "purpose": (
            "Cost-effective greedy heuristic: iteratively pick the column with "
            "smallest cost / new-rows-covered, skipping columns that conflict. "
            "Returns a (possibly partial) selected_columns. Often a good warm "
            "start; combine with complete_partial_via_ilp if uncovered_rows is "
            "non-empty."
        ),
    },
    {
        "name": "cheapest_column_covering_row",
        "input": "row: int, exclude: Iterable[int] = None",
        "output": "int | None",
        "purpose": (
            "Among columns covering `row` (excluding ids in `exclude`), return "
            "the cheapest one. Useful for repair: 'row r is uncovered, find me "
            "the cheapest column to add'."
        ),
    },
    # ----- Inspection -----
    {
        "name": "columns_covering_row",
        "input": "row: int",
        "output": "list[int]",
        "purpose": "All column ids that cover the given row.",
    },
    {
        "name": "column_rows",
        "input": "col_id: int",
        "output": "set[int]",
        "purpose": "Rows covered by a single column.",
    },
    {
        "name": "column_cost",
        "input": "col_id: int",
        "output": "int | None",
        "purpose": "Cost of a single column (or None if col_id not in instance).",
    },
    {
        "name": "covered_rows",
        "input": "selected: Iterable[int]",
        "output": "set[int]",
        "purpose": "Rows covered EXACTLY ONCE by the partial selection (no over-cover).",
    },
    {
        "name": "uncovered_rows",
        "input": "selected: Iterable[int]",
        "output": "set[int]",
        "purpose": "Rows the partial selection has not covered yet.",
    },
    {
        "name": "conflict_rows",
        "input": "selected: Iterable[int]",
        "output": "set[int]",
        "purpose": "Rows covered MORE THAN ONCE (>= 2). Feasibility requires this to be empty.",
    },
    {
        "name": "cost_of_selection",
        "input": "selected: Iterable[int]",
        "output": "int",
        "purpose": "Total cost of the columns in `selected` (no feasibility check).",
    },
    # ----- Validation -----
    {
        "name": "is_conflict_free",
        "input": "selected: Iterable[int]",
        "output": "bool",
        "purpose": (
            "True iff no two columns share any row. Useful precheck before "
            "calling complete_partial_via_ilp -- the input there MUST be "
            "conflict-free."
        ),
    },
    {
        "name": "feasible_columns_given_partial",
        "input": "partial: Iterable[int]",
        "output": "list[int]",
        "purpose": (
            "Columns that could be ADDED to `partial` without introducing any "
            "row-cover conflict. Useful for incremental construction one column "
            "at a time."
        ),
    },
]
