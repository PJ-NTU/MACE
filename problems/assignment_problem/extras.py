"""Per-problem extras for the classic (square) Assignment Problem.

Provides primitive building blocks for the classic n x n linear assignment
problem (each agent does exactly one task, each task done by exactly one
agent, minimize sum of costs).

Tool groups:
  (1) Queries:        cost, n
  (2) Feasibility:    cost_of_assignment, validate_assignment
  (3) Construction:   hungarian, greedy_min_cost, greedy_min_regret
  (4) Local search:   apply_swap_two_assignments

The classic AP is polynomial-time solvable: `hungarian()` calls
scipy.optimize.linear_sum_assignment, which is the O(n^3) exact algorithm
and should be the go-to tool for any vanilla instance. The other helpers are
pedagogical or useful for variants / experimentation.

The CO-Bench solution dict has shape:
    {"total_cost": float, "assignment": list[(int, int)]}
where each (i, j) is 1-INDEXED (item i to agent j). All tools below produce
and consume this 1-indexed list-of-tuples convention so values flow directly
into tools['is_feasible'] / ['objective'].
"""
from __future__ import annotations
from typing import Iterable, List, Tuple, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment


def extra_tools(instance: dict) -> dict:
    """Factory: returns AP-specific tool callables given the loaded instance.

    Instance schema (from CO-Bench Assignment-problem load_data, one case):
      - n:           int, number of items (== number of agents)
      - cost_matrix: numpy.ndarray of shape (n, n), 0-indexed
    """
    n: int = int(instance["n"])
    C = np.asarray(instance["cost_matrix"], dtype=np.float64)
    if C.shape != (n, n):
        raise ValueError(f"cost_matrix shape {C.shape} != ({n}, {n})")

    # ==================================================================
    # (1) Queries
    # ==================================================================
    def cost(i: int, j: int) -> float:
        """Cost of assigning item i (1-indexed) to agent j (1-indexed)."""
        ii, jj = int(i) - 1, int(j) - 1
        if not (0 <= ii < n and 0 <= jj < n):
            raise ValueError(f"(i, j)=({i}, {j}) out of range [1, {n}]")
        return float(C[ii, jj])

    def n_func() -> int:
        return n

    # ==================================================================
    # (2) Feasibility primitives
    # ==================================================================
    def _coerce_pairs(assignment: Iterable) -> List[Tuple[int, int]]:
        pairs = []
        for idx, p in enumerate(assignment):
            if not (isinstance(p, (list, tuple)) and len(p) == 2):
                raise ValueError(f"assignment entry {idx} not a 2-tuple: {p!r}")
            pairs.append((int(p[0]), int(p[1])))
        return pairs

    def cost_of_assignment(assignment: Iterable) -> float:
        """Total cost of `assignment` (a list of (i, j) 1-indexed pairs). Does
        NOT validate feasibility -- just sums cost_matrix[i-1][j-1] over the
        pairs. Useful when you trust the pairs and only want the objective."""
        pairs = _coerce_pairs(assignment)
        total = 0.0
        for (i, j) in pairs:
            if not (1 <= i <= n and 1 <= j <= n):
                raise ValueError(f"pair ({i}, {j}) out of range [1, {n}]")
            total += float(C[i - 1, j - 1])
        return total

    def validate_assignment(assignment: Iterable) -> Tuple[bool, Optional[str]]:
        """Cheap feasibility precheck WITHOUT calling eval_func: verifies that
        `assignment` is a permutation -- exactly n pairs, every item in [1, n]
        used exactly once and every agent in [1, n] used exactly once, and no
        pair has infinite cost. Returns (True, None) on success, else
        (False, reason)."""
        try:
            pairs = _coerce_pairs(assignment)
        except ValueError as e:
            return False, str(e)
        if len(pairs) != n:
            return False, f"expected {n} pairs, got {len(pairs)}"
        items_seen = set()
        agents_seen = set()
        for (i, j) in pairs:
            if not (1 <= i <= n):
                return False, f"item {i} out of range [1, {n}]"
            if not (1 <= j <= n):
                return False, f"agent {j} out of range [1, {n}]"
            if i in items_seen:
                return False, f"item {i} assigned more than once"
            if j in agents_seen:
                return False, f"agent {j} assigned more than once"
            if not np.isfinite(C[i - 1, j - 1]):
                return False, f"pair ({i}, {j}) has non-finite cost"
            items_seen.add(i)
            agents_seen.add(j)
        return True, None

    # ==================================================================
    # (3) Construction
    # ==================================================================
    def hungarian() -> List[Tuple[int, int]]:
        """Solve the classic Assignment Problem EXACTLY using the Hungarian
        algorithm (scipy.optimize.linear_sum_assignment, O(n^3)).

        Returns a 1-indexed list of (item, agent) tuples sorted by item.
        This is the OPTIMAL solution for the vanilla square AP -- no further
        search is needed. Use as the primary solver."""
        row_ind, col_ind = linear_sum_assignment(C)
        # row_ind == arange(n) for a square cost matrix.
        return [(int(r) + 1, int(c) + 1) for r, c in zip(row_ind, col_ind)]

    def greedy_min_cost() -> List[Tuple[int, int]]:
        """Greedy construction: repeatedly pick the cheapest remaining (i, j)
        pair whose item i and agent j are still free. O(n^2 log n). Usually
        worse than hungarian() but useful as a pedagogical baseline / warm
        start for local search on AP variants."""
        # Flatten and sort all (cost, i, j) entries.
        idx = np.argsort(C, axis=None, kind="stable")
        used_i = [False] * n
        used_j = [False] * n
        out: List[Tuple[int, int]] = [None] * n  # type: ignore
        placed = 0
        for flat in idx:
            i = int(flat // n)
            j = int(flat % n)
            if used_i[i] or used_j[j]:
                continue
            if not np.isfinite(C[i, j]):
                continue
            used_i[i] = True
            used_j[j] = True
            out[i] = (i + 1, j + 1)
            placed += 1
            if placed == n:
                break
        # If some items remain (e.g. due to infinities), fill with any free agent.
        if placed < n:
            free_j = [j + 1 for j in range(n) if not used_j[j]]
            k = 0
            for i in range(n):
                if not used_i[i]:
                    out[i] = (i + 1, free_j[k])
                    k += 1
        return out

    def greedy_min_regret() -> List[Tuple[int, int]]:
        """Regret-based greedy: at each step, for every still-unassigned item
        i compute its 'regret' = (2nd cheapest free agent's cost) - (cheapest
        free agent's cost); assign the item with the LARGEST regret to its
        cheapest free agent. Intuition: place items first where missing the
        best agent hurts the most. O(n^3) overall. Often better than plain
        greedy_min_cost, still suboptimal vs hungarian()."""
        remaining_items = list(range(n))
        remaining_agents = list(range(n))
        out: List[Tuple[int, int]] = [None] * n  # type: ignore
        while remaining_items:
            best_item = None
            best_agent = None
            best_regret = -np.inf
            for i in remaining_items:
                row = C[i, remaining_agents]
                if row.size == 0:
                    continue
                if row.size == 1:
                    cheapest_j_local = int(np.argmin(row))
                    regret = np.inf  # forced, treat as huge regret
                    cheapest_cost = float(row[cheapest_j_local])
                else:
                    # Partial sort for top 2 cheapest.
                    order = np.argpartition(row, 1)[:2]
                    # Make sure order[0] is the actually-cheapest one.
                    if row[order[0]] > row[order[1]]:
                        order = order[::-1]
                    cheapest_j_local = int(order[0])
                    cheapest_cost = float(row[order[0]])
                    second_cost = float(row[order[1]])
                    regret = second_cost - cheapest_cost
                if not np.isfinite(cheapest_cost):
                    continue
                if regret > best_regret:
                    best_regret = regret
                    best_item = i
                    best_agent = remaining_agents[cheapest_j_local]
            if best_item is None:
                # All remaining rows are inf -- arbitrary pairing.
                best_item = remaining_items[0]
                best_agent = remaining_agents[0]
            out[best_item] = (best_item + 1, best_agent + 1)
            remaining_items.remove(best_item)
            remaining_agents.remove(best_agent)
        return out

    # ==================================================================
    # (4) Local search
    # ==================================================================
    def apply_swap_two_assignments(
        assignment: Iterable, k1: int, k2: int
    ) -> List[Tuple[int, int]]:
        """Return a NEW assignment where the agents of the k1-th and k2-th
        pairs (0-indexed) are swapped. Pure function -- the input is not
        mutated.

        Given assignment = [(i_1, j_1), ..., (i_n, j_n)], swapping positions
        k1 and k2 produces a list with (i_{k1}, j_{k2}) and (i_{k2}, j_{k1}).
        Because each item still maps to exactly one agent and the two swapped
        agents are still each used once, the result remains a valid
        permutation. The cost change is:
            C[i_{k1}-1, j_{k2}-1] + C[i_{k2}-1, j_{k1}-1]
          - C[i_{k1}-1, j_{k1}-1] - C[i_{k2}-1, j_{k2}-1]
        Use as the elementary neighborhood move for 2-exchange local search.
        """
        pairs = _coerce_pairs(assignment)
        if not (0 <= int(k1) < len(pairs) and 0 <= int(k2) < len(pairs)):
            raise ValueError(
                f"k1={k1}, k2={k2} out of range [0, {len(pairs)})"
            )
        k1i, k2i = int(k1), int(k2)
        if k1i == k2i:
            return list(pairs)
        out = list(pairs)
        i1, j1 = out[k1i]
        i2, j2 = out[k2i]
        out[k1i] = (i1, j2)
        out[k2i] = (i2, j1)
        return out

    # ==================================================================
    # (5) Solution-dict builder + one-shot optimal solver
    # ==================================================================
    def make_solution(assignment: Iterable) -> dict:
        """Wrap a 1-indexed list of (item, agent) pairs into the EXACT dict
        shape eval_func expects: {'total_cost': float, 'assignment': list}.

        Computes total_cost from the cost matrix. Use this on the output of
        hungarian() / greedy_min_cost() / greedy_min_regret() so you don't
        accidentally return the wrong shape. Does NOT validate -- pair with
        validate_assignment() if needed."""
        pairs = _coerce_pairs(assignment)
        total = 0.0
        for (i, j) in pairs:
            if 1 <= i <= n and 1 <= j <= n:
                total += float(C[i - 1, j - 1])
        return {"total_cost": float(total), "assignment": pairs}

    def solve_optimal() -> dict:
        """ONE-LINE OPTIMAL SOLVER. Returns the complete solution dict
        {'total_cost': float, 'assignment': list[(i, j)]} for the classic
        Assignment Problem using scipy.optimize.linear_sum_assignment.

        This is the BEST possible solution -- no further search is needed for
        the vanilla AP. Use as the FIRST thing your solve() function calls;
        if you have time left you can fall back to local search but you
        cannot beat this."""
        return make_solution(hungarian())

    return {
        # (1) queries
        "cost": cost,
        "n": n_func,
        # (2) feasibility
        "cost_of_assignment": cost_of_assignment,
        "validate_assignment": validate_assignment,
        # (3) construction
        "hungarian": hungarian,
        "greedy_min_cost": greedy_min_cost,
        "greedy_min_regret": greedy_min_regret,
        # (4) local search
        "apply_swap_two_assignments": apply_swap_two_assignments,
        # (5) solution builder + one-shot optimal
        "make_solution": make_solution,
        "solve_optimal": solve_optimal,
    }


EXTRA_TOOLS_DESCRIPTION = [
    # ----- (5) ONE-SHOT OPTIMAL SOLVER (call this first!) -----
    {
        "name": "solve_optimal",
        "input": "(no args)",
        "output": "dict {'total_cost': float, 'assignment': list[(i, j)]}",
        "purpose": (
            "RECOMMENDED START: returns the OPTIMAL solution dict ready to "
            "return directly. Wraps Hungarian (scipy.optimize.linear_sum_"
            "assignment) and computes total_cost. The vanilla AP is "
            "polynomial-time -- this is the best answer; no local search "
            "needed. ONE LINE: `return tools['solve_optimal']()`."
        ),
    },
    {
        "name": "make_solution",
        "input": "assignment: list[(int, int)] (1-indexed pairs)",
        "output": "dict {'total_cost': float, 'assignment': list}",
        "purpose": (
            "Build the EXACT solution dict shape eval_func wants from a list "
            "of (item, agent) 1-indexed pairs. Computes total_cost from the "
            "cost matrix. Use on the output of hungarian() / greedy_min_cost() "
            "/ etc. so you never return the wrong dict shape."
        ),
    },
    # ----- (1) Queries -----
    {
        "name": "cost",
        "input": "i: int (1-indexed item), j: int (1-indexed agent)",
        "output": "float",
        "purpose": "cost_matrix[i-1][j-1]: cost of assigning item i to agent j.",
    },
    {
        "name": "n",
        "input": "(no args)",
        "output": "int",
        "purpose": "Problem size n (square cost matrix is n x n).",
    },
    # ----- (2) Feasibility primitives -----
    {
        "name": "cost_of_assignment",
        "input": "assignment: list[(int, int)] (1-indexed pairs)",
        "output": "float",
        "purpose": (
            "Sum of cost_matrix[i-1][j-1] over the (i, j) pairs in "
            "`assignment`. Does NOT validate that the pairs form a "
            "permutation -- use validate_assignment for that. Cheap; use "
            "inside local-search loops."
        ),
    },
    {
        "name": "validate_assignment",
        "input": "assignment: list[(int, int)]",
        "output": "(bool, str | None)",
        "purpose": (
            "Cheap feasibility precheck WITHOUT calling eval_func: verifies "
            "exactly n pairs, every item in [1, n] used exactly once, every "
            "agent in [1, n] used exactly once, and no pair has infinite "
            "cost. Returns (True, None) on success else (False, reason)."
        ),
    },
    # ----- (3) Construction -----
    {
        "name": "hungarian",
        "input": "(no args)",
        "output": "list[(int, int)]",
        "purpose": (
            "Hungarian algorithm (scipy linear_sum_assignment, O(n^3)) -- "
            "OPTIMAL list of (item, agent) pairs. NOTE: returns ONLY the "
            "pairs list, not the full solution dict; wrap with make_solution() "
            "or just call solve_optimal() instead to get the ready-to-return dict."
        ),
    },
    {
        "name": "greedy_min_cost",
        "input": "(no args)",
        "output": "list[(int, int)]",
        "purpose": (
            "Greedy construction: repeatedly pick the cheapest still-available "
            "(item, agent) pair. Suboptimal vs hungarian() but useful as a "
            "baseline warm start for local-search experiments."
        ),
    },
    {
        "name": "greedy_min_regret",
        "input": "(no args)",
        "output": "list[(int, int)]",
        "purpose": (
            "Regret-based greedy: at each step assign the item whose "
            "(2nd-cheapest - cheapest) free agent cost gap is largest, to "
            "its cheapest free agent. O(n^3). Usually better than "
            "greedy_min_cost; still beaten by hungarian()."
        ),
    },
    # ----- (4) Local search -----
    {
        "name": "apply_swap_two_assignments",
        "input": "assignment: list[(int, int)], k1: int (0-indexed), k2: int (0-indexed)",
        "output": "list[(int, int)]",
        "purpose": (
            "Return a NEW assignment where the agents of positions k1 and k2 "
            "are swapped. Pure function -- input is not mutated. The result "
            "is always a valid permutation, so feasibility is preserved. "
            "Elementary 2-exchange move for local search."
        ),
    },
]
