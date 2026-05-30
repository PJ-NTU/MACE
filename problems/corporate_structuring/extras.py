"""Per-problem extras for CO-Bench Corporate Structuring.

This task asks the LLM to build a rooted tree of "corporate entities". The
target country is the root (parent 0). All countries with profit > 0 should
appear in the tree exactly once. Dividends flow child -> parent. Each parent
receives child remittances (after the child's domestic + foreign tax and
after a withholding rate W[child][parent]). Parents then pay extra foreign
tax according to their tax code (1: exemption / 2: deduction /
3: source pooling / 4: world-wide pooling). The objective (maximize) is the
after-tax outcome at the root.

The eval_func is RECURSIVE over the tree and a bit slow due to repeated
`outcome` calls and the unconditional `print(P_cache)` it does. The tools
below offer:

  (1) Queries:     country_info, withholding_rate, positive_profit_countries,
                   parent_of (within a given structure)
  (2) Evaluation:  tree_score        -- silent, fast (memoized) port of eval_func
                   node_after_tax    -- outcome(i) on demand for any node
                   children_of       -- children map from a structure
  (3) Heuristics:  flat_tree         -- all positive nodes directly under target
                   greedy_attach     -- insert nodes one-by-one at the best parent
                   reparent_local_search  -- climb by re-attaching one node at a time

All are best-effort building blocks. The LLM is free to use any subset.
"""
from __future__ import annotations
import random
import time
from typing import Iterable, Optional


def extra_tools(instance: dict) -> dict:
    N: int = int(instance["N"])
    target: int = int(instance["target"])
    countries: dict = instance["countries"]
    withholding: dict = instance["withholding"]

    # Pre-extract per-country fields.
    tax_code = {i: countries[i][0] for i in countries}
    f_rate = {i: countries[i][1] for i in countries}
    d_rate = {i: countries[i][2] for i in countries}
    profit = {i: countries[i][3] for i in countries}

    pos_nodes = sorted(i for i in range(1, N + 1) if profit[i] > 0)
    pos_set = set(pos_nodes)

    # ==================================================================
    # (2) Fast in-process scorer. Mirrors eval_func semantics but without
    #     the global `print(P_cache)` and with light memoization. Returns
    #     `outcome(target)` (the raw maximization value).
    # ==================================================================
    def _children_from_structure(structure: dict) -> dict:
        children = {i: [] for i in range(1, N + 1)}
        for c, p in structure.items():
            if p == 0:
                continue
            if 1 <= p <= N and 1 <= c <= N:
                children[p].append(c)
        return children

    def _compute(structure: dict):
        """Return (raw_score, outcome_map, foreign_income_map, P_map, children_map).

        Pure replication of eval_func's math; never raises -- returns None if
        the structure is malformed (cycle, parent out of range, etc.).
        """
        # Validate parents.
        for c, p in structure.items():
            if not (1 <= c <= N):
                return None
            if p != 0 and not (1 <= p <= N):
                return None
        children = _children_from_structure(structure)
        # Detect cycles via iterative root-walk from each node up to target.
        # Each node in structure must have a path to target with parent 0.
        # Cheap: parent map.
        parent = {c: p for c, p in structure.items()}
        if parent.get(target, None) != 0:
            return None
        # Walk every node up: must reach target without revisiting.
        for c in parent:
            seen = set()
            cur = c
            while cur != 0:
                if cur in seen:
                    return None  # cycle
                seen.add(cur)
                nxt = parent.get(cur, None)
                if nxt is None:
                    return None  # disconnected
                cur = nxt

        # P[i] = sum of profits in subtree rooted at i.
        P = {}

        def get_P(i):
            if i in P:
                return P[i]
            t = profit[i]
            for c in children[i]:
                t += get_P(c)
            P[i] = t
            return t

        for i in range(1, N + 1):
            get_P(i)

        outcome_map = {}
        f_income_map = {}

        def foreign_income(i):
            if not children[i]:
                f_income_map[i] = {}
                return {}
            tot = {}
            for c in children[i]:
                a = outcome(c)
                tot[c] = a * (1 - withholding[c][i])
            f_income_map[i] = tot
            return tot

        def outcome(i):
            if i in outcome_map:
                return outcome_map[i]
            d_income = profit[i] * (1 - d_rate[i])
            f_inc = foreign_income(i)
            total_f = sum(f_inc.values())
            code = tax_code[i]
            if code == 1:
                v = d_income + total_f
            elif code == 2:
                v = d_income + total_f * (1 - f_rate[i])
            elif code == 3:
                v = d_income + total_f - sum(
                    max(0, f_inc[c] - (1 - f_rate[i]) * P[c]) for c in children[i]
                )
            else:  # 4
                v = d_income + total_f - max(
                    0, total_f - (1 - f_rate[i]) * (P[i] - profit[i])
                )
            outcome_map[i] = v
            return v

        try:
            raw = outcome(target)
        except Exception:
            return None
        return raw, outcome_map, f_income_map, P, children

    def tree_score(structure: dict) -> float:
        """Silent, fast `eval_func`-equivalent. Returns the raw after-tax
        outcome at the target (HIGHER IS BETTER), or `-inf` if `structure`
        is malformed (cycle, broken parent chain, missing target root)."""
        r = _compute(structure)
        if r is None:
            return float("-inf")
        return float(r[0])

    def node_after_tax(i: int, structure: dict) -> float:
        """outcome(i) under the given structure -- the after-(local-)tax
        amount country `i` produces. Returns `-inf` if structure is invalid
        or `i` is not reachable."""
        r = _compute(structure)
        if r is None:
            return float("-inf")
        _, outcome_map, _, _, _ = r
        return float(outcome_map.get(int(i), float("-inf")))

    def children_of(structure: dict, node: int) -> list:
        """Direct children of `node` in `structure` (empty list if none)."""
        ch = _children_from_structure(structure)
        return list(ch.get(int(node), []))

    # ==================================================================
    # (1) Queries
    # ==================================================================
    def country_info(i: int) -> dict:
        i = int(i)
        if i not in countries:
            return {}
        return {
            "tax_code": tax_code[i],
            "foreign_rate": f_rate[i],
            "domestic_rate": d_rate[i],
            "profit": profit[i],
        }

    def withholding_rate(src: int, dst: int) -> float:
        return float(withholding[int(src)][int(dst)])

    def positive_profit_countries() -> list:
        return list(pos_nodes)

    def parent_of(structure: dict, node: int) -> Optional[int]:
        return structure.get(int(node), None)

    # ==================================================================
    # (3) Construction / local search
    # ==================================================================
    def flat_tree() -> dict:
        """Every positive-profit country attached directly to the target.
        Solid baseline: typically within a few % of optimum on most cases."""
        struct = {target: 0}
        for k in pos_nodes:
            if k != target:
                struct[k] = target
        return struct

    def _try_score(struct: dict) -> float:
        s = tree_score(struct)
        return s if s != float("-inf") else float("-inf")

    def greedy_attach(order: Optional[Iterable[int]] = None,
                     time_limit_s: float = 5.0) -> dict:
        """Start from {target: 0}; insert each remaining positive-profit
        country one at a time, picking the parent that yields the BEST
        (highest) `tree_score` among already-inserted nodes. Greedy O(K^2)
        evaluations where K = #positive_profit countries.

        `order` (optional): the insertion order of non-target countries.
        Defaults to descending profit (richest first)."""
        if order is None:
            order_list = sorted([k for k in pos_nodes if k != target],
                                key=lambda x: -profit[x])
        else:
            order_list = [int(k) for k in order if int(k) != target]

        struct = {target: 0}
        inserted = {target}
        t0 = time.time()
        for node in order_list:
            if time.time() - t0 > time_limit_s:
                # Fall back: attach the rest directly to target.
                struct[node] = target
                inserted.add(node)
                continue
            best_parent = target
            best_val = float("-inf")
            for p in list(inserted):
                struct[node] = p
                v = tree_score(struct)
                if v > best_val:
                    best_val = v
                    best_parent = p
            struct[node] = best_parent
            inserted.add(node)
        return struct

    def reparent_local_search(structure: dict,
                              time_limit_s: float = 10.0,
                              first_improvement: bool = True) -> dict:
        """Hill-climb by changing one country's parent at a time. For each
        non-root node, try every other in-tree node as a new parent (skipping
        moves that would create a cycle). Keeps any move that strictly
        improves `tree_score`. Stops at a local maximum or when time runs out.

        Returns the improved structure (new dict). The input structure is
        not modified."""
        struct = dict(structure)
        best = tree_score(struct)
        if best == float("-inf"):
            return struct
        t0 = time.time()
        improved = True
        while improved and (time.time() - t0) < time_limit_s - 0.05:
            improved = False
            nodes = [k for k in struct.keys() if struct[k] != 0]
            # Determine descendants of each node to avoid creating cycles.
            children = _children_from_structure(struct)

            def descendants(root):
                out = set()
                stack = [root]
                while stack:
                    cur = stack.pop()
                    for c in children.get(cur, []):
                        if c not in out:
                            out.add(c)
                            stack.append(c)
                return out

            best_move = None
            best_delta = 0.0
            for node in nodes:
                if (time.time() - t0) >= time_limit_s - 0.05:
                    break
                desc = descendants(node) | {node}
                old_p = struct[node]
                candidates = [p for p in struct.keys() if p not in desc and p != old_p]
                for new_p in candidates:
                    struct[node] = new_p
                    v = tree_score(struct)
                    struct[node] = old_p  # revert
                    if v == float("-inf"):
                        continue
                    if v > best + 1e-9:
                        delta = v - best
                        if first_improvement:
                            struct[node] = new_p
                            best = v
                            improved = True
                            break
                        elif delta > best_delta:
                            best_delta = delta
                            best_move = (node, new_p, v)
                if first_improvement and improved:
                    break
            if (not first_improvement) and best_move is not None:
                node, new_p, v = best_move
                struct[node] = new_p
                best = v
                improved = True
        return struct

    return {
        # queries
        "country_info": country_info,
        "withholding_rate": withholding_rate,
        "positive_profit_countries": positive_profit_countries,
        "parent_of": parent_of,
        # evaluation
        "tree_score": tree_score,
        "node_after_tax": node_after_tax,
        "children_of": children_of,
        # heuristics
        "flat_tree": flat_tree,
        "greedy_attach": greedy_attach,
        "reparent_local_search": reparent_local_search,
    }


EXTRA_TOOLS_DESCRIPTION = [
    # ----- Queries -----
    {
        "name": "country_info",
        "input": "i: int",
        "output": "dict",
        "purpose": (
            "Return {'tax_code', 'foreign_rate', 'domestic_rate', 'profit'} "
            "for country `i` (1-indexed). Empty dict if `i` is out of range."
        ),
    },
    {
        "name": "withholding_rate",
        "input": "src: int, dst: int",
        "output": "float",
        "purpose": (
            "withholding[src][dst]: the dividend tax rate applied when `src` "
            "remits to `dst`. Use when deciding whether to route a child "
            "through an intermediate country."
        ),
    },
    {
        "name": "positive_profit_countries",
        "input": "(no args)",
        "output": "list[int]",
        "purpose": (
            "Sorted list of country ids whose profit > 0. The problem REQUIRES "
            "each of these to appear in the tree exactly once."
        ),
    },
    {
        "name": "parent_of",
        "input": "structure: dict, node: int",
        "output": "int | None",
        "purpose": "Parent of `node` in the given structure (None if absent).",
    },
    # ----- Evaluation -----
    {
        "name": "tree_score",
        "input": "structure: dict",
        "output": "float",
        "purpose": (
            "Silent, fast equivalent of CO-Bench's eval_func. Returns the raw "
            "after-tax outcome at the root (HIGHER IS BETTER). Returns -inf "
            "if `structure` has a cycle, a broken parent chain, a parent out "
            "of range, or the target is not the root. Prefer this inside "
            "loops -- it avoids eval_func's unconditional `print(P_cache)`."
        ),
    },
    {
        "name": "node_after_tax",
        "input": "i: int, structure: dict",
        "output": "float",
        "purpose": (
            "outcome(i) under `structure`: country `i`'s after-(local-)tax "
            "amount before remittance to its parent. Useful for diagnosing "
            "which subtree is leaking value. -inf if structure is invalid."
        ),
    },
    {
        "name": "children_of",
        "input": "structure: dict, node: int",
        "output": "list[int]",
        "purpose": "Direct children of `node` derived from `structure`.",
    },
    # ----- Heuristics -----
    {
        "name": "flat_tree",
        "input": "(no args)",
        "output": "dict",
        "purpose": (
            "Baseline structure: every positive-profit country attached "
            "directly to the target. Often within a few percent of optimum "
            "and a robust warm start for local search."
        ),
    },
    {
        "name": "greedy_attach",
        "input": "order: Iterable[int] = None, time_limit_s: float = 5.0",
        "output": "dict",
        "purpose": (
            "Build a tree by inserting countries one at a time. For each "
            "candidate, try every already-inserted node as its parent and "
            "pick the one giving the highest `tree_score`. Default order is "
            "descending profit (richest first). O(K^2) evaluations."
        ),
    },
    {
        "name": "reparent_local_search",
        "input": "structure: dict, time_limit_s: float = 10.0, first_improvement: bool = True",
        "output": "dict",
        "purpose": (
            "Hill-climb on `structure`: for each non-root node, try every "
            "other in-tree node as a new parent (skipping moves that would "
            "create a cycle). Accept any strictly-improving move. Stops at a "
            "local maximum or when time runs out. Returns a NEW dict; the "
            "input is not modified. Pair with `flat_tree` or `greedy_attach` "
            "as a warm start."
        ),
    },
]
