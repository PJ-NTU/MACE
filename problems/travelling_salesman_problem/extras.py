"""Per-problem extras for CO-Bench Travelling Salesman Problem.

Provides primitive building blocks so the LLM can compose construction +
local-search heuristics without reinventing 2-opt, NN-construction, etc.
All tools share a precomputed distance matrix D and k-NN cache per instance.

Tool groups:
  (1) Construction:   nn_construct, random_tour
  (2) Local search:   apply_2opt, apply_or_opt_single, two_opt_delta
  (3) Queries:        tour_length, k_nearest

All are optional. LLM may use any subset, or write everything from scratch.
"""
from __future__ import annotations
import random
import time
from typing import Optional

import numpy as np


def extra_tools(instance: dict) -> dict:
    """Factory: returns TSP-specific tool callables given the loaded instance.

    Instance schema (from CO-Bench TSP load_data):
      - nodes:      list[(x, y)] coordinates
      - label_tour: Concorde-optimal tour (used by eval_func; not for solve)
    """
    nodes = instance["nodes"]
    n = len(nodes)
    coords = np.array(nodes, dtype=np.float64)

    # Precompute pairwise distance matrix (Euclidean) once per instance.
    diff = coords[:, None, :] - coords[None, :, :]
    D = np.sqrt((diff ** 2).sum(axis=-1))

    # Precompute k-nearest ordering: knn[i] = node indices sorted by D[i, *].
    # knn[i, 0] == i (zero distance), knn[i, 1:k+1] is the top-k neighbors.
    knn = np.argsort(D, axis=1)

    # ==================================================================
    # (1) Construction
    # ==================================================================
    def nn_construct(start_node: int = 0) -> list:
        if not (0 <= start_node < n):
            raise ValueError(f"start_node={start_node} out of range [0, {n})")
        tour = [int(start_node)]
        unvisited = set(range(n)) - {int(start_node)}
        while unvisited:
            last = tour[-1]
            nxt = min(unvisited, key=lambda u: D[last, u])
            tour.append(int(nxt))
            unvisited.discard(nxt)
        return tour

    def random_tour() -> list:
        perm = list(range(n))
        random.shuffle(perm)
        return perm

    # ==================================================================
    # (2) Local search
    # ==================================================================
    def tour_length(tour: list) -> float:
        L = len(tour)
        cost = 0.0
        for i in range(L):
            cost += float(D[tour[i], tour[(i + 1) % L]])
        return cost

    def two_opt_delta(i: int, j: int, tour: list) -> float:
        L = len(tour)
        if not (0 < i < j < L):
            raise ValueError(f"2-opt requires 0 < i < j < n, got i={i}, j={j}, n={L}")
        a, b = tour[i - 1], tour[i]
        c, d = tour[j], tour[(j + 1) % L]
        return float(D[a, c] + D[b, d] - D[a, b] - D[c, d])

    def apply_2opt(tour: list, time_limit_s: float = 5.0,
                   first_improvement: bool = True) -> list:
        t = list(tour)
        L = len(t)
        if L < 4:
            return t
        t0 = time.time()
        safety = 0.05  # leave 50ms headroom
        improved = True
        while improved and (time.time() - t0) < time_limit_s - safety:
            improved = False
            best_delta = 0.0
            best_ij = None
            for i in range(1, L - 1):
                if (time.time() - t0) >= time_limit_s - safety:
                    break
                for j in range(i + 1, L):
                    a, b = t[i - 1], t[i]
                    c, d = t[j], t[(j + 1) % L]
                    delta = float(D[a, c] + D[b, d] - D[a, b] - D[c, d])
                    if delta < -1e-10:
                        if first_improvement:
                            t[i:j + 1] = t[i:j + 1][::-1]
                            improved = True
                            break
                        elif delta < best_delta:
                            best_delta = delta
                            best_ij = (i, j)
                if first_improvement and improved:
                    break
            if (not first_improvement) and best_ij is not None:
                i, j = best_ij
                t[i:j + 1] = t[i:j + 1][::-1]
                improved = True
        return t

    def apply_or_opt_single(tour: list, time_limit_s: float = 5.0) -> list:
        """Or-opt with segment length 1: try moving each city to a better
        position in the tour. First-improvement, restarts from index 0 after
        each successful move. Complementary to 2-opt -- often escapes when
        2-opt is stuck."""
        t = list(tour)
        L = len(t)
        if L < 4:
            return t
        t0 = time.time()
        safety = 0.05
        improved = True
        while improved and (time.time() - t0) < time_limit_s - safety:
            improved = False
            for i in range(L):
                if (time.time() - t0) >= time_limit_s - safety:
                    break
                a_prev = t[(i - 1) % L]
                x = t[i]
                a_next = t[(i + 1) % L]
                d_remove = float(D[a_prev, x] + D[x, a_next] - D[a_prev, a_next])
                best_delta = 0.0
                best_k = None
                for k in range(L):
                    # don't reinsert adjacent to original position
                    if k == i or k == (i - 1) % L:
                        continue
                    b = t[k]
                    b_next = t[(k + 1) % L]
                    d_insert = float(D[b, x] + D[x, b_next] - D[b, b_next])
                    delta = d_insert - d_remove
                    if delta < best_delta - 1e-10:
                        best_delta = delta
                        best_k = k
                if best_k is not None:
                    rest = t[:i] + t[i + 1:]
                    new_k = best_k if best_k < i else best_k - 1
                    t = rest[:new_k + 1] + [x] + rest[new_k + 1:]
                    improved = True
                    break
        return t

    # ==================================================================
    # (3) Queries
    # ==================================================================
    def k_nearest(node: int, k: int = 10) -> list:
        if not (0 <= node < n):
            raise ValueError(f"node={node} out of range [0, {n})")
        k = max(1, min(int(k), n - 1))
        return knn[int(node), 1:k + 1].tolist()

    return {
        "nn_construct": nn_construct,
        "random_tour": random_tour,
        "apply_2opt": apply_2opt,
        "apply_or_opt_single": apply_or_opt_single,
        "two_opt_delta": two_opt_delta,
        "tour_length": tour_length,
        "k_nearest": k_nearest,
    }


EXTRA_TOOLS_DESCRIPTION = [
    # ----- Construction -----
    {
        "name": "nn_construct",
        "input": "start_node: int = 0",
        "output": "list[int]",
        "purpose": (
            "Nearest-neighbor tour construction starting from `start_node`. "
            "Returns a tour as a list of city indices (permutation of [0, n)). "
            "O(n^2). Good warm start for any local-search method."
        ),
    },
    {
        "name": "random_tour",
        "input": "(no args)",
        "output": "list[int]",
        "purpose": (
            "Uniformly random permutation of [0, n). Useful for restarts and "
            "diversification in multi-start / ILS heuristics."
        ),
    },
    # ----- Local search -----
    {
        "name": "apply_2opt",
        "input": "tour: list[int], time_limit_s: float = 5.0, first_improvement: bool = True",
        "output": "list[int]",
        "purpose": (
            "Run 2-opt local search on `tour` until no improving swap exists "
            "or time_limit_s elapses. Returns the improved tour (new list). "
            "Pure Python, O(n^2) per pass. For n>500 you may want to limit "
            "time_limit_s to a few seconds and run multiple restarts."
        ),
    },
    {
        "name": "apply_or_opt_single",
        "input": "tour: list[int], time_limit_s: float = 5.0",
        "output": "list[int]",
        "purpose": (
            "Or-opt local search with segment length 1: tries moving each "
            "city to a better position. First-improvement. Complementary to "
            "2-opt; often escapes 2-opt's local minima."
        ),
    },
    {
        "name": "two_opt_delta",
        "input": "i: int, j: int, tour: list[int]",
        "output": "float",
        "purpose": (
            "Cost change if you reverse tour[i:j+1] (a 2-opt move). O(1) -- "
            "touches only the four endpoint edges. Use to evaluate moves "
            "without recomputing the full tour length."
        ),
    },
    # ----- Queries -----
    {
        "name": "tour_length",
        "input": "tour: list[int]",
        "output": "float",
        "purpose": (
            "Sum of Euclidean distances along the cyclic tour. Faster than "
            "calling tools['objective'] when you don't need a feasibility "
            "check. O(n)."
        ),
    },
    {
        "name": "k_nearest",
        "input": "node: int, k: int = 10",
        "output": "list[int]",
        "purpose": (
            "Top-k nearest neighbors of `node` (excluding itself), sorted by "
            "distance ascending. Precomputed once per instance. Useful for "
            "restricting 2-opt to candidate lists (cuts O(n^2) to O(n*k))."
        ),
    },
]
