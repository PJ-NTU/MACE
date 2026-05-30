"""MILP-based portfolio selection (MACE-v1 Eq. 8, rank-based).

Given F_pool (M_train x P), pick `n_select` heuristic indices minimizing
the average per-instance rank-of-the-best heuristic chosen from the
selected subset.

Formulation:
    e_h in {0,1}              whether h is selected
    c_{xi, h} in {0,1}        whether instance xi is "covered" by h

    min   (1/M) * sum_{xi, h} r_{xi, h} * c_{xi, h}
    s.t.  sum_h c_{xi, h} = 1                            (each instance covered once)
          c_{xi, h} <= e_h                               (only via selected)
          sum_h e_h = n_select

Backend: python-mip (https://www.python-mip.com) with the embedded CBC
solver. CBC ships inside the `mip` wheel for Windows / Mac / Linux, so
`pip install mip` is the only install step (no license, no native build).
For our pool size (P <= ~20) the model has ~M*P + P binary variables --
typically < 2000 -- and the solve completes well under a second.
"""
from __future__ import annotations
import numpy as np
from mip import Model, BINARY, MINIMIZE, xsum, OptimizationStatus

from .rank_matrix import compute_rank_matrix


def milp_select(
    F_pool: np.ndarray,
    n_select: int,
    time_limit_s: float = 60.0,
    log_to_console: bool = False,
) -> list[int]:
    if F_pool.ndim != 2:
        raise ValueError(f"F_pool must be 2D, got shape {F_pool.shape}")
    M, P = F_pool.shape
    if n_select > P:
        raise ValueError(f"n_select={n_select} > pool size {P}")
    if n_select == P:
        return list(range(P))

    R = compute_rank_matrix(F_pool)

    m = Model("milp_select", sense=MINIMIZE)
    m.verbose = 1 if log_to_console else 0
    m.max_seconds = time_limit_s

    # e_h: whether heuristic h is selected for the new portfolio
    e = [m.add_var(var_type=BINARY, name=f"e[{h}]") for h in range(P)]
    # c_{xi, h}: whether instance xi is covered by heuristic h
    c = [[m.add_var(var_type=BINARY, name=f"c[{xi},{h}]") for h in range(P)]
         for xi in range(M)]

    m.objective = xsum(
        int(R[xi, h]) * c[xi][h] for xi in range(M) for h in range(P)
    ) / M

    for xi in range(M):
        m += xsum(c[xi][h] for h in range(P)) == 1, f"cov[{xi}]"
        for h in range(P):
            m += c[xi][h] <= e[h], f"link[{xi},{h}]"
    m += xsum(e[h] for h in range(P)) == n_select, "size"

    status = m.optimize()

    if status not in (OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE):
        raise RuntimeError(f"MILP solver returned status {status}")

    # m.num_solutions reports how many incumbents found; require >= 1.
    if m.num_solutions < 1:
        raise RuntimeError("MILP found no feasible solution within time limit")

    selected = sorted(
        h for h in range(P) if e[h].x is not None and e[h].x > 0.5
    )
    if len(selected) != n_select:
        raise RuntimeError(
            f"MILP selected {len(selected)} heuristics, expected {n_select}; "
            f"status={status}, x values = {[e[h].x for h in range(P)]}"
        )
    return selected
