"""Rank matrix and rank-derived statistics for Stage Two.

The rank matrix R has shape (M, N) with R[xi, h] = rank of heuristic h on
instance xi within the current pool, 1 = best (lowest cost).

Failure cells (cost >= PENALTY) are pushed to the worst ranks; ties broken
by lower heuristic index (np.argsort with kind='stable' over cost).
"""
from __future__ import annotations
import numpy as np

from .cpi import PENALTY


def compute_rank_matrix(F: np.ndarray) -> np.ndarray:
    if F.ndim != 2 or F.size == 0:
        raise ValueError(f"F must be a non-empty 2D array, got shape {F.shape}")
    M, N = F.shape
    R = np.empty((M, N), dtype=np.int32)
    for xi in range(M):
        order = np.argsort(F[xi], kind="stable")  # indices, best first
        ranks = np.empty(N, dtype=np.int32)
        ranks[order] = np.arange(1, N + 1)
        R[xi] = ranks
    return R


def mean_rank_per_heuristic(R: np.ndarray) -> np.ndarray:
    return R.mean(axis=0)


def complementarity_score(R: np.ndarray) -> np.ndarray:
    """C[a, b] = (1/M) * sum_xi | R[xi, a] - R[xi, b] |.

    Symmetric, zero diagonal. Large C[a,b] means a and b disagree most on
    which instances each handles best — i.e., they are complementary.
    """
    M, N = R.shape
    diffs = np.abs(R[:, :, None].astype(np.float64) - R[:, None, :].astype(np.float64))
    return diffs.mean(axis=0)


def feasibility_mask(F: np.ndarray, penalty: float = PENALTY) -> np.ndarray:
    """Returns (M, N) bool: True where the heuristic produced a real (non-penalty) cost."""
    return F < penalty
