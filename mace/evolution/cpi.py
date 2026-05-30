"""Complementary Portfolio Index (CPI).

CPI(F) = (1/M) * sum_xi min_h f_{xi,h}

Lower is better. F is the (M x N) cost matrix with F[xi, h] the cost of
heuristic h on instance xi (lower = better). Cells where evaluation failed
must be encoded as PENALTY (a large sentinel >= 1e10). If every heuristic
fails on a given instance, that row contributes PENALTY to the CPI.
"""
from __future__ import annotations
import numpy as np

PENALTY: float = 1e10


def compute_cpi(F: np.ndarray) -> float:
    if F.ndim != 2 or F.size == 0:
        raise ValueError(f"F must be a non-empty 2D array, got shape {F.shape}")
    return float(F.min(axis=1).mean())


def per_instance_best(F: np.ndarray) -> np.ndarray:
    """Returns shape (M,) — best cost for each instance."""
    return F.min(axis=1)
