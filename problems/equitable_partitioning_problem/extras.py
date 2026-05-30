"""Per-problem extras for CO-Bench Equitable Partitioning Problem.

Partition N individuals (each a binary attribute vector) into EXACTLY k=8 groups
so that the per-attribute count-of-1s is balanced across groups.

Objective (from eval_func, lower is better):
    score = sum_j sum_g | count[g, j] - mean_j |
where count[g, j] = number of individuals in group g with a 1 in attribute j,
and mean_j = (sum_i data[i, j]) / k.

There are no scalar item weights in this problem; each item has an *attribute
vector*. We expose `item_weight(i)` as the row-sum (a scalar proxy useful for
size-balancing heuristics), and `item_attributes(i)` for the full vector.

Tool groups:
  (1) Queries:      n_items, n_attributes, k_groups, item_attributes,
                    item_weight, attribute_totals
  (2) Feasibility:  group_size, group_attribute_counts,
                    is_valid_partition, attribute_imbalance, total_imbalance,
                    max_group_imbalance
  (3) Construction: round_robin_partition, greedy_balanced_split,
                    balanced_kmeans
  (4) Improvement:  apply_swap_items_across_groups
  (5) Exact heavy:  ilp_partition
"""
from __future__ import annotations
import random
import time
from typing import Optional, Iterable

import numpy as np


# eval_func hard-codes k = 8 groups; mirror that here.
_K_GROUPS = 8


def extra_tools(instance: dict) -> dict:
    """Factory: returns EPP-specific tool callables for `instance`.

    Instance schema (from CO-Bench EPP load_data):
      - data: list[list[int]] of shape (n_items, n_attributes), values in {0,1}.
    """
    data_list = instance["data"]
    A = np.asarray(data_list, dtype=np.int64)
    if A.ndim != 2:
        # tolerate empty / degenerate cases
        A = A.reshape(len(data_list), -1) if len(data_list) > 0 else np.zeros((0, 0), dtype=np.int64)
    n_items_v = int(A.shape[0])
    n_attr_v = int(A.shape[1]) if A.ndim == 2 else 0
    k = _K_GROUPS

    # Per-attribute totals and per-attribute mean count.
    attr_totals_v = A.sum(axis=0).astype(np.int64) if n_attr_v > 0 else np.zeros(0, dtype=np.int64)
    # mean per attribute (float); kept for objective math.
    attr_mean_v = (attr_totals_v / k) if n_attr_v > 0 else np.zeros(0, dtype=np.float64)

    # Per-item attribute sum (useful as a scalar "weight" proxy).
    item_sums_v = A.sum(axis=1).astype(np.int64) if n_items_v > 0 else np.zeros(0, dtype=np.int64)

    # ==================================================================
    # Helpers
    # ==================================================================
    def _to_zero_indexed_groups(partition):
        """Accept either 1-indexed (eval_func format) or 0-indexed assignment
        of length n_items. Returns a list[int] of zero-indexed group ids in
        [0, k). Raises ValueError on shape / range issues."""
        if partition is None:
            raise ValueError("partition is None")
        try:
            arr = list(partition)
        except TypeError:
            raise ValueError(f"partition is not iterable: {type(partition).__name__}")
        if len(arr) != n_items_v:
            raise ValueError(
                f"partition length {len(arr)} != n_items {n_items_v}")
        out = []
        # Decide indexing from the data: if any group id is 0, treat as
        # 0-indexed; else if max >= k+1 or min == 1, treat as 1-indexed.
        ints = []
        for i, g in enumerate(arr):
            try:
                gi = int(g)
            except Exception:
                raise ValueError(f"non-integer group at index {i}: {g!r}")
            ints.append(gi)
        if not ints:
            return []
        lo, hi = min(ints), max(ints)
        if lo >= 1 and hi <= k:
            # 1-indexed (eval_func convention).
            for gi in ints:
                out.append(gi - 1)
        elif lo >= 0 and hi <= k - 1:
            out = list(ints)
        else:
            # Mixed / out-of-range: try to coerce. Prefer 1-indexed shift if
            # min >= 1, else clamp.
            if lo >= 1:
                out = [gi - 1 for gi in ints]
            else:
                out = list(ints)
            for gi in out:
                if not (0 <= gi < k):
                    raise ValueError(
                        f"group id {gi} out of range [0, {k}) after normalization"
                    )
        return out

    def _to_one_indexed(zero_idx_partition):
        return [int(g) + 1 for g in zero_idx_partition]

    def _group_counts(zero_idx_partition) -> np.ndarray:
        """Return (k, n_attr) matrix of attribute-1 counts per group."""
        C = np.zeros((k, n_attr_v), dtype=np.int64)
        if n_attr_v == 0:
            return C
        for i, g in enumerate(zero_idx_partition):
            C[g] += A[i]
        return C

    # ==================================================================
    # (1) Queries
    # ==================================================================
    def n_items() -> int:
        return n_items_v

    def n_attributes() -> int:
        return n_attr_v

    def k_groups() -> int:
        return k

    def item_attributes(i: int) -> list:
        ii = int(i)
        if not (0 <= ii < n_items_v):
            raise ValueError(f"item={i} out of range [0, {n_items_v})")
        return A[ii].tolist()

    def item_weight(i: int) -> int:
        """Sum of binary attributes for item i. Useful scalar weight proxy
        for size-balancing heuristics; the underlying objective is on per-
        attribute counts, not on this scalar."""
        ii = int(i)
        if not (0 <= ii < n_items_v):
            raise ValueError(f"item={i} out of range [0, {n_items_v})")
        return int(item_sums_v[ii])

    def attribute_totals() -> list:
        """Total number of 1s per attribute across all items. The ideal
        per-group count per attribute is attribute_totals()[j] / k."""
        return attr_totals_v.tolist()

    # ==================================================================
    # (2) Feasibility primitives
    # ==================================================================
    def group_size(g: int, partition) -> int:
        """Number of items in group `g` under `partition`. Accepts 1- or
        0-indexed group ids in `partition`; `g` is interpreted in the SAME
        convention as `partition` (1-indexed if any value in partition is 1
        but never 0, else 0-indexed)."""
        gi_raw = int(g)
        # Decide convention from partition.
        try:
            zp = _to_zero_indexed_groups(partition)
        except Exception:
            return 0
        # Determine if input was 1-indexed; if `g` >= 1 and any item in
        # partition is >= 1 and none is 0, assume 1-indexed g.
        flat = [int(x) for x in partition]
        if flat and min(flat) >= 1 and max(flat) <= k:
            gi = gi_raw - 1
        else:
            gi = gi_raw
        if not (0 <= gi < k):
            return 0
        return int(sum(1 for z in zp if z == gi))

    def group_attribute_counts(partition) -> list:
        """For each group g (in 1..k order), return a list[int] of length
        n_attributes giving the per-attribute 1-count of items in group g.
        Output shape: list of k lists, each of length n_attributes."""
        zp = _to_zero_indexed_groups(partition)
        C = _group_counts(zp)
        return [C[g].tolist() for g in range(k)]

    def is_valid_partition(partition) -> bool:
        """True iff `partition` has length n_items, every entry is an integer
        in [1, k] (or [0, k-1]), AND every group id appears at least once.
        Mirrors eval_func's hard requirement that exactly k=8 groups exist."""
        try:
            zp = _to_zero_indexed_groups(partition)
        except Exception:
            return False
        if len(zp) != n_items_v:
            return False
        seen = set(zp)
        if len(seen) != k:
            return False
        return all(0 <= g < k for g in zp)

    def attribute_imbalance(j: int, partition) -> float:
        """Imbalance for a single attribute j: sum_g |count[g, j] - mean_j|.
        This is the per-attribute contribution to the total objective."""
        ji = int(j)
        if not (0 <= ji < n_attr_v):
            raise ValueError(f"attribute={j} out of range [0, {n_attr_v})")
        zp = _to_zero_indexed_groups(partition)
        C = _group_counts(zp)
        col = C[:, ji].astype(np.float64)
        mean = float(attr_mean_v[ji])
        return float(np.abs(col - mean).sum())

    def total_imbalance(partition) -> float:
        """Sum of attribute_imbalance over all attributes; matches eval_func's
        returned score exactly. (Useful when you want the score without
        invoking the framework's `objective` wrapper.)"""
        zp = _to_zero_indexed_groups(partition)
        if n_attr_v == 0:
            return 0.0
        C = _group_counts(zp).astype(np.float64)
        diff = np.abs(C - attr_mean_v[None, :])
        return float(diff.sum())

    def max_group_imbalance(partition) -> float:
        """Largest single |count[g, j] - mean_j| over all (g, j). Useful as
        a worst-case proxy: minimizing it tends to flatten outlier groups."""
        zp = _to_zero_indexed_groups(partition)
        if n_attr_v == 0:
            return 0.0
        C = _group_counts(zp).astype(np.float64)
        diff = np.abs(C - attr_mean_v[None, :])
        return float(diff.max())

    # ==================================================================
    # (3) Construction
    # ==================================================================
    def round_robin_partition(seed: Optional[int] = None) -> list:
        """Shuffle items (deterministic if seed given) and deal them out
        round-robin to the k groups. Guarantees |group_size_i - group_size_j|
        <= 1 and that all k groups are non-empty (assuming n_items >= k).
        Returns a 1-indexed assignment list of length n_items."""
        rng = random.Random(seed) if seed is not None else random.Random()
        order = list(range(n_items_v))
        rng.shuffle(order)
        zp = [0] * n_items_v
        for slot, i in enumerate(order):
            zp[i] = slot % k
        # Ensure all k groups exist if n_items >= k (round-robin guarantees this).
        return _to_one_indexed(zp)

    def greedy_balanced_split(seed: Optional[int] = None) -> list:
        """Greedy attribute-balanced construction: process items in random
        order; for each item, place it in the group that, AFTER adding it,
        minimizes the total imbalance objective. Often noticeably better
        than round-robin. Returns a 1-indexed assignment of length n_items.

        Always uses all k=8 groups: the first k items in the shuffled order
        are forcibly placed in groups 0..k-1, then greedy from item k onward."""
        rng = random.Random(seed) if seed is not None else random.Random()
        order = list(range(n_items_v))
        rng.shuffle(order)

        zp = [-1] * n_items_v
        C = np.zeros((k, n_attr_v), dtype=np.int64)

        # First k items: force one per group, in shuffle order.
        for slot in range(min(k, n_items_v)):
            i = order[slot]
            g = slot
            zp[i] = g
            if n_attr_v > 0:
                C[g] += A[i]

        # Remaining items: pick the group minimizing post-add imbalance.
        for idx in range(k, n_items_v):
            i = order[idx]
            best_g = 0
            best_score = float("inf")
            if n_attr_v == 0:
                # Only size matters; pick smallest group.
                sizes = [sum(1 for z in zp if z == g) for g in range(k)]
                best_g = int(np.argmin(sizes))
            else:
                row = A[i]
                # Vectorised: for each candidate group g, compute new column
                # sums and the resulting L1 imbalance.
                # delta = sum_j |C[g, j] + row[j] - mean_j| - |C[g, j] - mean_j|
                cur = np.abs(C - attr_mean_v[None, :])  # (k, n_attr)
                hyp = np.abs((C + row[None, :]) - attr_mean_v[None, :])  # (k, n_attr)
                delta_per_g = (hyp - cur).sum(axis=1)  # (k,)
                best_g = int(np.argmin(delta_per_g))
                best_score = float(delta_per_g[best_g])
            zp[i] = best_g
            if n_attr_v > 0:
                C[best_g] += A[i]

        return _to_one_indexed(zp)

    def balanced_kmeans(seed: Optional[int] = None, max_iters: int = 30) -> list:
        """K-means-like construction respecting near-equal group SIZES.

        Centroids live in attribute space (each centroid is the mean attribute
        vector of items currently in the group). Each iteration:
          1) Sort items by ascending min-distance to any centroid (most
             "decided" first).
          2) Assign each item to its nearest centroid that still has room
             (cap_g = ceil(n / k) or floor(n / k) so total capacity = n).
          3) Recompute centroids; stop when assignment unchanged.

        Guarantees all k groups are populated when n_items >= k. Returns a
        1-indexed assignment list. Note: this targets SIZE balance and uses
        L2 distance on attributes; the actual objective (per-attribute
        count L1) is correlated but not identical -- usually a decent warm
        start that you should follow with apply_swap_items_across_groups.
        """
        rng = random.Random(seed) if seed is not None else random.Random()
        if n_items_v == 0:
            return []
        if n_items_v < k:
            # Not enough items to fill k groups -- degrade to round-robin
            # (the resulting assignment will fail is_valid_partition, but
            # we return *something* and let caller decide).
            return round_robin_partition(seed)

        # Per-group capacity: largest = ceil(n/k), smallest = floor(n/k).
        # Build capacity vector with the largest size attached to the first
        # n % k groups.
        base = n_items_v // k
        rem = n_items_v - base * k
        caps = [base + 1 if g < rem else base for g in range(k)]

        # Init centroids: k random items.
        seeds = rng.sample(range(n_items_v), k)
        centroids = (A[seeds].astype(np.float64) if n_attr_v > 0
                     else np.zeros((k, 0), dtype=np.float64))

        prev_zp = None
        zp = [0] * n_items_v
        for _ in range(int(max_iters)):
            # Distances (n_items, k).
            if n_attr_v == 0:
                dists = np.zeros((n_items_v, k), dtype=np.float64)
            else:
                diff = A.astype(np.float64)[:, None, :] - centroids[None, :, :]
                dists = np.sqrt((diff ** 2).sum(axis=-1))

            # Sort items by ascending min-distance: assign easy items first.
            min_d = dists.min(axis=1)
            order = sorted(range(n_items_v), key=lambda i: (min_d[i], i))

            remaining = list(caps)
            new_zp = [-1] * n_items_v
            for i in order:
                # Try groups in order of distance.
                ranking = np.argsort(dists[i])
                for g in ranking:
                    gi = int(g)
                    if remaining[gi] > 0:
                        new_zp[i] = gi
                        remaining[gi] -= 1
                        break
                if new_zp[i] == -1:
                    # Should not happen since sum(caps) == n.
                    new_zp[i] = int(np.argmin(dists[i]))

            # Recompute centroids.
            if n_attr_v > 0:
                new_cent = np.zeros((k, n_attr_v), dtype=np.float64)
                cnt = np.zeros(k, dtype=np.int64)
                for i, g in enumerate(new_zp):
                    new_cent[g] += A[i]
                    cnt[g] += 1
                for g in range(k):
                    if cnt[g] > 0:
                        new_cent[g] /= cnt[g]
                centroids = new_cent

            if new_zp == prev_zp:
                zp = new_zp
                break
            prev_zp = new_zp
            zp = new_zp

        return _to_one_indexed(zp)

    # ==================================================================
    # (4) Improvement
    # ==================================================================
    def apply_swap_items_across_groups(partition, time_limit_s: float = 5.0,
                                       seed: Optional[int] = None) -> list:
        """Local search: try MOVES (item -> another group) and SWAPS (two
        items in different groups exchange) that strictly reduce the total
        imbalance. First-improvement, restarts from the top after each
        improving move. Stops at a local optimum or when time_limit_s
        expires. Returns a 1-indexed assignment.

        Preserves the 'every group used' invariant: never empties a group.
        If the input partition does NOT use all k groups, we first patch it
        by moving items from the largest group into empty groups."""
        rng = random.Random(seed) if seed is not None else random.Random()
        try:
            zp = _to_zero_indexed_groups(partition)
        except Exception:
            # Bad input -- fall back to round-robin.
            return round_robin_partition(seed)

        # Patch empty groups (so we keep k groups in use throughout).
        sizes = [0] * k
        for g in zp:
            sizes[g] += 1
        empties = [g for g in range(k) if sizes[g] == 0]
        for empty_g in empties:
            # Find largest group with >= 2 items.
            donor = max(range(k), key=lambda g: sizes[g])
            if sizes[donor] < 2:
                break  # n_items < k, cannot fix
            # Pick any item in donor.
            for i, g in enumerate(zp):
                if g == donor:
                    zp[i] = empty_g
                    sizes[donor] -= 1
                    sizes[empty_g] += 1
                    break

        if n_attr_v == 0:
            return _to_one_indexed(zp)

        # Maintain count matrix incrementally.
        C = _group_counts(zp).astype(np.float64)

        def _imbalance_from(C_arr):
            return float(np.abs(C_arr - attr_mean_v[None, :]).sum())

        cur_obj = _imbalance_from(C)
        t0 = time.time()
        safety = 0.05

        improved = True
        while improved and (time.time() - t0) < time_limit_s - safety:
            improved = False
            order = list(range(n_items_v))
            rng.shuffle(order)

            # MOVE moves.
            for i in order:
                if (time.time() - t0) >= time_limit_s - safety:
                    break
                g_from = zp[i]
                if sizes[g_from] <= 1:
                    continue  # cannot empty a group
                row = A[i].astype(np.float64)
                # Current contributions to obj from g_from row and any other.
                for g_to in range(k):
                    if g_to == g_from:
                        continue
                    # Delta: remove from g_from, add to g_to.
                    c_from = C[g_from]
                    c_to = C[g_to]
                    m = attr_mean_v
                    old = np.abs(c_from - m).sum() + np.abs(c_to - m).sum()
                    new = (np.abs(c_from - row - m).sum()
                           + np.abs(c_to + row - m).sum())
                    if new < old - 1e-12:
                        # Apply move.
                        C[g_from] -= row
                        C[g_to] += row
                        sizes[g_from] -= 1
                        sizes[g_to] += 1
                        zp[i] = g_to
                        cur_obj += float(new - old)
                        improved = True
                        break
                if improved:
                    break

            if improved:
                continue

            # SWAP moves (i, j) with different current groups.
            for ii in range(len(order)):
                if (time.time() - t0) >= time_limit_s - safety:
                    break
                i = order[ii]
                gi = zp[i]
                ri = A[i].astype(np.float64)
                m = attr_mean_v
                for jj in range(ii + 1, len(order)):
                    j = order[jj]
                    gj = zp[j]
                    if gi == gj:
                        continue
                    rj = A[j].astype(np.float64)
                    c_i = C[gi]
                    c_j = C[gj]
                    old = np.abs(c_i - m).sum() + np.abs(c_j - m).sum()
                    new = (np.abs(c_i - ri + rj - m).sum()
                           + np.abs(c_j - rj + ri - m).sum())
                    if new < old - 1e-12:
                        C[gi] = C[gi] - ri + rj
                        C[gj] = C[gj] - rj + ri
                        zp[i] = gj
                        zp[j] = gi
                        cur_obj += float(new - old)
                        improved = True
                        break
                if improved:
                    break

        return _to_one_indexed(zp)

    # ==================================================================
    # (5) Exact / heavy
    # ==================================================================
    def ilp_partition(time_limit_s: float = 30.0) -> Optional[list]:
        """Solve the equitable-partitioning ILP exactly with CBC.

        Variables:
          x[i, g] in {0,1}     : item i assigned to group g, for g in 0..k-1
          d[g, j] >= 0 (int)   : auxiliary, equals |k*count[g, j] - m_j|
                                 (we work with k*count - m_j to keep all
                                 coefficients integer; the true objective is
                                 (1/k) * sum d.)

        Constraints:
          sum_g x[i, g] = 1                                        forall i
          sum_i x[i, g] >= 1                                       forall g
              (every group used; matches eval_func's k=8 requirement)
          d[g, j] >= k * sum_i x[i, g] * data[i, j] - m_j          forall g, j
          d[g, j] >= m_j - k * sum_i x[i, g] * data[i, j]          forall g, j

        Symmetry-breaking: lex-sort groups by smallest item index
          (first item -> group 0, then for each next item allow group up to
          max-used-so-far + 1). We approximate cheaply by fixing item 0 to
          group 0 only.

        Objective: minimize (1/k) * sum_{g, j} d[g, j].
        Returns 1-indexed assignment, or None on failure / no solution.
        """
        try:
            from mip import Model, BINARY, INTEGER, MINIMIZE, xsum, OptimizationStatus
        except Exception:
            return None
        if n_items_v == 0 or n_attr_v == 0:
            return None
        if n_items_v < k:
            # Cannot satisfy "exactly k groups".
            return None

        m = Model(sense=MINIMIZE)
        m.verbose = 0
        m.max_seconds = float(time_limit_s)

        x = {(i, g): m.add_var(var_type=BINARY, name=f"x_{i}_{g}")
             for i in range(n_items_v) for g in range(k)}
        d = {(g, j): m.add_var(var_type=INTEGER, lb=0,
                               name=f"d_{g}_{j}")
             for g in range(k) for j in range(n_attr_v)}

        # Objective: minimize sum d (1/k is constant scaling).
        m.objective = xsum(d[g, j] for g in range(k) for j in range(n_attr_v))

        # Each item in exactly one group.
        for i in range(n_items_v):
            m += xsum(x[i, g] for g in range(k)) == 1, f"one_g_{i}"

        # Every group must contain at least one item.
        for g in range(k):
            m += xsum(x[i, g] for i in range(n_items_v)) >= 1, f"used_{g}"

        # Symmetry breaking: anchor item 0 to group 0.
        m += x[0, 0] == 1, "anchor_0"

        # Imbalance linearization (in scaled integer form, factor k).
        # k * count[g, j] = k * sum_i x[i, g] * A[i, j]
        # d[g, j] >= k * count - m_j
        # d[g, j] >= m_j - k * count
        for g in range(k):
            for j in range(n_attr_v):
                kc = xsum(int(k * A[i, j]) * x[i, g]
                          for i in range(n_items_v) if A[i, j] != 0)
                mj = int(attr_totals_v[j])
                m += d[g, j] >= kc - mj, f"absp_{g}_{j}"
                m += d[g, j] >= mj - kc, f"absn_{g}_{j}"

        status = m.optimize()
        if status not in (OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE):
            return None
        if m.num_solutions < 1:
            return None

        zp = [-1] * n_items_v
        for i in range(n_items_v):
            for g in range(k):
                val = x[i, g].x
                if val is not None and val > 0.5:
                    zp[i] = g
                    break
            if zp[i] == -1:
                return None
        return _to_one_indexed(zp)

    return {
        # (1) Queries
        "n_items": n_items,
        "n_attributes": n_attributes,
        "k_groups": k_groups,
        "item_attributes": item_attributes,
        "item_weight": item_weight,
        "attribute_totals": attribute_totals,
        # (2) Feasibility primitives
        "group_size": group_size,
        "group_attribute_counts": group_attribute_counts,
        "is_valid_partition": is_valid_partition,
        "attribute_imbalance": attribute_imbalance,
        "total_imbalance": total_imbalance,
        "max_group_imbalance": max_group_imbalance,
        # (3) Construction
        "round_robin_partition": round_robin_partition,
        "greedy_balanced_split": greedy_balanced_split,
        "balanced_kmeans": balanced_kmeans,
        # (4) Improvement
        "apply_swap_items_across_groups": apply_swap_items_across_groups,
        # (5) Exact / heavy
        "ilp_partition": ilp_partition,
    }


EXTRA_TOOLS_DESCRIPTION = [
    # ----- Queries -----
    {
        "name": "n_items",
        "input": "(no args)",
        "output": "int",
        "purpose": "Number of individuals (rows in instance['data']).",
    },
    {
        "name": "n_attributes",
        "input": "(no args)",
        "output": "int",
        "purpose": "Number of binary attributes per individual.",
    },
    {
        "name": "k_groups",
        "input": "(no args)",
        "output": "int",
        "purpose": (
            "Number of groups the partition MUST use (8 for this task -- "
            "eval_func raises if your assignment uses anything else)."
        ),
    },
    {
        "name": "item_attributes",
        "input": "i: int",
        "output": "list[int]",
        "purpose": (
            "Binary attribute vector for item i (0-indexed). Length == "
            "n_attributes; entries in {0,1}."
        ),
    },
    {
        "name": "item_weight",
        "input": "i: int",
        "output": "int",
        "purpose": (
            "Scalar 'weight' proxy: sum of item i's binary attributes "
            "(== number of 1s). The true objective is NOT a function of "
            "this scalar alone -- it is per-attribute -- but row-sums are a "
            "useful coarse signal for size-balancing heuristics."
        ),
    },
    {
        "name": "attribute_totals",
        "input": "(no args)",
        "output": "list[int]",
        "purpose": (
            "Total number of 1s per attribute across all items. The IDEAL "
            "per-group count for attribute j is attribute_totals()[j] / k. "
            "Use this to spot attributes whose totals are NOT divisible by "
            "k (those contribute an unavoidable floor to the imbalance)."
        ),
    },
    # ----- Feasibility primitives -----
    {
        "name": "group_size",
        "input": "g: int, partition: list[int]",
        "output": "int",
        "purpose": (
            "Number of items in group g under `partition`. `g` follows the "
            "same indexing convention as the values in `partition` (1-indexed "
            "as expected by eval_func, or 0-indexed if you prefer; both work)."
        ),
    },
    {
        "name": "group_attribute_counts",
        "input": "partition: list[int]",
        "output": "list[list[int]]",
        "purpose": (
            "Returns a (k, n_attributes) matrix as nested lists, where "
            "entry [g][j] = number of items in group (g+1) (i.e., 1-indexed "
            "group g+1) whose attribute j == 1. These are exactly the counts "
            "the eval_func metric is computed from."
        ),
    },
    {
        "name": "is_valid_partition",
        "input": "partition: list[int]",
        "output": "bool",
        "purpose": (
            "True iff `partition` has length n_items, every value is an int "
            "in [1, k] (or [0, k-1]), AND every one of the k groups is used "
            "at least once. Fast feasibility check -- eval_func raises if "
            "the number of distinct groups != k=8, so check this first."
        ),
    },
    {
        "name": "attribute_imbalance",
        "input": "j: int, partition: list[int]",
        "output": "float",
        "purpose": (
            "Imbalance contributed by attribute j alone: sum_g |count[g, j] "
            "- mean_j| where mean_j = attribute_totals()[j] / k. Useful for "
            "diagnosing which attributes drive the objective."
        ),
    },
    {
        "name": "total_imbalance",
        "input": "partition: list[int]",
        "output": "float",
        "purpose": (
            "Sum of attribute_imbalance over all attributes. EQUAL to "
            "eval_func(data, assignment), but skips the framework wrapping "
            "in tools['objective']. Lower is better."
        ),
    },
    {
        "name": "max_group_imbalance",
        "input": "partition: list[int]",
        "output": "float",
        "purpose": (
            "Largest single |count[g, j] - mean_j| over all (g, j) pairs. "
            "A worst-case proxy -- minimizing it tends to flatten outlier "
            "groups even when total imbalance is similar."
        ),
    },
    # ----- Construction -----
    {
        "name": "round_robin_partition",
        "input": "seed: int | None = None",
        "output": "list[int]",
        "purpose": (
            "Shuffle items (deterministic if seed given) and deal them out "
            "round-robin into k groups. Guarantees group sizes differ by at "
            "most 1 and that every group is used. Returns a 1-indexed "
            "assignment list of length n_items. Cheap baseline."
        ),
    },
    {
        "name": "greedy_balanced_split",
        "input": "seed: int | None = None",
        "output": "list[int]",
        "purpose": (
            "Greedy objective-aware construction: visit items in random "
            "order; for each item, place it in the group whose post-add "
            "total imbalance is smallest. The first k items are forced into "
            "groups 1..k so all groups get used. Typically MUCH better than "
            "round-robin -- recommended warm start."
        ),
    },
    {
        "name": "balanced_kmeans",
        "input": "seed: int | None = None, max_iters: int = 30",
        "output": "list[int]",
        "purpose": (
            "K-means-like construction with SIZE caps (each group capped at "
            "ceil(n/k) or floor(n/k) so total capacity = n). Centroids are "
            "mean attribute vectors; items are assigned in order of "
            "confidence (smallest min-distance first) to the nearest group "
            "with remaining capacity. Iterates up to max_iters times. "
            "Returns 1-indexed assignment. Optimizes attribute-space "
            "compactness (L2), which correlates with -- but is not identical "
            "to -- the L1 count-imbalance objective; usually a decent warm "
            "start to follow with apply_swap_items_across_groups."
        ),
    },
    # ----- Improvement -----
    {
        "name": "apply_swap_items_across_groups",
        "input": "partition: list[int], time_limit_s: float = 5.0, seed: int | None = None",
        "output": "list[int]",
        "purpose": (
            "Local search on `partition`: alternates between MOVE (item -> "
            "different group) and SWAP (exchange two items in different "
            "groups), first-improvement, until no improving move exists or "
            "time_limit_s elapses. Never empties a group (preserves the "
            "k-groups-used invariant); auto-patches empty groups in the "
            "input by donating from the largest group. Returns 1-indexed "
            "assignment. Strong improvement step on any warm start."
        ),
    },
    # ----- Exact / heavy -----
    {
        "name": "ilp_partition",
        "input": "time_limit_s: float = 30.0",
        "output": "list[int] | None",
        "purpose": (
            "Exact EPP ILP via python-mip / CBC. Variables x[i, g] in {0,1} "
            "with auxiliary d[g, j] linearizing |k*count[g, j] - "
            "attribute_total[j]| (scaled by k to stay integer; objective "
            "equals k * true_imbalance, same minimizer). Constraints: each "
            "item in exactly one group; every group used at least once; "
            "x[0,0] = 1 as a cheap symmetry break. Returns 1-indexed "
            "assignment within time_limit_s, or None if the solver returns "
            "nothing. Practical up to ~80 items x 30 attributes; for larger "
            "instances combine greedy_balanced_split + "
            "apply_swap_items_across_groups."
        ),
    },
]
