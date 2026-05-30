"""Auto-generated step-by-step is_feasible reference (LLM-rewritten
from CO-Bench eval_func). Read by spec.py to prepend to feasibility_doc."""

FEASIBILITY_STEPS_PY = r'''def is_feasible(solution):
    # (1) solution dict shape
    if not isinstance(solution, dict):
        return False, f"solution must be dict, got {type(solution).__name__}"

    # (2) required keys present
    if "num_bins" not in solution:
        return False, "solution missing 'num_bins' key"
    if "bins" not in solution:
        return False, "solution missing 'bins' key"

    # (3) correct types
    num_bins = solution["num_bins"]
    bins = solution["bins"]

    if not isinstance(num_bins, int):
        return False, f"'num_bins' must be int, got {type(num_bins).__name__}"
    if not isinstance(bins, (list, tuple)):
        return False, f"'bins' must be list, got {type(bins).__name__}"

    # (4) declared num_bins matches actual bins list length
    if len(bins) != num_bins:
        return False, f"declared num_bins={num_bins} does not match len(bins)={len(bins)}"

    # (5) per-element and global constraints
    item_counts = [0] * (num_items + 1)  # 1-based; index 0 unused

    for bin_index, bin_items in enumerate(bins, start=1):
        if not isinstance(bin_items, (list, tuple)):
            return False, f"bin {bin_index} must be a list, got {type(bin_items).__name__}"
        bin_total = 0
        for item_idx in bin_items:
            if not isinstance(item_idx, int):
                return False, f"bin {bin_index} contains non-int item index: {item_idx!r}"
            if item_idx < 1 or item_idx > num_items:
                return False, f"bin {bin_index} contains invalid item index: {item_idx} (must be 1..{num_items})"
            bin_total += items[item_idx - 1]
            item_counts[item_idx] += 1
        if bin_total > bin_capacity:
            return False, f"bin {bin_index} exceeds capacity: total size {bin_total} > bin_capacity={bin_capacity}"

    for i in range(1, num_items + 1):
        if item_counts[i] != 1:
            return False, f"item {i} appears {item_counts[i]} times (expected exactly once)"

    return True, None
'''
