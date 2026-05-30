"""Auto-generated step-by-step is_feasible reference (LLM-rewritten
from CO-Bench eval_func). Read by spec.py to prepend to feasibility_doc."""

FEASIBILITY_STEPS_PY = r'''def is_feasible(solution):
    if not isinstance(solution, dict):
        return False, f"solution must be dict, got {type(solution).__name__}"

    permutation = solution.get("permutation")
    batch_assignment = solution.get("batch_assignment")

    if permutation is None:
        return False, "solution missing 'permutation' key"
    if batch_assignment is None:
        return False, "solution missing 'batch_assignment' key"

    if not isinstance(permutation, (list, tuple)):
        return False, f"'permutation' must be list, got {type(permutation).__name__}"
    if not isinstance(batch_assignment, (list, tuple)):
        return False, f"'batch_assignment' must be list, got {type(batch_assignment).__name__}"

    if len(permutation) != n_jobs:
        return False, f"'permutation' length {len(permutation)} != n_jobs={n_jobs}"
    if len(batch_assignment) != n_jobs:
        return False, f"'batch_assignment' length {len(batch_assignment)} != n_jobs={n_jobs}"

    if any(not isinstance(x, int) for x in permutation):
        return False, "'permutation' entries must be int"
    if sorted(permutation) != list(range(1, n_jobs + 1)):
        return False, "'permutation' must be a valid permutation of job indices 1 through n_jobs"

    if any(not isinstance(x, int) for x in batch_assignment):
        return False, "'batch_assignment' entries must be int"
    if any(x < 1 or x > n_machines for x in batch_assignment):
        return False, f"'batch_assignment' entries must be in range [1, n_machines={n_machines}]"

    return True, None
'''
