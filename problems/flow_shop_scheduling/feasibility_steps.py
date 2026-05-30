"""Auto-generated step-by-step is_feasible reference (LLM-rewritten
from CO-Bench eval_func). Read by spec.py to prepend to feasibility_doc."""

FEASIBILITY_STEPS_PY = r'''def is_feasible(solution):
    if not isinstance(solution, dict):
        return False, f"solution must be dict, got {type(solution).__name__}"

    job_sequence = solution.get("job_sequence")
    if job_sequence is None:
        return False, "solution missing 'job_sequence' key"

    if not isinstance(job_sequence, (list, tuple)):
        return False, f"'job_sequence' must be list, got {type(job_sequence).__name__}"

    if len(job_sequence) != n:
        return False, f"'job_sequence' length {len(job_sequence)} != n={n}"

    if any(not isinstance(x, int) for x in job_sequence):
        return False, "'job_sequence' entries must be int"

    if set(job_sequence) != set(range(1, n + 1)):
        return False, f"'job_sequence' must be a permutation of [1, ..., {n}]"

    return True, None
'''
