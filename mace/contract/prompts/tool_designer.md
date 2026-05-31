You are designing the TOOL contract (T) — the ground-truth evaluator — of an ISTH problem.

# Problem description

{nl}

# Locked input schema

{input_schema}

# Locked output (solution) schema

{output_schema}

# Locked trivial placeholder solution

```python
{placeholder}
```

# Worked example (eval_func + feasibility_steps from an existing problem)

```python
{example}
```

# Your task

Output ONE fenced ```python block that defines exactly these three objects:

1. `eval_func(**kwargs)` — the ground-truth evaluator.
   - It receives the instance dict merged with the solution dict as keyword arguments.
   - It must check EVERY constraint stated in the problem description.
   - On ANY violation it must `raise ValueError` with a clear human-readable message identifying which constraint was broken.
   - It must return the objective cost as a non-negative `float` (lower-is-better, i.e. minimization).

2. `FEASIBILITY_STEPS_PY` — a raw string constant (assign it like `FEASIBILITY_STEPS_PY = r'''...'''`) holding the source of an `is_feasible(solution)` function. Rules for this function:
   - Label each constraint with a code `C1`, `C2`, `C3`, ... in a comment on the same line as the check, matching the constraint order in `eval_func`.
   - Return `(True, None)` when the solution is feasible.
   - Return `(False, message)` at the first violated constraint, where `message` names the constraint code and the violated value.
   - The constraints and their codes must be in 1-to-1 correspondence with the `raise ValueError` branches in `eval_func`.
   - Use the C1..Cn labelled-constraint style shown in the worked example — this is mandatory.

3. `infeasible_make_solution(instance)` — builds a solution dict that violates EXACTLY ONE constraint (any single constraint is fine). This is used only for a smoke test to confirm that `eval_func` actually rejects bad solutions. It must accept one positional argument `instance` (the dict returned by `load_data`) and return a `dict` with the same top-level keys as a normal solution.

Additional rules:
- Output ONLY one ```python block. No prose before or after it.
- `eval_func` must enforce every constraint in the problem description — do not omit any.
- `infeasible_make_solution` must return a solution that `eval_func` will reject (raise ValueError).
- The cost returned by `eval_func` on a feasible solution must be >= 0.
- Do not import anything that is not in the Python standard library.
