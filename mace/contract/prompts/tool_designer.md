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

Output ONE fenced ```python block that defines exactly these TWO objects:

1. `eval_func(**kwargs)` — the ground-truth evaluator (the heart of the tool library T).
   - It receives the instance dict merged with the solution dict as keyword arguments.
   - It must check EVERY constraint stated in the problem description, in order. A
     short `# C1`, `# C2`, ... comment on each check (naming the constraint) is
     encouraged so the logic stays scannable.
   - On ANY violation it must `raise ValueError` with a clear human-readable message identifying which constraint was broken.
   - It must return the objective cost as a non-negative `float` (lower-is-better, i.e. minimization).

2. `infeasible_make_solution(instance)` — builds a solution dict that violates EXACTLY ONE constraint (any single constraint is fine). This is used only for a smoke test to confirm that `eval_func` actually rejects bad solutions. It must accept one positional argument `instance` (the dict returned by `load_data`) and return a `dict` with the same top-level keys as a normal solution.

Additional rules:
- Output ONLY one ```python block. No prose before or after it.
- `eval_func` must enforce every constraint in the problem description — do not omit any.
  Before you finish, re-read the description and self-check that each constraint has a
  matching check in `eval_func` (this is your reflection step — do it in-line, silently).
- `infeasible_make_solution` must return a solution that `eval_func` will reject (raise ValueError).
- The cost returned by `eval_func` on a feasible solution must be >= 0.
- Do not import anything that is not in the Python standard library.
