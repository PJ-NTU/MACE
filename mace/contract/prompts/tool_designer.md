You are designing the core of the TOOL library (T) for an ISTH combinatorial-optimization problem: the feasibility checker and the objective evaluator. There is NO pre-existing evaluator — you are authoring the formal definition of "feasible" and "cost" for this problem from its description.

# Problem description

{nl}

# Locked input schema (I) — the `instance` dict

{input_schema}

# Locked output schema (O) — the `solution` dict

{output_schema}

# Locked trivial placeholder solution (a known-feasible baseline)

```python
{placeholder}
```

# Your task

Output ONE fenced ```python block that defines EXACTLY these two functions:

1. `is_feasible(instance, solution)` — returns `(True, None)` if the solution
   satisfies EVERY constraint in the problem description, else `(False, message)`
   at the first violation. Put a short `# C1`, `# C2`, ... comment on each check
   naming the constraint, so the logic stays scannable. Do NOT raise; return the
   tuple.
2. `objective(instance, solution)` — returns the objective value as a float,
   **LOWER IS BETTER** (if the problem is a maximization, return the negated /
   reciprocal value so that smaller = better). Assume the solution is feasible.

Both take the `instance` dict (shape I) and the `solution` dict (shape O).

# Example of the exact shape expected (for a different, tiny problem)

```python
def is_feasible(instance, solution):
    picked = solution.get("picked", [])
    # C1: must pick at least one item
    if len(picked) < 1:
        return False, "C1: must pick at least one item"
    n = instance["n"]
    # C2: every index is a valid item
    if any((not isinstance(i, int)) or i < 0 or i >= n for i in picked):
        return False, "C2: invalid item index"
    return True, None

def objective(instance, solution):
    # LOWER IS BETTER: total weight of the picked items
    return float(sum(instance["weights"][i] for i in solution["picked"]))
```

# Rules

- Output ONLY one ```python block (plus any `import` it needs at module top). No prose outside it.
- `is_feasible` must enforce EVERY constraint in the description — before you finish,
  re-read the description and self-check that each constraint has a matching `# Ck` check.
- The trivial placeholder solution above MUST be accepted by your `is_feasible`.
- Use only the Python standard library and numpy.
