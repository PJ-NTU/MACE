You are designing the OUTPUT contract (O) of an ISTH combinatorial-optimization problem.

# Problem description

{nl}

# Locked input schema (from the Input Designer)

{input_schema}

# Locked load_data

```python
{load_data_code}
```

# Worked example (an existing problem's spec.py — study its starter_code Returns: section)

```python
{example}
```

# Your task

Output ONE fenced ```python block defining:

- `OUTPUT_SCHEMA` = a string describing the exact solution dict shape the solver must return (keys, value types, indexing conventions).
- `make_solution(instance)` returning a trivial FEASIBLE-BY-CONSTRUCTION placeholder solution built ONLY from instance data (the "do nothing" baseline). The function must return a `dict` whose keys match `OUTPUT_SCHEMA`.

Rules:
- Output ONLY one ```python block. No prose before or after it.
- `OUTPUT_SCHEMA` must be a plain string literal (no f-string, no interpolation).
- `make_solution` must use ONLY data from `instance` to build the return value — no hard-coded constants that depend on hidden problem size.
