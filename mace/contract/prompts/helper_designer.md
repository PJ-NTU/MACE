You are extending the TOOL library (T) of an ISTH combinatorial-optimization problem with a FEW domain helper functions — reusable building blocks a solver would call (e.g. distance between nodes, route/tour cost, a delta-evaluation for a local move, a feasibility-preserving repair). Keep it to AT MOST 3 helpers; only add ones a heuristic would genuinely use.

# Problem description

{nl}

# Locked input schema (I)

{input_schema}

# Locked output schema (O)

{output_schema}

# Locked feasibility checker

```python
{is_feasible}
```

# Locked objective

```python
{objective}
```

# Your task

Output ONE fenced ```python block defining 0–3 helper functions. Each helper:
- takes `instance` as its FIRST argument (the framework binds it), e.g. `def tour_length(instance, tour): ...`
- has a one-line docstring stating what it does (this becomes its tool description)
- is something a solver would actually call while constructing or improving a solution.

If no helper is genuinely useful for this problem, output an empty ```python block.

# Example (for a different problem)

```python
def tour_length(instance, tour):
    """Total Euclidean length of a closed tour (list of city indices)."""
    import math
    pts = instance["coordinates"]
    total = 0.0
    for a, b in zip(tour, tour[1:] + tour[:1]):
        total += math.hypot(pts[a][0] - pts[b][0], pts[a][1] - pts[b][1])
    return total
```

# Rules

- Output ONLY one ```python block (with any `import` the helpers need). No prose outside it.
- First argument of every helper is `instance`.
- Use only the Python standard library and numpy.
