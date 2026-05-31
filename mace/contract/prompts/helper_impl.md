You are implementing ONE domain helper tool for an ISTH combinatorial-optimization problem.

# Problem description

{nl}

# Locked input schema (I) — the `instance` dict

{input_schema}

# Locked output schema (O) — the `solution` dict

{output_schema}

# Locked feasibility checker

```python
{is_feasible}
```

# Locked objective

```python
{objective}
```

# Helper to implement

name: {name}
purpose: {purpose}

# Your task

Write exactly this one function:

```python
def {name}(instance, ...):
    """one-line docstring restating the purpose."""
    ...
```

- The FIRST argument is `instance` (the framework binds it, so a solver calls it as `tools['{name}'](...)` WITHOUT passing the instance).
- Choose the remaining arguments to fit the purpose (e.g. a partial/whole solution, indices).
- Put any `import` it needs INSIDE the function body (so the helper is self-contained).
- Output ONLY this one function in a single ```python block. No prose outside it.
- Use only the Python standard library and numpy.
