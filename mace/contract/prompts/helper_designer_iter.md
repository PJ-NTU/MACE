You are designing the domain helper tools (T) for an ISTH combinatorial-optimization problem, ONE TOOL AT A TIME. Each helper is a reusable building block a solver would call (e.g. distance between nodes, route/tour cost, a delta-evaluation for a local move, a feasibility-preserving repair, a state query, an exact subroutine for small instances). You are not writing code yet; you are choosing the single most useful NEXT tool.

# Problem description

{nl}

# Locked input schema (I)

{input_schema}

# Locked output schema (O)

{output_schema}

# Helpers already admitted in earlier rounds (name and purpose only)

{existing}

# Your task

Propose exactly ONE new helper tool that a heuristic for THIS problem would genuinely call, and that clearly differs from every already-admitted helper above. Pick the most useful gap the existing set leaves open: construction routines, neighbourhood moves, delta evaluations, repair procedures, state queries, exact subroutines.

Output ONE fenced ```python block assigning a list named `HELPERS_PLAN` containing that single entry: a dict with a valid Python identifier under key name and a one-line description under key purpose. If nothing genuinely new remains, output `HELPERS_PLAN = []`.

# Example (for a different problem)

```python
HELPERS_PLAN = [
    dict(name="two_opt_delta", purpose="Change in tour length from a 2-opt swap of two edges."),
]
```

# Rules

- Output ONLY one ```python block assigning HELPERS_PLAN. No prose outside it.
- Exactly one entry (or `[]` if nothing new is useful).
- The entry must be clearly different from every already-admitted helper.
