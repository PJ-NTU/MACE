You are PLANNING the domain helper tools (T) for an ISTH combinatorial-optimization problem — deciding WHICH reusable building blocks a solver would call (e.g. distance between nodes, route/tour cost, a delta-evaluation for a local move, a feasibility-preserving repair). You are not writing code yet; you are choosing a small, NON-OVERLAPPING set.

# Problem description

{nl}

# Locked input schema (I)

{input_schema}

# Locked output schema (O)

{output_schema}

# Your task

Propose AT MOST 3 distinct, complementary helper tools that a heuristic for THIS problem would genuinely call. Avoid overlap or duplicates — each must do something clearly different and useful.

Output ONE fenced ```python block assigning a list named `HELPERS_PLAN`, where each entry is a dict with two keys: a valid Python identifier under key name, and a one-line description under key purpose. If no helper is genuinely useful, output `HELPERS_PLAN = []`.

# Example (for a different problem)

```python
HELPERS_PLAN = [
    dict(name="tour_length", purpose="Total Euclidean length of a closed tour."),
    dict(name="two_opt_delta", purpose="Change in tour length from a 2-opt swap of two edges."),
]
```

# Rules

- Output ONLY one ```python block assigning HELPERS_PLAN. No prose outside it.
- At most 3 entries; fewer is fine; `[]` if none are useful.
