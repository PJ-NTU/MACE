You are designing the INPUT contract (I) of an ISTH combinatorial-optimization problem.

# Problem description

{nl}

# Raw sample instance file

```
{sample}
```

# Worked example (a correct, existing problem's config.py)

```python
{example}
```

# Your task

Produce a single fenced ```python block that defines:

1. A leading comment block (lines starting with `#`) listing each input field name and its type — this is the input schema.
2. `DESCRIPTION` — a clean, self-contained string restating the problem in plain language.
3. `load_data(path)` — a function that reads an instance file at `path` and returns a `dict` (or a `list` of `dict` for multi-case files) whose keys are exactly the problem's input fields shown in the schema comment.  The parser must handle the real file format shown in the raw sample instance above.

Rules:
- Output ONLY one ```python block. No prose before or after it.
- Every key declared in the schema comment must appear in the returned dict.
- Do not invent keys that are not present in the instance file.
