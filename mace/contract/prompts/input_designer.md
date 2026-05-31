You are designing the INPUT contract (I) of an ISTH combinatorial-optimization problem.

# Problem description

{nl}

# Raw sample instance file(s)

These are the ACTUAL bytes your `load_data` must parse. Study the exact layout
before writing any code.

```
{sample}
```

# File-format discipline (read carefully — most failures happen here)

The problem description above explains the problem SEMANTICS, not the file
layout. Your `load_data` must match the REAL bytes shown above, not what you
assume the format "should" be. Before writing the parser, work out on the raw
sample:
- How many whitespace-separated values are on the FIRST line, and what each one
  means (do NOT assume; count them in the sample).
- For each record/entity, exactly how many values its line(s) carry, and whether
  a single logical record SPANS MULTIPLE LINES (e.g. a row of a matrix that wraps
  across lines until N values are collected).
- Whether the file holds ONE instance or MULTIPLE concatenated instances.
- Parse by consuming a flat token stream / line cursor when records wrap lines —
  do not assume one logical record == one physical line.

# Worked example (a correct, existing problem's config.py)

```python
{example}
```

# Your task

Produce a single fenced ```python block that defines:

1. A leading comment block (lines starting with `#`) listing each input field name and its type — this is the input schema.
2. `DESCRIPTION` — a clean, self-contained string restating the problem in plain language.
3. `load_data(path)` — a function that reads an instance file at `path` and returns a `dict` (or a `list` of `dict` for multi-case files) whose keys are exactly the problem's input fields shown in the schema comment.  The parser must parse the EXACT file format shown in the raw sample(s) above without raising.

Rules:
- Output ONLY one ```python block. No prose before or after it.
- Every key declared in the schema comment must appear in the returned dict.
- Do not invent keys that are not present in the instance file.
- Your `load_data` MUST run on the raw sample shown above without raising
  (IndexError/ValueError = you mis-read the format; re-derive it from the bytes).
