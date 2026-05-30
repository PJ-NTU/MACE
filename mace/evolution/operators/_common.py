"""Shared helpers for Stage Two operators.

- extract_python(text): strip markdown fences and return Python source.
- AVAILABLE_IMPORTS_HINT: appended to every generation prompt to suppress
  hallucinated imports of project-specific modules.
- SYSTEM_PREAMBLE: stylistic header explaining the role.
- format_heuristic(code, **stats): pretty wrap a heuristic for prompts.

NO-TOOLS MODE: when env var MACE_NO_TOOLS=1, prompts instruct the LLM that
the `tools` dict will be EMPTY at runtime and to implement all logic from
scratch. The 3-arg `solve(instance, tools, time_limit_s)` signature is kept
for API compatibility; framework.run_solve passes `tools={}`.
"""
from __future__ import annotations
import os
import re
from typing import Optional

_NO_TOOLS = os.environ.get("MACE_NO_TOOLS") == "1"

_NO_TOOLS_WARNING = """\
**IMPORTANT — NO-TOOLS MODE.** The `tools` dict will be **empty `{}`** when
your `solve` is called. Do NOT call `tools['name'](...)` for any name. Do NOT
rely on any helper from a "tools" section (there is none in this run).
Implement every routine you need (feasibility checks, scoring, CBC fallback,
etc.) entirely from scratch using only `numpy` + Python stdlib.
"""

AVAILABLE_IMPORTS_HINT = (
    """\
You may import only from the Python standard library and numpy. Do NOT import
from any project-specific path (no `from src...`, `from problems...`, etc.).
Common useful imports: `import time`, `import math`, `import random`,
`import heapq`, `import itertools`, `from collections import deque`,
`import numpy as np`. The `tools` dict will be EMPTY at runtime — do NOT
reference it inside `solve`.
"""
    if _NO_TOOLS else
    """\
You may import only from the Python standard library and numpy. Do NOT import
from any project-specific path (no `from src...`, `from problems...`, etc.).
Common useful imports: `import time`, `import math`, `import random`,
`import heapq`, `import itertools`, `from collections import deque`,
`import numpy as np`. The `tools` dict is passed as an argument to `solve` —
do NOT try to import it.
"""
)

SYSTEM_PREAMBLE = (
    """\
You are an expert researcher in combinatorial optimization heuristics. You
write clean, efficient, correct Python that respects the ISTH interface:

    def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
        ...

Your `solve` must self-monitor wall-clock time against `time_limit_s` and
return a feasible solution before the budget elapses.

**NO-TOOLS MODE** is active for this run. The `tools` dict argument will be
empty (`{}`). Do not call anything inside it. Implement every helper you need
yourself from the Python stdlib and numpy.
"""
    if _NO_TOOLS else
    """\
You are an expert researcher in combinatorial optimization heuristics. You
write clean, efficient, correct Python that respects the ISTH interface:

    def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:
        ...

Your `solve` must self-monitor wall-clock time against `time_limit_s` and
return a feasible solution before the budget elapses.
"""
)


_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_python(text: str) -> str:
    """Return the largest fenced Python block in `text`, or `text` itself if no fences."""
    if not text:
        return ""
    blocks = _FENCE_RE.findall(text)
    if not blocks:
        return text.strip()
    # take the longest block (heuristics: LLMs sometimes emit a short example fence too)
    return max(blocks, key=len).strip()


def format_heuristic(code: str, header: Optional[str] = None) -> str:
    """Wrap heuristic code in a fenced block for inclusion in prompts."""
    h = f"# {header}\n" if header else ""
    return f"{h}```python\n{code.strip()}\n```"


TOOLS_PROLOGUE = """\
These functions are accessible via `tools['<name>'](...)` from inside your
`solve`. They are **optional** -- you can use them to keep your algorithm
correct and efficient, or implement everything yourself from scratch if you
prefer. Using them is encouraged where they save work.
"""


def _render_tools_section(tools: list) -> str:
    """Render spec.tools_description into prompt markdown."""
    if not tools:
        return ""
    lines = ["# Available tools (optional helpers)", TOOLS_PROLOGUE]
    for t in tools:
        name = t.get("name", "?")
        inp = t.get("input", "...")
        out = t.get("output", "...")
        purpose = t.get("purpose", "")
        lines.append(f"- `{name}({inp}) -> {out}`")
        if purpose:
            lines.append(f"  {purpose}")
    return "\n".join(lines) + "\n"


def _render_feasibility_section(doc: str) -> str:
    if not doc:
        return ""
    return f"# Feasibility check\n{doc.strip()}\n"


def build_prompt(spec, body: str) -> str:
    """Common prompt scaffolding for all generation / repair operators.

    Sections in order (each one omitted if its data is empty):
      1. System preamble — role + ISTH signature
      2. Problem name + description
      3. Feasibility check (from spec.feasibility_doc)
      4. Available tools (from spec.tools_description, with optional-use prologue)
         -- SKIPPED entirely when MACE_NO_TOOLS=1
      5. Heuristic interface (spec.starter_code)
      6. Operator-specific task body
      7. Output rules + available-imports hint
    """
    parts = [SYSTEM_PREAMBLE]
    if _NO_TOOLS:
        # Big warning right after preamble so it can't be missed
        parts.append(_NO_TOOLS_WARNING)
    parts.append(f"# Problem: {spec.name}\n{spec.description}")
    fea = _render_feasibility_section(getattr(spec, "feasibility_doc", ""))
    if fea:
        parts.append(fea)
    if not _NO_TOOLS:
        tools_sec = _render_tools_section(getattr(spec, "tools_description", []))
        if tools_sec:
            parts.append(tools_sec)
    parts.append(f"# Heuristic interface (starter_code)\n```python\n{spec.starter_code}\n```")
    parts.append(body)
    parts.append(
        "# Output rules\n"
        "Return ONLY the complete Python source of the new `solve` function (and any\n"
        "helper functions / imports it needs). No prose, no explanations, no markdown\n"
        "outside one fenced ```python block.\n\n"
        + AVAILABLE_IMPORTS_HINT.strip()
    )
    return "\n\n".join(parts)
