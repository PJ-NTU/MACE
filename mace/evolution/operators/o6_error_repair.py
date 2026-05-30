"""O6 — Error-Driven Repair (reactive).

Triggered when a candidate from O1-O5 fails execution (syntax error,
runtime exception, infeasible output). Reads the full error / infeasibility
report and repairs the code WITHOUT changing core logic.

Cap: I_rep = 3 retries; discard if still failing afterwards.
"""
from __future__ import annotations

from ._common import build_prompt, extract_python, format_heuristic


def repair(spec, broken_code: str, error_msg: str, llm_client) -> str:
    body = (
        f"# Broken heuristic\n{format_heuristic(broken_code)}\n\n"
        f"# Failure report\n```\n{error_msg}\n```\n\n"
        f"# Task — Error-Driven Repair (O6)\n"
        f"The heuristic above failed with the error shown. Fix it WITHOUT changing\n"
        f"the core algorithmic logic — preserve construction strategy, improvement\n"
        f"strategy, dispatch logic, etc. Touch only what the error report identifies\n"
        f"as broken. Common causes:\n"
        f"  - wrong solution dict schema (missing key, wrong type)\n"
        f"  - permutation has duplicates / wrong length\n"
        f"  - infeasibility (e.g., capacity violation, missing customer)\n"
        f"  - undeclared variable, wrong import, hallucinated module\n"
        f"  - exception in a corner case (empty input, single node, etc.)\n\n"
        f"Output the complete fixed Python source."
    )
    prompt = build_prompt(spec, body)
    response = llm_client.chat(prompt)
    return extract_python(response)
