"""Independent reviewer (LLM-B): audit whether a generated I/O artifact correctly
and completely captures the problem. Returns (approved, feedback)."""
from __future__ import annotations


def review(llm_client, role: str, problem_description: str, artifact_code: str,
           extra_context: str = "") -> tuple[bool, str | None]:
    prompt = (
        f"You are an INDEPENDENT reviewer auditing the {role} of a combinatorial-"
        f"optimization problem contract. Your job is to catch SUBSTANTIVE errors "
        f"only — not to polish.\n\n"
        f"# Problem description\n{problem_description}\n\n"
        + (f"{extra_context}\n\n" if extra_context else "")
        + f"# {role} under review\n```python\n{artifact_code}\n```\n\n"
        f"# Your judgment — bias toward APPROVED\n"
        f"REJECT ONLY for a substantive defect that makes the contract WRONG or "
        f"UNUSABLE, such as:\n"
        f"  - a field/key that is missing, has the wrong type, or is invented and not in the data;\n"
        f"  - the solution shape cannot represent a valid solution;\n"
        f"  - a hard constraint from the description that is simply not represented.\n"
        f"Do NOT reject for any of these (they are NOT defects):\n"
        f"  - wording/phrasing of the description or docstrings;\n"
        f"  - style, naming, comments, or how Euclidean/derived quantities are described;\n"
        f"  - hypothetical edge cases that do NOT occur in the sample data shown;\n"
        f"  - 'could be more explicit' / 'should also mention' suggestions.\n"
        f"If the artifact is correct and usable as-is, even if imperfectly worded, APPROVE. "
        f"When in doubt, APPROVE.\n\n"
        f"- If acceptable, reply with EXACTLY the single word: APPROVED\n"
        f"- Only if there is a real defect above, reply 'REJECTED:' then the concrete defect(s).\n"
    )
    resp = (llm_client.chat(prompt) or "").strip()
    if resp.upper().startswith("APPROVED"):
        return True, None
    return False, resp
