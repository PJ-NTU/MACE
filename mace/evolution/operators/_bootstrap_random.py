"""Bootstrap helper: from-scratch random restart generator.

Not a formal Stage Two generation operator -- does NOT participate in the
uniform O1..O5 sampling of the main evolution loop. Used only by
`build_initial_portfolio` (and the cobench all-O5 fallback path) when no
parent algorithm exists yet and we need to seed the portfolio.

Parent selection: none. LLM is asked to design a heuristic from scratch.
"""
from __future__ import annotations

from ._common import build_prompt, extract_python


def generate(spec, portfolio, F, R, llm_client):
    body = (
        f"# Task -- Bootstrap (from-scratch generation)\n"
        f"Design a {spec.name} heuristic from scratch. You see ONLY the problem\n"
        f"description and the ISTH interface above -- no existing portfolio members,\n"
        f"no prior code.\n\n"
        f"Pick ANY standard CO algorithmic technique that you think is well-suited\n"
        f"to this problem (e.g., greedy + local search, savings, GRASP, simulated\n"
        f"annealing, large-neighborhood search, beam search, dynamic programming\n"
        f"over a restricted state). Be creative; this is the first heuristic in\n"
        f"the portfolio so produce something reasonable end-to-end."
    )

    prompt = build_prompt(spec, body)
    response = llm_client.chat(prompt)
    code = extract_python(response)
    return code, {"operator": "bootstrap_random"}
