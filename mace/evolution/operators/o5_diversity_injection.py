"""O5 — Diversity Injection.

Parent selection: feed the entire portfolio to the LLM.
LLM task: identify common decision patterns shared across the portfolio
(construction strategy, neighborhood, acceptance, etc.), then write a NEW
heuristic whose core logic differs along as many of those dimensions as
possible.
Goal: expose logical blind spots of the current portfolio.
"""
from __future__ import annotations

from ._common import build_prompt, extract_python, format_heuristic


def generate(spec, portfolio, F, R, llm_client):
    blocks = "\n\n".join(
        format_heuristic(code, header=f"Portfolio member #{i}")
        for i, code in enumerate(portfolio)
    )

    body = (
        f"# Current portfolio ({len(portfolio)} members)\n"
        f"{blocks}\n\n"
        f"# Task — Diversity Injection (O5)\n"
        f"1. Briefly identify the common decision patterns shared by most of these\n"
        f"   heuristics (e.g., 'all use nearest-neighbor construction', 'all use\n"
        f"   2-opt with first-improvement', 'none use perturbation', 'all are\n"
        f"   purely greedy').\n"
        f"2. Design a new heuristic whose CORE LOGIC differs from the portfolio\n"
        f"   along as many of those dimensions as you can identify. Pick algorithmic\n"
        f"   techniques the portfolio is NOT using.\n\n"
        f"The goal is to cover blind spots, not to outperform on every instance —\n"
        f"the result will be selected via complementary objective, so being\n"
        f"different in the right places is more valuable than being uniformly good."
    )

    prompt = build_prompt(spec, body)
    response = llm_client.chat(prompt)
    code = extract_python(response)
    return code, {"operator": "O5", "portfolio_size": len(portfolio)}
