"""O1 — Weighted Mutation.

Parent selection: sample one h from the portfolio with probability
proportional to 1 / r̄_h  (favor strong heuristics).
LLM task: identify ONE specific decision module in the parent and apply a
local modification; keep the rest of the code intact.
Goal: exploitation — small steps around already-good heuristics.
"""
from __future__ import annotations
import numpy as np

from ._common import build_prompt, extract_python, format_heuristic
from ..rank_matrix import mean_rank_per_heuristic


def generate(spec, portfolio, F, R, llm_client):
    rbar = mean_rank_per_heuristic(R)
    w = 1.0 / np.maximum(rbar, 1e-9)
    w = w / w.sum()
    parent_idx = int(np.random.choice(len(portfolio), p=w))
    parent_code = portfolio[parent_idx]

    body = (
        f"# Parent heuristic (idx {parent_idx}, mean rank {rbar[parent_idx]:.2f} "
        f"out of {len(portfolio)})\n"
        f"{format_heuristic(parent_code)}\n\n"
        f"# Task — Weighted Mutation (O1)\n"
        f"This parent is one of the stronger heuristics in the current portfolio.\n"
        f"Identify ONE specific decision module in its code (e.g., the construction\n"
        f"step, a neighborhood-selection rule, a tie-breaking rule, an acceptance\n"
        f"criterion, a time budget) and apply a LOCAL modification that you believe\n"
        f"will improve performance on its weak instances. Keep everything else\n"
        f"intact. Do NOT rewrite the algorithm; one focused change."
    )

    prompt = build_prompt(spec, body)
    response = llm_client.chat(prompt)
    code = extract_python(response)
    return code, {"operator": "O1", "parent": parent_idx, "parent_rbar": float(rbar[parent_idx])}
