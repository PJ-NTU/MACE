"""O2 — Reflective Redesign.

Parent selection: sample one h weighted by r̄_h (favor weak heuristics).
LLM task: analyze structural weaknesses and substantially rebuild.
Goal: transformative exploration — weak heuristics can unlock new directions
when restructured.
"""
from __future__ import annotations
import numpy as np

from ._common import build_prompt, extract_python, format_heuristic
from ..rank_matrix import mean_rank_per_heuristic


def generate(spec, portfolio, F, R, llm_client):
    rbar = mean_rank_per_heuristic(R)
    w = rbar / rbar.sum()
    parent_idx = int(np.random.choice(len(portfolio), p=w))
    parent_code = portfolio[parent_idx]

    body = (
        f"# Parent heuristic (idx {parent_idx}, mean rank {rbar[parent_idx]:.2f} "
        f"out of {len(portfolio)} — relatively weak in the current portfolio)\n"
        f"{format_heuristic(parent_code)}\n\n"
        f"# Task — Reflective Redesign (O2)\n"
        f"This heuristic underperforms the rest of the portfolio. First, briefly\n"
        f"diagnose its structural weakness (e.g., greedy construction with no\n"
        f"improvement step; improvement step too local; misses obvious dominance\n"
        f"pruning; wastes time on irrelevant operations). Then substantially\n"
        f"REDESIGN it: keep only the parts that are sound, replace the rest with\n"
        f"a different algorithmic skeleton. Calibrate the aggressiveness of the\n"
        f"rebuild to how weak the parent is — the weaker, the bolder."
    )

    prompt = build_prompt(spec, body)
    response = llm_client.chat(prompt)
    code = extract_python(response)
    return code, {"operator": "O2", "parent": parent_idx, "parent_rbar": float(rbar[parent_idx])}
