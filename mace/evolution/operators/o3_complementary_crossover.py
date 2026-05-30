"""O3 — Complementary Crossover.

Parent selection: sample a pair (h_a, h_b) with probability proportional to
their complementarity C[a, b] = (1/M) * sum_xi |R[xi, a] - R[xi, b]|.
LLM task: synthesize a hybrid in dispatcher style — use h_a-style strategy
on instances where h_a is stronger, h_b-style where h_b is stronger.
Goal: bridge two specialists; cover their combined strengths.
"""
from __future__ import annotations
import numpy as np

from ._common import build_prompt, extract_python, format_heuristic
from ..rank_matrix import complementarity_score


def generate(spec, portfolio, F, R, llm_client):
    N = len(portfolio)
    C = complementarity_score(R).copy()
    np.fill_diagonal(C, 0.0)
    # symmetric: only sample upper triangle to avoid double counting
    iu = np.triu_indices(N, k=1)
    weights = C[iu]
    if weights.sum() <= 0:
        # degenerate (e.g., all heuristics identical) — fall back to uniform
        weights = np.ones_like(weights)
    weights = weights / weights.sum()
    idx_flat = int(np.random.choice(len(weights), p=weights))
    a, b = int(iu[0][idx_flat]), int(iu[1][idx_flat])

    a_wins = int(np.sum(R[:, a] < R[:, b]))
    b_wins = int(np.sum(R[:, b] < R[:, a]))
    ties = R.shape[0] - a_wins - b_wins

    body = (
        f"# Parent A (idx {a}; wins on {a_wins}/{R.shape[0]} instances)\n"
        f"{format_heuristic(portfolio[a])}\n\n"
        f"# Parent B (idx {b}; wins on {b_wins}/{R.shape[0]} instances; "
        f"{ties} ties)\n"
        f"{format_heuristic(portfolio[b])}\n\n"
        f"# Task — Complementary Crossover (O3)\n"
        f"These two heuristics disagree on which instances they handle best — they\n"
        f"are complementary. Synthesize a NEW heuristic in 'dispatcher style':\n\n"
        f"  1. At the start of `solve`, inspect cheap features of `instance`\n"
        f"     (e.g., size, density, distance distribution, structural cues).\n"
        f"  2. Decide which regime the instance falls into.\n"
        f"  3. Apply an A-style strategy when the instance looks like A's strong\n"
        f"     ground; B-style otherwise.\n\n"
        f"Avoid trivial concatenation (do not just `if random < 0.5: A else: B`)\n"
        f"and avoid naive averaging. The dispatch criterion should reflect a real\n"
        f"hypothesis about why A or B is the right tool for that instance."
    )

    prompt = build_prompt(spec, body)
    response = llm_client.chat(prompt)
    code = extract_python(response)
    return code, {
        "operator": "O3",
        "parents": (a, b),
        "complementarity": float(C[a, b]),
        "a_wins": a_wins,
        "b_wins": b_wins,
    }
