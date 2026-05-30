"""O4 — Comparative Synthesis.

Parent selection: independently sample two heuristics (h_a, h_b) from the
portfolio:
  - h_a weighted by 1 / r̄_h  (prefer strong: smaller mean rank)
  - h_b weighted by r̄_h  from portfolio \\ {h_a}  (prefer weak: larger mean rank)
where r̄_h is the within-portfolio mean rank of h.

LLM task: contrast the strong h_a against the weak h_b -- diagnose why h_a
outperforms, why h_b underperforms, on which instances each wins -- and
then design a new heuristic informed by this comparison. The new
heuristic is NOT required to be a literal hybrid; it can be a fresh design,
as long as the diagnostic comparison drives the decisions.

Goal: occupy the "strong-vs-weak contrast and reflection" action space,
distinct from O3 (complementarity-driven crossover with no quality bias).
"""
from __future__ import annotations
import numpy as np

from ._common import build_prompt, extract_python, format_heuristic
from ..rank_matrix import mean_rank_per_heuristic


def generate(spec, portfolio, F, R, llm_client):
    N = len(portfolio)
    if N < 2:
        raise RuntimeError(
            "Comparative Synthesis (O4) requires portfolio size >= 2; "
            f"got {N}."
        )

    rbar = mean_rank_per_heuristic(R)
    eps = 1e-9

    # h_a: prefer strong (small mean rank) -> weight = 1 / rbar
    w_a = 1.0 / np.maximum(rbar, eps)
    w_a = w_a / w_a.sum()
    sampled_a = int(np.random.choice(N, p=w_a))

    # h_b: prefer weak (large mean rank), excluding sampled_a
    mask = np.ones(N, dtype=bool)
    mask[sampled_a] = False
    idx_rest = np.where(mask)[0]
    rbar_rest = rbar[idx_rest]
    w_b = rbar_rest / rbar_rest.sum() if rbar_rest.sum() > 0 else None
    if w_b is None:
        sampled_b = int(np.random.choice(idx_rest))
    else:
        sampled_b = int(idx_rest[np.random.choice(len(idx_rest), p=w_b)])

    # Sampling is preferential, not deterministic; small portfolios can flip the
    # intended strong/weak ordering. Re-label by actual rbar so the LLM-facing
    # text is always honest.
    if rbar[sampled_a] <= rbar[sampled_b]:
        a, b = sampled_a, sampled_b
    else:
        a, b = sampled_b, sampled_a

    M_inst = R.shape[0]
    a_wins = int(np.sum(R[:, a] < R[:, b]))
    b_wins = int(np.sum(R[:, b] < R[:, a]))
    ties = M_inst - a_wins - b_wins

    body = (
        f"# Stronger parent h_a  (idx {a}, mean rank {rbar[a]:.2f} of {N})\n"
        f"{format_heuristic(portfolio[a])}\n\n"
        f"# Weaker parent h_b  (idx {b}, mean rank {rbar[b]:.2f} of {N})\n"
        f"{format_heuristic(portfolio[b])}\n\n"
        f"# Head-to-head on training set ({M_inst} instances):\n"
        f"  h_a beats h_b on  {a_wins}/{M_inst}\n"
        f"  h_b beats h_a on  {b_wins}/{M_inst}\n"
        f"  ties / both-fail  {ties}/{M_inst}\n\n"
        f"# Task -- Comparative Synthesis (O4)\n"
        f"Compare h_a (stronger overall) against h_b (weaker overall).\n"
        f"  1. What specific algorithmic choices does h_a make that h_b lacks?\n"
        f"     (construction order, neighborhood, acceptance rule, time budget, etc.)\n"
        f"  2. On the instances where h_b wins against h_a, what does h_b do that\n"
        f"     might be the cause -- and is that worth keeping in the new design?\n"
        f"  3. What systemic weakness in h_b explains its overall underperformance?\n\n"
        f"Based on this contrast, design a NEW heuristic. You are NOT required to\n"
        f"hybridize their code -- the new heuristic can be a completely fresh\n"
        f"approach, as long as it is informed by the diagnostic comparison above."
    )

    prompt = build_prompt(spec, body)
    response = llm_client.chat(prompt)
    code = extract_python(response)
    return code, {
        "operator": "O4",
        "parents": (a, b),
        "strong_idx": a,
        "weak_idx": b,
        "rbar_strong": float(rbar[a]),
        "rbar_weak": float(rbar[b]),
        "a_wins": a_wins,
        "b_wins": b_wins,
    }
