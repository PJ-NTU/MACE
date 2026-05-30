"""O7 — Efficiency-Driven Repair (reactive).

Triggered when a candidate exceeds T_max on some training instance. Given
the slow code + instance-size context, REDUCE algorithmic complexity while
preserving the core strategy.

Cap: I_eff = 3 retries; discard if still timing out afterwards.
"""
from __future__ import annotations

from ._common import build_prompt, extract_python, format_heuristic


def repair(
    spec,
    slow_code: str,
    instance_info: str,
    elapsed_s: float,
    time_limit_s: float,
    llm_client,
) -> str:
    body = (
        f"# Slow heuristic\n{format_heuristic(slow_code)}\n\n"
        f"# Timeout report\n"
        f"Instance: {instance_info}\n"
        f"Elapsed: {elapsed_s:.1f}s   (time limit: {time_limit_s:.1f}s)\n\n"
        f"# Task — Efficiency-Driven Repair (O7)\n"
        f"The heuristic above ran too long on the instance shown. Reduce its\n"
        f"algorithmic complexity while PRESERVING the core strategy. Standard\n"
        f"levers, in rough order of impact:\n\n"
        f"  - replace exhaustive neighborhood search with k-nearest-neighbor pruning\n"
        f"  - early-exit the improvement loop when a pass yields no gain\n"
        f"  - vectorize hot loops with numpy where possible\n"
        f"  - cap the number of improvement iterations or restarts\n"
        f"  - drop the most expensive optional pass when little time remains\n"
        f"  - add an explicit `time.time() - t0 > time_limit_s - margin` guard\n"
        f"    inside any deeply nested loop\n\n"
        f"Keep the construction / improvement skeleton recognizable. Output the\n"
        f"complete revised source."
    )
    prompt = build_prompt(spec, body)
    response = llm_client.chat(prompt)
    return extract_python(response)
