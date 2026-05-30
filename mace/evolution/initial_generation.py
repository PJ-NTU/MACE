"""Build the initial portfolio Π for Stage Two iteration 0.

Strategy: start from a hand-written baseline, then generate (N-1) more
candidates via O5 (random restart). Each candidate must pass smoke test;
failures are repaired via O6 up to I_rep times before being discarded.
"""
from __future__ import annotations
import logging
from typing import Optional

from .smoke_test import smoke_test
# Bootstrap from-scratch generator (was the old O5 Random Restart; now lives
# outside GENERATION_OPERATORS since the new O5 is Diversity Injection).
from .operators import _bootstrap_random as o5_random_restart
from .operators import o6_error_repair

logger = logging.getLogger(__name__)


def build_initial_portfolio(
    spec,
    baseline_code: str,
    llm_client,
    N: int,
    smoke_instance_path: str,
    smoke_time_limit_s: float = 30.0,
    I_rep: int = 3,
    max_attempts: Optional[int] = None,
    spec_module_path: Optional[str] = None,
    use_subprocess: bool = False,
    hard_kill_slack: float = 2.0,
) -> list[str]:
    """Return N solve-code strings. Slot 0 is the baseline."""
    portfolio: list[str] = []

    smoke_kwargs = dict(
        spec_module_path=spec_module_path,
        use_subprocess=use_subprocess,
        hard_kill_slack=hard_kill_slack,
    )

    # baseline first — verify it passes smoke
    passed, err = smoke_test(baseline_code, spec, smoke_instance_path,
                              smoke_time_limit_s, **smoke_kwargs)
    if not passed:
        raise RuntimeError(f"Baseline failed smoke test: {err}")
    portfolio.append(baseline_code)
    logger.info("Initial portfolio: baseline slot 0 OK")

    if max_attempts is None:
        max_attempts = (N - 1) * (I_rep + 1) * 3  # generous cap

    attempts = 0
    while len(portfolio) < N and attempts < max_attempts:
        attempts += 1
        try:
            code, meta = o5_random_restart.generate(spec, portfolio, None, None, llm_client)
        except Exception as e:
            logger.warning("O5 generation raised: %s", e)
            continue

        passed, err = smoke_test(code, spec, smoke_instance_path,
                                  smoke_time_limit_s, **smoke_kwargs)
        repair_round = 0
        while not passed and repair_round < I_rep:
            repair_round += 1
            try:
                code = o6_error_repair.repair(spec, code, err, llm_client)
            except Exception as e:
                logger.warning("O6 repair raised: %s", e)
                break
            passed, err = smoke_test(code, spec, smoke_instance_path,
                                      smoke_time_limit_s, **smoke_kwargs)

        if passed:
            portfolio.append(code)
            logger.info(
                "Initial portfolio: slot %d added (O5%s)",
                len(portfolio) - 1,
                f" + O6×{repair_round}" if repair_round else "",
            )
        else:
            logger.info(
                "Initial portfolio: candidate discarded after %d repair attempts: %s",
                repair_round, err,
            )

    if len(portfolio) < N:
        raise RuntimeError(
            f"Could not build initial portfolio of size {N} after {attempts} attempts; "
            f"only got {len(portfolio)}."
        )
    return portfolio
