"""Run MACE end-to-end on a single problem (Stage One + Stage Two).

This generates a *fresh* heuristic portfolio with the LLM. It needs an
OpenRouter API key in the environment:

    export OPENROUTER_API_KEY=sk-or-...        # Linux / macOS
    $env:OPENROUTER_API_KEY = "sk-or-..."      # Windows PowerShell

Usage:
    python scripts/run_mace.py --problem aircraft_landing
    python scripts/run_mace.py --problem aircraft_landing --N 10 --I-iter 8 --T-max 10

`--problem` is the folder name under `problems/` (its slug). Output is
written to `runs/<problem>/<run-tag>/`.
"""
from __future__ import annotations
import argparse
import importlib
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from mace.evolution.llm_client import OpenRouterClient
from mace.evolution.initial_generation import build_initial_portfolio
from mace.evolution.evolve import evolve, evaluate_portfolio, evaluate_on_test
from mace.evolution.cpi import PENALTY
from mace.evolution.smoke_test import smoke_test
from mace.evolution.operators import _bootstrap_random as o5_random_restart
from mace.evolution.operators import o6_error_repair

SKIP_FILENAMES = {"config.py", ".DS_Store", "Thumbs.db", "desktop.ini"}
DEFAULT_MODEL = "google/gemini-3-flash-preview"


def collect_instances(problem_slug: str, spec, max_total: int = 12) -> list[str]:
    """Build single-case 'paths' (file or file::idx) from problems/<slug>/instances.

    Recurses one level into subdirs; counts cases per file via the spec's raw
    config loader so problems with few files but many cases stay usable. Smaller
    files come first so the train pool starts with the easiest instances.
    """
    data_dir = REPO_ROOT / "problems" / problem_slug / "instances"
    DIR_PRIORITY = {"er_valid": 0, "er_test": 1, "er_large_test": 2, "train": 0, "test": 1}

    def _skip(p: Path) -> bool:
        return p.name in SKIP_FILENAMES or p.name.startswith((".", "__"))

    triples: list[tuple[int, Path]] = []
    for p in sorted(data_dir.iterdir()):
        if _skip(p):
            continue
        if p.is_file():
            triples.append((0, p))
        elif p.is_dir():
            pri = DIR_PRIORITY.get(p.name, 5)
            for sub in sorted(p.iterdir()):
                if _skip(sub) or not sub.is_file():
                    continue
                triples.append((pri, sub))

    def _fsize(p):
        try:
            return p.stat().st_size
        except Exception:
            return 0
    triples.sort(key=lambda t: (t[0], _fsize(t[1]), t[1].name))
    files = [t[1] for t in triples]

    raw_load = None
    spec_mod = sys.modules.get(spec.__class__.__module__)
    if spec_mod is not None and hasattr(spec_mod, "_cfg"):
        raw_load = spec_mod._cfg.load_data

    out: list[str] = []
    for f in files:
        if len(out) >= max_total:
            break
        n_cases = 1
        if raw_load is not None:
            try:
                cases = raw_load(str(f))
                if isinstance(cases, list):
                    n_cases = max(1, len(cases))
            except Exception:
                continue
        for idx in range(n_cases):
            if len(out) >= max_total:
                break
            out.append(f"{f}::{idx}" if idx > 0 else str(f))
    return out


def _build_all_o5_portfolio(spec, llm, N, smoke_path, smoke_tl, I_rep, log, spec_module_path):
    """Fallback when the trivial baseline fails smoke: build N from-scratch heuristics."""
    smoke_kwargs = dict(spec_module_path=spec_module_path, use_subprocess=True, hard_kill_slack=2.0)
    portfolio: list[str] = []
    attempts, max_attempts = 0, N * (I_rep + 1) * 3
    while len(portfolio) < N and attempts < max_attempts:
        attempts += 1
        try:
            code, _ = o5_random_restart.generate(spec, portfolio, None, None, llm)
        except Exception as e:
            log.warning("O5.generate raised: %s", e)
            continue
        passed, err = smoke_test(code, spec, smoke_path, smoke_tl, **smoke_kwargs)
        repairs = 0
        while not passed and repairs < I_rep:
            repairs += 1
            try:
                code = o6_error_repair.repair(spec, code, err, llm)
            except Exception as e:
                log.warning("O6.repair raised: %s", e)
                break
            passed, err = smoke_test(code, spec, smoke_path, smoke_tl, **smoke_kwargs)
        if passed:
            portfolio.append(code)
            log.info("  generated initial heuristic %d/%d", len(portfolio), N)
    return portfolio


def main():
    ap = argparse.ArgumentParser(description="Run MACE on one problem.")
    ap.add_argument("--problem", required=True, help="problem folder name under problems/ (slug)")
    ap.add_argument("--N", type=int, default=10, help="portfolio size")
    ap.add_argument("--I-iter", type=int, default=8, help="Stage Two evolution iterations")
    ap.add_argument("--T-max", type=float, default=10.0, help="per-instance runtime budget (s)")
    ap.add_argument("--I-rep", type=int, default=3)
    ap.add_argument("--I-eff", type=int, default=3)
    ap.add_argument("--n-train", type=int, default=6)
    ap.add_argument("--n-test", type=int, default=4)
    ap.add_argument("--api-key", default=os.environ.get("OPENROUTER_API_KEY", ""))
    ap.add_argument("--model", default=os.environ.get("MACE_MODEL", DEFAULT_MODEL))
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--max-calls", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke-time-limit", type=float, default=10.0)
    ap.add_argument("--milp-time-limit", type=float, default=20.0)
    ap.add_argument("--n-workers", type=int, default=1)
    ap.add_argument("--hard-kill-slack", type=float, default=2.0)
    ap.add_argument("--run-tag", default="run")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
                        stream=sys.stdout)
    log = logging.getLogger("run_mace")
    # keep the console focused: silence HTTP traffic and per-iteration internals
    for _noisy in ("httpx", "openai", "mace.evolution"):
        logging.getLogger(_noisy).setLevel(logging.WARNING)

    if not args.api_key:
        log.error("No API key. Set OPENROUTER_API_KEY or pass --api-key. "
                  "(To only evaluate the shipped portfolio, use scripts/evaluate_portfolio.py.)")
        sys.exit(2)

    slug = args.problem
    spec_module_path = f"problems.{slug}.spec"
    out_dir = REPO_ROOT / "runs" / slug / args.run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("=" * 70)
    log.info("MACE run: %s  (model=%s)", slug, args.model)
    log.info("=" * 70)

    spec_mod = importlib.import_module(spec_module_path)
    spec = spec_mod.SPEC

    instances = collect_instances(slug, spec, max_total=args.n_train + args.n_test + 2)
    if len(instances) < args.n_train + args.n_test:
        log.error("only %d usable instances, need %d", len(instances), args.n_train + args.n_test)
        sys.exit(1)
    train_paths = instances[: args.n_train]
    test_paths = instances[args.n_train: args.n_train + args.n_test]
    log.info("%d development + %d test instances", len(train_paths), len(test_paths))

    llm = OpenRouterClient(api_key=args.api_key, model=args.model,
                           temperature=args.temperature, max_tokens=args.max_tokens,
                           max_calls=args.max_calls)

    # ---- Stage One: initial portfolio ----
    log.info("Stage One: building initial portfolio (%d heuristics) ...", args.N)
    try:
        portfolio = build_initial_portfolio(
            spec=spec, baseline_code=spec.starter_code, llm_client=llm,
            N=args.N, smoke_instance_path=train_paths[0],
            smoke_time_limit_s=args.smoke_time_limit, I_rep=args.I_rep,
            spec_module_path=spec_module_path, use_subprocess=True, hard_kill_slack=2.0,
        )
    except RuntimeError:
        # The starter template is an intentional placeholder, so it does not pass
        # the smoke test; build the initial heuristics from scratch instead.
        portfolio = _build_all_o5_portfolio(spec, llm, args.N, train_paths[0],
                                            args.smoke_time_limit, args.I_rep, log, spec_module_path)
        if len(portfolio) < args.N:
            log.error("could not build a full initial portfolio (%d/%d)", len(portfolio), args.N)
            sys.exit(1)
    (out_dir / "initial_portfolio.json").write_text(
        json.dumps(portfolio, indent=2, ensure_ascii=False), encoding="utf-8")

    _use_subproc = (args.n_workers == 1)
    F0, _ = evaluate_portfolio(spec, portfolio, train_paths, T_max=args.T_max,
                               spec_module_path=spec_module_path, use_subprocess=_use_subproc,
                               n_workers=args.n_workers, hard_kill_slack=args.hard_kill_slack)
    log.info("Stage One complete: %d heuristics.", len(portfolio))

    # ---- Stage Two: complementary evolution ----
    log.info("Stage Two: evolving portfolio (%d iterations) ...", args.I_iter)
    portfolio_final, F_final, history = evolve(
        spec=spec, portfolio=portfolio, F=F0, training_instances=train_paths,
        N=args.N, I_iter=args.I_iter, T_max=args.T_max, llm_client=llm,
        I_rep=args.I_rep, I_eff=args.I_eff, output_dir=str(out_dir / "iters"),
        rng_seed=args.seed, milp_time_limit_s=args.milp_time_limit,
        spec_module_path=spec_module_path, n_workers=args.n_workers,
        use_subprocess=_use_subproc, hard_kill_slack=args.hard_kill_slack)
    (out_dir / "final_portfolio.json").write_text(
        json.dumps(portfolio_final, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Stage Two complete: final portfolio saved to %s", out_dir / "final_portfolio.json")

    # ---- Test ----
    log.info("Evaluating final portfolio on held-out test instances ...")
    test_results = evaluate_on_test(spec, portfolio_final, test_paths, T_max=args.T_max,
                                    spec_module_path=spec_module_path, use_subprocess=_use_subproc,
                                    n_workers=args.n_workers, hard_kill_slack=args.hard_kill_slack)
    test_F = np.array(test_results["F_test"])
    feasible_pct = 100.0 * float((test_F < PENALTY).any(axis=1).mean())
    stats = llm.stats()
    log.info("Result: portfolio is feasible on %.1f%% of test instances.", feasible_pct)
    log.info("LLM usage: %d calls, %d failed.", stats.get("n_calls", 0), stats.get("n_failed_calls", 0))


if __name__ == "__main__":
    main()
