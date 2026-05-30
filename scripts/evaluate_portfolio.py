"""Evaluate a problem's shipped 10-heuristic portfolio. No API key required.

Loads `problems/<slug>/portfolio/heuristic_*.py`, runs each on the problem's
instances, and reports per-heuristic feasibility / objective plus the
portfolio's best-per-instance score (the complementary-portfolio view).

Usage:
    python scripts/evaluate_portfolio.py --problem aircraft_landing
    python scripts/evaluate_portfolio.py --problem set_covering --n-instances 5 --T-max 10
"""
from __future__ import annotations
import argparse
import glob
import importlib
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from mace.framework import run_solve, load_heuristic


def list_instances(slug: str) -> list[Path]:
    base = REPO_ROOT / "problems" / slug / "instances"
    files = [p for p in base.rglob("*") if p.is_file()
             and p.name not in {"config.py"} and not p.name.startswith((".", "__"))]
    files.sort(key=lambda p: p.stat().st_size)   # smallest first
    return files


def main():
    ap = argparse.ArgumentParser(description="Evaluate shipped portfolio (no LLM).")
    ap.add_argument("--problem", required=True, help="problem folder name (slug)")
    ap.add_argument("--n-instances", type=int, default=3, help="how many instances to score")
    ap.add_argument("--T-max", type=float, default=5.0, help="per-run time budget (s)")
    args = ap.parse_args()

    slug = args.problem
    spec = importlib.import_module(f"problems.{slug}.spec").SPEC
    heur_paths = sorted(glob.glob(str(REPO_ROOT / "problems" / slug / "portfolio" / "heuristic_*.py")))
    heurs = [(Path(p).stem, load_heuristic(p)) for p in heur_paths]
    insts = list_instances(slug)[: args.n_instances]
    if not insts:
        print(f"no instances found for {slug}"); sys.exit(1)

    print(f"=== {slug}: {len(heurs)} heuristics x {len(insts)} instances "
          f"(T_max={args.T_max}s) ===\n")
    portfolio_best = []
    for inst_path in insts:
        instance = spec.load_data(str(inst_path))
        print(f"[{inst_path.name}]")
        best = None
        for name, fn in heurs:
            r = run_solve(spec, instance, fn, time_limit_s=args.T_max)
            obj = f"{r.objective:.4g}" if r.feasible else "infeasible"
            print(f"   {name}: {obj}")
            if r.feasible and (best is None or r.objective < best):
                best = r.objective
        portfolio_best.append(best)
        print(f"   -> portfolio best: {best:.4g}\n" if best is not None else "   -> no feasible\n")

    feasible = [b for b in portfolio_best if b is not None]
    print(f"portfolio feasible on {len(feasible)}/{len(insts)} instances")


if __name__ == "__main__":
    main()
