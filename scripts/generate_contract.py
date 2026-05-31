#!/usr/bin/env python
"""Generate a drop-in ISTH contract for a new CO problem.

Example:
  export OPENROUTER_API_KEY=sk-or-...
  python scripts/generate_contract.py \
      --slug my_problem \
      --description path/to/description.txt \
      --instances problems/my_problem/instances \
      --keys cap items \
      --model google/gemini-3.1-flash-lite \
      --example aircraft_landing
"""
from __future__ import annotations
import argparse
import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mace.contract import generate_contract
from mace.evolution.llm_client import OpenRouterClient


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True)
    ap.add_argument("--description", required=True, help="path to NL description .txt")
    ap.add_argument("--instances", required=True, help="dir of raw instance files")
    ap.add_argument("--out", default=None, help="output dir (default problems/<slug>)")
    ap.add_argument("--keys", nargs="*", default=[], help="expected top-level instance keys")
    ap.add_argument("--model", default="google/gemini-3.1-flash-lite")
    ap.add_argument("--example", default="aircraft_landing")
    ap.add_argument("--direction", choices=["min", "max"], default="min")
    ap.add_argument("--i-rep", type=int, default=3)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    nl = Path(args.description).read_text(encoding="utf-8")
    out = args.out or str(REPO_ROOT / "problems" / args.slug)

    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        sys.exit("OPENROUTER_API_KEY not set")
    llm = OpenRouterClient(api_key=key, model=args.model)

    path = generate_contract(
        slug=args.slug, nl_description=nl, instances_dir=args.instances,
        out_dir=out, llm_client=llm, required_keys=args.keys,
        example_slug=args.example, direction=args.direction, i_rep=args.i_rep,
    )
    print(f"Contract written to {path}")
    print(f"LLM stats: {llm.stats()}")


if __name__ == "__main__":
    main()
