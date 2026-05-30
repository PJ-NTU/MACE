"""Verify that the repository physically supports exactly the 7109 instances
listed in instances.json.

For each problem, every manifest entry (file, case_idx) must resolve to a file
under problems/<slug>/instances/ and be loadable by that problem's spec. Prints
a per-problem count and asserts the grand total.

Usage:
    python scripts/verify_instances.py
"""
from __future__ import annotations
import importlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

EXPECTED_TOTAL = 7109


def main():
    manifest = json.loads((REPO_ROOT / "instances.json").read_text(encoding="utf-8"))
    grand = 0
    problems_ok = 0
    failures = []

    for slug in sorted(manifest):
        entries = manifest[slug]
        spec_mod = importlib.import_module(f"problems.{slug}.spec")
        # raw loader returns the list of all cases in a file (spec.load_data
        # returns a single case); use the raw loader to count cases.
        raw_load = spec_mod._cfg.load_data
        inst_dir = REPO_ROOT / "problems" / slug / "instances"

        # load each referenced file once, record its case count
        ncases: dict[str, int] = {}
        ok = 0
        for e in entries:
            rel = e["file"]
            if rel not in ncases:
                p = inst_dir / rel
                if not p.exists():
                    failures.append(f"{slug}: missing file {rel}")
                    ncases[rel] = -1
                else:
                    try:
                        r = raw_load(str(p))
                        ncases[rel] = len(r) if isinstance(r, list) else 1
                    except Exception as ex:
                        failures.append(f"{slug}: load fail {rel}: {type(ex).__name__}: {ex}")
                        ncases[rel] = -1
            n = ncases[rel]
            if n >= 0 and e.get("case_idx", 0) < n:
                ok += 1
            else:
                failures.append(f"{slug}: bad case_idx {e.get('case_idx')} for {rel} (ncases={n})")
        grand += ok
        if ok == len(entries):
            problems_ok += 1
        print(f"{slug:48s} {ok:>5d} / {len(entries):<5d}")

    print("=" * 64)
    print(f"problems fully verified : {problems_ok}/{len(manifest)}")
    print(f"total instances verified: {grand}  (expected {EXPECTED_TOTAL})")
    if failures:
        print(f"\n{len(failures)} FAILURES:")
        for f in failures[:30]:
            print("  ", f)
    if grand == EXPECTED_TOTAL and not failures:
        print("\nOK: repository supports exactly 7109 instances.")
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
