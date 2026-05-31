# Reproducing MACE — step by step

This guide reproduces the paper's results. There are four levels, from
zero-effort to a full from-scratch run. **The first three need no API key**;
levels 1–2 need no installation at all.

> **Already verified.** We ran all 40 problems end-to-end from a clean checkout
> (clean server, Python 3.10, `google/gemini-3-flash-preview` backbone):
> **40/40 problems pass**, the full pipeline (Stage One → Stage Two → MILP
> selection → held-out test) completes for every problem, with **659 LLM calls
> and 0 failed calls**, and 39/40 problems reach a test feasibility rate of 1.0.

---

## Level 0 — Inspect everything in your browser (no install, no run)

Open the interactive Supplementary Information page:

**https://pj-ntu.github.io/MACE/mace_supplementary_information.html**

It exposes, for all 40 problems: the problem specifications, the seven operator
prompt templates with rendered examples, the 400 final heuristic programs
(10 per problem), and complete per-instance results. Nothing to install or run.

---

## Level 1 — One-click online reproduction (no install, no API key) — recommended

Open the reproduction notebook in Google Colab and choose **Runtime ▸ Run all**:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PJ-NTU/MACE/blob/main/notebooks/reproduce.ipynb)

In about 5 minutes, entirely in your browser, it clones the repository, installs
the dependencies, and runs Levels 2's two no-key checks below. No local setup
and no API key are required.

---

## Level 2 — Local reproduction (no API key)

Prerequisite: Python ≥ 3.10.

```bash
git clone https://github.com/PJ-NTU/MACE.git
cd MACE
pip install -r requirements.txt
```

`requirements.txt` pulls only `numpy`, `mip`, `scipy`, `networkx`, `openai`.
The MILP selection uses `python-mip`'s **bundled CBC** solver, so **no
commercial solver (e.g. Gurobi) is required**.

### 2a. Verify the repository contains exactly the paper's 7109 instances

```bash
python scripts/verify_instances.py
```

Expected (last lines):

```
problems fully verified : 40/40
total instances verified: 7109  (expected 7109)

OK: repository supports exactly 7109 instances.
```

### 2b. Reproduce a shipped portfolio's scores (no LLM call)

This runs the paper's 10 evolved heuristics for a problem on its instances and
reports the best-per-instance (complementary-portfolio) score:

```bash
python scripts/evaluate_portfolio.py --problem aircraft_landing
```

Expected (abridged):

```
=== aircraft_landing: 10 heuristics x 3 instances (T_max=5.0s) ===

[airland1.txt]
   heuristic_01: 1210
   heuristic_03: 700
   ...
   -> portfolio best: 700
...
portfolio feasible on 3/3 instances
```

Substitute any of the 40 folder names under `problems/` for `--problem`.

---

## Level 3 — Full pipeline from scratch (needs an OpenRouter API key)

This generates a fresh portfolio with Stage One + Stage Two. Get a key at
https://openrouter.ai/keys (a small demo run costs only a few cents).

```bash
# Linux / macOS
export OPENROUTER_API_KEY=sk-or-...
# Windows PowerShell
# $env:OPENROUTER_API_KEY = "sk-or-..."

# small demo (~minutes): a 3-heuristic portfolio, 1 evolution iteration
python scripts/run_mace.py --problem aircraft_landing --N 3 --I-iter 1 --T-max 6 --n-train 3 --n-test 2

# paper-scale settings:
# python scripts/run_mace.py --problem aircraft_landing --N 10 --I-iter 8 --T-max 10
```

Expected (last lines): Stage One builds an initial portfolio, Stage Two evolves
it (operators accepted, MILP selection), and a held-out test evaluation prints,
e.g.:

```
final CPI=... -> runs/aircraft_landing/run/final_portfolio.json
test CPI=...  feasible_rate=1.000
llm stats: {'model': 'google/gemini-3-flash-preview', 'n_calls': ..., 'n_failed_calls': 0, ...}
```

Output is written to `runs/<problem>/<run-tag>/`. Useful flags: `--N` portfolio
size, `--I-iter` evolution iterations, `--T-max` per-instance runtime budget (s),
`--model` backbone (default `google/gemini-3-flash-preview`).

---

## Notes

- The four novel port-logistics problems also include a `build_instances.py`
  instance generator.
- Some classical problems pack several instances per file; an instance is
  addressed as `path::idx`. `instances.json` is the authoritative list of the
  7109 instances. See [docs/DATA.md](docs/DATA.md).
- Heuristics in each `problems/<slug>/portfolio/` are preserved exactly as MACE
  evolved them; an individual heuristic may error on some inputs — the
  complementary portfolio is designed so at least one heuristic covers each
  instance, and `run_solve` scores a raising heuristic as infeasible rather than
  crashing.

---

## Regenerating a problem contract (Stage Zero)

MACE's I-O-T problem contract for a NEW problem can be generated automatically
from a natural-language description and raw instance files using `mace/contract/`,
closing the loop on the paper's "the problem contract is constructed automatically
by an LLM agent" claim. A new problem has no pre-existing evaluator, so the tool
library T is authored directly as separate functions — `is_feasible` and
`objective` (plus a few domain helpers) — with no `eval_func`. The pipeline:

- **I (input schema + `load_data`)** and **O (solution schema + a trivial
  `make_solution`)** are each generated, then audited by an independent reviewer
  LLM for correctness (machine checks run too: `load_data` must parse every instance).
- **T core (`is_feasible` + `objective`)** is validated the way MACE actually
  uses a contract: an LLM writes a real heuristic that runs through I → O → T; if a
  solver can produce a feasible, scored solution, the core works.
- **Helpers** (a few domain tools) are validated by a heuristic that calls them.
- A final heuristic gate runs the whole assembled contract before it is written out.

```bash
export OPENROUTER_API_KEY=sk-or-...
python scripts/generate_contract.py \
    --slug my_problem \
    --description path/to/description.txt \
    --instances problems/my_problem/instances \
    --model google/gemini-2.5-flash \
    --example aircraft_landing
```

Note: validating that a heuristic runs through the contract proves it is usable
and self-consistent; the semantic correctness of `is_feasible`/`objective` for a
brand-new problem (no external ground truth) ultimately rests on the description
and human review.

The unit tests run fully offline (`python -m pytest tests/contract/ -v`); the
faithfulness integration test (`tests/contract/test_faithfulness_integration.py`)
runs only when `OPENROUTER_API_KEY` is set.
