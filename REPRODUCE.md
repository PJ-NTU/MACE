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

Given a problem's natural-language description, its **known input data contract**
(`load_data` — for a CO problem the input format is already known, e.g. provided
by the benchmark or by the problem designer), and a few instance files,
`mace/contract/` automatically constructs the rest of the I-O-T contract: the
output schema, the tool library T (`is_feasible` + `objective`, plus a few domain
helpers — no `eval_func`), validated end to end by real LLM-written heuristics.

The pipeline:

- **I — input contract (known, adopted).** The `load_data` is supplied (via
  `--load-data`, typically the problem's existing `config.py`); it is adopted
  verbatim after a machine check that it actually parses the instances. (Only when
  no `load_data` is supplied does an LLM generate one + an independent reviewer
  audit it.) A sample instance is then parsed and its **real structure** (actual
  keys, types, shapes) is introspected and fed to every later stage, so they use
  the true instance keys instead of guessing.
- **O — output schema + trivial `make_solution`.** Generated; an independent
  reviewer LLM judges ONLY whether the solution *schema* is structurally valid
  (the placeholder solver's strategy is not reviewed — its feasibility is checked
  later by T).
- **T core — `is_feasible(instance, solution)` + `objective(instance, solution)`**
  (no `eval_func`). MANDATORY: validated the way MACE actually uses a contract — an
  LLM writes a real heuristic that runs through I → O → T; it must produce a
  feasible, scored solution. Repaired up to the budget; if it cannot pass, the
  contract is rejected.
- **Helpers — optional, two-phase.** Phase 1 plans a small non-overlapping set
  (names + purposes, one LLM call). Phase 2 implements and validates EACH helper
  individually, with a heuristic required to actually call it (instrumented to
  confirm invocation). A helper that cannot be fixed within the budget is
  DISCARDED — helpers never fail the pipeline.
- **Final gate.** One more real heuristic runs through the fully assembled contract
  before it is written out.

```bash
export OPENROUTER_API_KEY=sk-or-...
python scripts/generate_contract.py \
    --slug my_problem \
    --description path/to/description.txt \
    --instances problems/my_problem/instances \
    --load-data problems/my_problem/config.py \
    --model google/gemini-2.5-flash \
    --example aircraft_landing
```

Scope of the automation: the input data contract is taken as known (it always is
for a real CO problem); MACE then **automatically constructs the output schema,
the tool library T, and the heuristic portfolio H** on top of it. Validating that
a heuristic runs through the contract proves it is usable and self-consistent; the
semantic correctness of `is_feasible`/`objective` for a brand-new problem (no
external ground truth) ultimately rests on the description and human review.

The unit tests run fully offline (`python -m pytest tests/contract/ -v`); the
faithfulness integration test (`tests/contract/test_faithfulness_integration.py`)
runs only when `OPENROUTER_API_KEY` is set.
