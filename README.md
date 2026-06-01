# MACE — Modular Algorithm Construction and Evolution

Reference implementation of the paper **"Large Language Models Discover
Complementary Heuristics for Combinatorial Optimization."**

> **Just want to see it run?** Open the one-click Colab notebook
> [`notebooks/reproduce.ipynb`](https://colab.research.google.com/github/PJ-NTU/MACE/blob/main/notebooks/reproduce.ipynb)
> — it reproduces the paper's portfolios in your browser with no local install.
> For a guided local walkthrough see [REPRODUCE.md](REPRODUCE.md).

MACE turns a natural-language description of a combinatorial-optimization (CO)
problem into a portfolio of executable heuristics, without per-problem human
engineering. It works in two stages:

- **Stage One — Universal modular decomposition.** An LLM agent builds a
  verified *problem contract* by decomposing the problem into four modules, the
  **I-O-T-H interface**: an **I**nput schema, an **O**utput schema, a **T**ool
  library (feasibility checker, objective evaluator, domain helpers), and an
  initial **H**euristic portfolio.
- **Stage Two — Time-constrained complementary evolution.** The portfolio is
  refined under a strict runtime budget by seven operators (five generators —
  LR, RR, CC, CS, DI — and two reactive repairers — ER, EI), with a
  ranking-based MILP selection that keeps a *complementary* set of specialists
  covering every instance.

The Stage-One I-O-T contract is generated automatically by MACE's contract
generator (`mace/contract/`, run via `scripts/generate_contract.py`): it turns a
problem's natural-language description and instance files into a drop-in contract
— the output schema, the tool library (`is_feasible` / `objective`), and a few
domain helpers — each module validated by smoke testing before heuristic
generation begins. Every shipped problem already includes one, so running the
generator is optional.

This repository ships the framework plus **40 problems** (36 classical
CO-Bench problems + 4 structurally novel port-logistics problems), each with its
contract, instance data, and the evolved 10-heuristic portfolio from the paper.

## Repository layout

```
MACE/
├── mace/                     # Framework core
│   ├── framework.py          #   ProblemSpec / run_solve / load_heuristic
│   ├── contract/             #   Stage One: auto-generate the I-O-T contract from NL
│   └── evolution/            #   Stage Two: 7 operators, MILP selection, LLM client
│       ├── evolve.py, milp_selection.py, rank_matrix.py, ...
│       └── operators/        #   o1..o7
├── problems/                 # 40 self-contained problems
│   └── <slug>/
│       ├── spec.py           #   I + O contract
│       ├── feasibility_steps.py, extras.py, config.py   # T: tools / loaders
│       ├── instances/        #   instance data
│       └── portfolio/        #   evolved heuristics: heuristic_01..10.py (+ portfolio.json)
├── instances.json            # manifest: the exact 7109 instances (per problem)
├── scripts/
│   ├── run_mace.py           # run MACE end-to-end on one problem (needs API key)
│   ├── generate_contract.py  # auto-generate a problem's I-O-T contract (optional; needs API key)
│   ├── evaluate_portfolio.py # score the shipped portfolio (no API key)
│   └── verify_instances.py   # check the repo supports exactly 7109 instances
└── docs/                     # ARCHITECTURE.md, DATA.md
```

## Install

```bash
pip install -r requirements.txt
```

Python ≥ 3.10. The MILP selection uses `python-mip` with its bundled CBC solver —
no commercial solver required (Gurobi is used automatically only if installed).

## Quick start

**1. Evaluate a shipped portfolio (no API key needed).** Runs the paper's
evolved 10 heuristics on a problem's instances and reports the best-per-instance
(complementary) score:

```bash
python scripts/evaluate_portfolio.py --problem aircraft_landing
```

**2. Run MACE from scratch (needs an OpenRouter API key).** Generates a fresh
portfolio via Stage One + Stage Two:

```bash
# Linux / macOS
export OPENROUTER_API_KEY=sk-or-...
# Windows PowerShell
# $env:OPENROUTER_API_KEY = "sk-or-..."

python scripts/run_mace.py --problem aircraft_landing --N 10 --I-iter 8 --T-max 10
```

Output is written to `runs/<problem>/<run-tag>/`. Key flags: `--N` portfolio
size, `--I-iter` evolution iterations, `--T-max` per-instance runtime budget (s),
`--model` backbone (default `google/gemini-3-flash-preview`).

**3. (Optional) Auto-generate a problem's contract (needs an OpenRouter API key).**
The Stage-One I-O-T contract is produced automatically from a natural-language
description plus instance files (output schema + `is_feasible` / `objective` +
domain helpers); every shipped problem already includes one:

```bash
python scripts/generate_contract.py \
    --slug aircraft_landing \
    --description path/to/description.txt \
    --instances problems/aircraft_landing/instances \
    --load-data problems/aircraft_landing/config.py \
    --out runs/aircraft_landing_contract
```

Each generated module is admitted only after a real LLM-written heuristic solves
through it end to end. `--load-data` (a known parser) makes the Input Designer
adopt it verbatim and skip input generation; omit it to generate the input schema
too. Without `--out` the contract is written to `problems/<slug>/`, **overwriting
the shipped one** — point `--out` elsewhere to keep it.

## Problems

`problems/<slug>/` — slug is the folder name. The 40 problems:

| Family | Problems |
|---|---|
| Scheduling | aircraft_landing, common_due_date_scheduling, crew_scheduling, flow_shop_scheduling, hybrid_reentrant_shop_scheduling, job_shop_scheduling, open_shop_scheduling |
| Routing & graphs | euclidean_steiner_problem, graph_colouring, maximal_independent_set, resource_constrained_shortest_path, travelling_salesman_problem, vehicle_routing_period_routing |
| Packing & cutting | bin_packing_one_dimensional, constrained_guillotine_cutting, constrained_non_guillotine_cutting, container_loading, container_loading_with_weight_restrictions, packing_unequal_circles, packing_unequal_circles_area, packing_unequal_rectangles_and_squares, packing_unequal_rectangles_and_squares_area, unconstrained_guillotine_cutting |
| Location & assignment | assignment_problem, capacitated_warehouse_location, corporate_structuring, equitable_partitioning_problem, generalised_assignment_problem, p_median_capacitated, p_median_uncapacitated, set_covering, set_partitioning, uncapacitated_warehouse_location |
| Selection & knapsack | assortment_problem, multidimensional_knapsack_problem, multi_demand_multidimensional_knapsack_problem |
| Novel (port logistics) | port_scheduling_problem, multi_tugboat_routing_and_scheduling_problem, multi_base_tugboat_routing_and_scheduling_problem, multi_tugboat_routing_problem_with_variable_speed |

The four novel problems also include a `build_instances.py` instance generator.

## Data

The repository ships the **exact test instance set used in the paper — 7109
test instances across the 40 problems** — in the original OR-Library file formats
under `problems/<slug>/instances/`. Many classical files are multi-instance
(one file holds several cases, addressed as `file::idx`), so the file count is
smaller than the instance count. The authoritative list is `instances.json`
(per problem: file + case index + split). Verify it:

```bash
python scripts/verify_instances.py     # asserts the repo supports exactly 7109
```

The repository is self-contained — no external download is needed to reproduce
the paper's instances. See [docs/DATA.md](docs/DATA.md) for the data layout.

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — how the code maps to the paper:
  the I-O-T-H interface, `ProblemSpec` / `run_solve`, and where each of the seven
  Stage-Two operators and the MILP selection live.
- [docs/DATA.md](docs/DATA.md) — instance layout, the `instances.json` manifest,
  and the per-problem subfolder conventions (flat files vs. `train/`/`test/` /
  `er_*`), all handled automatically by the scripts.
- **Supplementary information** — interactive page with per-problem results,
  prompt templates, and final portfolios:
  **[pj-ntu.github.io/MACE/mace_supplementary_information.html](https://pj-ntu.github.io/MACE/mace_supplementary_information.html)**
  (source: [docs/mace_supplementary_information.html](docs/mace_supplementary_information.html)).

## Citation

See [CITATION.cff](CITATION.cff).

## License

MIT — see [LICENSE](LICENSE).
