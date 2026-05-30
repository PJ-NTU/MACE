# Architecture

MACE decomposes automated heuristic design into a two-stage pipeline. This note
maps the paper's concepts to the code.

## The I-O-T-H interface (Stage One)

A problem is a `ProblemSpec` (`mace/framework.py`). Each `problems/<slug>/spec.py`
defines one, exposed as the module-level `SPEC`:

- **I — Input.** `spec.load_data(path) -> dict`. Parses an instance file into an
  instance dict. Backed by the problem's `config.py` loader.
- **O — Output.** The solution shape, documented in `spec.starter_code` and
  enforced by the feasibility tool.
- **T — Tools.** `spec.tools(instance) -> dict[str, Callable]`. Always provides
  `is_feasible(solution) -> (bool, msg)` and `objective(solution) -> float`
  (lower is better; maximization problems are inverted internally). Extra
  domain helpers come from the problem's `extras.py`.
- **H — Heuristic.** A free-form `solve(instance, tools, time_limit_s) -> dict`.
  The shipped, evolved portfolio lives in `problems/<slug>/portfolio/`.

`run_solve` (`mace/framework.py`) executes one `(spec, instance, solve)` triple
and scores it.

## Complementary evolution (Stage Two)

`mace/evolution/evolve.py` drives the loop. Each iteration generates `N`
candidates, pools them with the current portfolio (`2N` total), and selects `N`
back by a ranking-based MILP that minimizes the mean rank of the best-covering
heuristic per instance.

- **Generation operators** (`mace/evolution/operators/`):
  `o1_weighted_mutation` (LR), `o2_reflective_redesign` (RR),
  `o3_complementary_crossover` (CC), `o4_comparative_synthesis` (CS),
  `o5_diversity_injection` (DI).
- **Reactive repair operators:** `o6_error_repair` (ER, on execution failure),
  `o7_efficiency_repair` (EI, on timeout).
- **Selection:** `milp_selection.py` + `rank_matrix.py`.
- **Evaluation:** `parallel_eval.py` (and optional `remote_eval.py`); the
  per-instance objective matrix feeds `cpi.py` (complementary performance index).
- **LLM backbone:** `llm_client.py` (OpenRouter, OpenAI-compatible). The API key
  is read from the `OPENROUTER_API_KEY` environment variable.

Running Stage Two under different `--T-max` budgets yields portfolios spanning
the quality–time trade-off.
