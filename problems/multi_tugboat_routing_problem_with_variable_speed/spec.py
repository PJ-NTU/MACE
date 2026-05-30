"""ISTH spec for CO-Bench task: 'Multi-Tugboat Routing Problem with Variable Speed' (auto-generated)."""
from __future__ import annotations
import inspect
import math
import sys
import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from mace.framework import ProblemSpec  # noqa: E402


_CONFIG_PATH = Path(__file__).resolve().parent / "config.py"
_mod_spec = importlib.util.spec_from_file_location("_cobench_cfg_multi_tugboat_routing_problem_with_variable_speed", _CONFIG_PATH)
_cfg = importlib.util.module_from_spec(_mod_spec)
_mod_spec.loader.exec_module(_cfg)


_DIRECTION = 'min'   # 'min' or 'max' (see tools/build_cobench_specs.py)


def _filtered_eval_call(merged: dict) -> float:
    # Stringify non-string top-level keys so `**merged` works for problems
    # whose solution dict is naturally int-keyed (graph_colouring).
    # CO-Bench eval_funcs that need int keys typically do
    # int(k) when interpreting kwargs, so stringification is harmless.
    merged = {(k if isinstance(k, str) else str(k)): v for k, v in merged.items()}
    sig = inspect.signature(_cfg.eval_func)
    accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD
                         for p in sig.parameters.values())
    if accepts_kwargs:
        return _cfg.eval_func(**merged)
    filtered = {k: v for k, v in merged.items() if k in sig.parameters}
    return _cfg.eval_func(**filtered)


# eval_func source code shown verbatim in the LLM prompt's feasibility section
# so the model can see every raise / return False branch and avoid them.
try:
    _EVAL_FUNC_SRC = inspect.getsource(_cfg.eval_func)
except Exception:
    _EVAL_FUNC_SRC = "<eval_func source unavailable>"

# Optional step-by-step is_feasible reference (LLM-rewritten from eval_func).
# If feasibility_steps.py exists next to this spec, surface its narrative
# check function first so the LLM can read the constraints quickly without
# parsing the full eval_func.
try:
    from .feasibility_steps import FEASIBILITY_STEPS_PY as _FEASIBILITY_STEPS_PY
except ImportError:
    _FEASIBILITY_STEPS_PY = ""

_FEASIBILITY_DOC = (
    "A solution is feasible iff CO-Bench's `eval_func(**instance, **solution)` "
    "does NOT raise an exception. This is what `tools['is_feasible']` delegates "
    "to internally.\n\n"
    + (
        "Below is a step-by-step view of the constraints (cost computation removed) "
        "in the same style as `is_feasible` for simple problems. The instance keys "
        "shown in the `solve` docstring are in scope:\n\n"
        "```python\n" + _FEASIBILITY_STEPS_PY + "\n```\n\n"
        if _FEASIBILITY_STEPS_PY else ""
    )
    + "Full CO-Bench `eval_func` source (ground truth -- the step-by-step view "
      "above is derived from this, but this is what actually runs):\n\n"
      "```python\n" + _EVAL_FUNC_SRC + "\n```"
)

# Direction-aware objective purpose, so the LLM knows what `tools['objective']`
# returns (always lower-better after the spec's direction wrapping).
_OBJECTIVE_PURPOSE = (
    "LOWER IS BETTER. " +
    ("Cost-like quantity (e.g., makespan, distance, penalty) directly from the "
     "underlying problem." if _DIRECTION == 'min' else
     "Equals 1 / raw_eval_func_value -- the framework reverses direction "
     "internally because the underlying problem is naturally maximization "
     "(e.g., profit, coverage, score). Smaller objective <=> larger raw score.")
)


# Optional per-problem helpers: if a `extras.py` exists next to this spec,
# import its `extra_tools(instance)` factory + `EXTRA_TOOLS_DESCRIPTION` list
# and merge them into the spec's tools() / tools_description. Used when the
# problem has constraint structure that benefits from a dedicated helper
# (e.g., an ILP solver tool for Set Partitioning).
try:
    from .extras import extra_tools as _EXTRA_TOOLS_FACTORY  # type: ignore
    from .extras import EXTRA_TOOLS_DESCRIPTION as _EXTRA_TOOLS_DESC  # type: ignore
except ImportError:
    _EXTRA_TOOLS_FACTORY = None
    _EXTRA_TOOLS_DESC = []


class _CoBenchSpec(ProblemSpec):
    name = "cobench_multi_tugboat_routing_problem_with_variable_speed"
    description = _cfg.DESCRIPTION

    starter_code = 'def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:\n    """Solve a CO-Bench \'Multi-Tugboat Routing Problem with Variable Speed\' instance under the ISTH interface.\n\n    Args:\n      instance: dict containing every keyword argument the original CO-Bench\n                task expects. Access via instance[<key>]. The full schema and\n                semantics from the original CO-Bench task description below:\n\n        Solve a Multi-Tugboat Routing Problem with Variable Speed (MTRSP-VS) instance.\n\nArgs (keyword arguments — instance dict unpacked):\n\n────── Dimensions ──────\n    num_tasks         (int):  n, number of tasks.\n    num_tugboats      (int):  m, number of tugboats.\n    num_speed_levels  (int):  3 (slow / medium / fast — fixed).\n\n────── Per-task arrays (length n, indexed by s ∈ [0, n)) ──────\n    task_max_tugs           (list[int]):   Γₛ — max concurrent tugs.\n    task_min_horsepower     (list[float]): H^min_s — required HP sum.\n    task_time_window_lower  (list[float]): aₛ — earliest service start (h).\n    task_time_window_upper  (list[float]): bₛ — latest service start (h).\n    task_service_distance   (list[float]): dₛ — service length (n.m.).\n\n────── Per-tug arrays (length m, indexed by k ∈ [0, m)) ──────\n    tugboat_horsepower     (list[float]): HPₖ — engine power (kW).\n    tugboat_fuel_capacity  (list[float]): F^max_k — fuel tank (kg).\n    tugboat_alpha          (list[float]): αₖ — service fuel coef (kg/kW/h).\n    tugboat_beta           (list[float]): βₖ — transit fuel coef (kg/kW/h).\n\n────── Speed levels ──────\n    speed_level_names         (list[str]):   [\'slow\', \'medium\', \'fast\'].\n    speed_values              (list[float]): [6.0, 10.0, 15.0]  (kn).\n    speed_power_coefficients  (list[float]): [0.216, 1.0, 3.375] (ρₗ = (vₗ/v_medium)³).\n\n────── Distances (all in nautical miles) ──────\n    depot_to_task_distance  (list[float]):       length n, d(depot → entrance of task j).\n    task_to_depot_distance  (list[float]):       length n, d(exit of task i → depot).\n    task_to_task_distance   (list[list[float]]): n×n, [i][j] = d(exit i → entrance j).\n                                                 [i][i] = 0.0 (never used).\n\n────── System parameters ──────\n    big_M             (float): 1000.0 — Big-M for linearization (informational).\n    planning_horizon  (float): T^max — total operations horizon (hours, typically 24).\n    penalty_weight    (float): W — per-unexecuted-task penalty (typically 1e5).\n\nReturns:\n    dict with EXACTLY these 4 keys.\n\n    \'routes\' (dict[int, list[int]]):\n        Tug-id (0..m-1, ALL present) → list of task-ids (0-indexed) in\n        visit order. Empty list = unused tug. A task may appear in\n        MULTIPLE tugs\' routes (collaborative service).\n\n    \'service_speeds\' (dict[int, int]):\n        Task-id → speed level ∈ {0, 1, 2}. Only executed tasks.\n\n    \'start_times\' (dict[int, float]):\n        Task-id → service start time τₛ (hours, ≥ 0). Only executed tasks.\n\n    \'transit_speeds\' (dict[int, list[int]]):\n        Tug-id (0..m-1, ALL present) → list of speed levels for each arc.\n        Length = len(routes[k]) + 1. Arc 0 = depot→first;\n        arc i = tasks[i-1]→tasks[i]; last = lastTask→depot.\n\nExample (n=4 tasks, m=2 tugs; tug 0 solo-serves task 3 then collab-serves\ntask 1 with tug 1; task 0 unexecuted):\n    {\n      \'routes\':         {0: [3, 1], 1: [1]},\n      \'service_speeds\': {3: 1, 1: 0},\n      \'start_times\':    {3: 0.5, 1: 4.2},\n      \'transit_speeds\': {0: [1, 2, 1], 1: [1, 1]},\n    }\n\nThe "do nothing" return below is FEASIBLE — it pays maximum penalty\nW · n. A strong solver executes as many tasks as physically feasible\nwhile keeping total fuel below capacity.\n\n      tools:        see the "Available tools" section above.\n      time_limit_s: max wall-clock seconds (self-monitor with time.time()).\n\n    Returns:\n      The solution dict in the shape the task expects. For this task the\n      original CO-Bench solve template returns: below\n      (Implement an algorithm that produces such a dict for the given\n      instance and beats the trivial example.)\n    """\n    # Trivial placeholder; replace with your algorithm.\n    return below\n'

    feasibility_doc = _FEASIBILITY_DOC

    tools_description = [
        {
            "name": "is_feasible",
            "input": "solution: dict",
            "output": "(bool, str | None)",
            "purpose": (
                "Returns (True, None) if `solution` satisfies all problem constraints, "
                "else (False, error_message). Internally calls CO-Bench's eval_func "
                "(source shown in the # Feasibility check section above). Useful "
                "inside local search to filter infeasible neighbors before computing "
                "objective."
            ),
        },
        {
            "name": "objective",
            "input": "solution: dict",
            "output": "float",
            "purpose": _OBJECTIVE_PURPOSE,
        },
    ] + list(_EXTRA_TOOLS_DESC)

    def load_data(self, path: str) -> dict:
        """Path may be 'file::idx' to select a specific case from a multi-case file."""
        if "::" in path:
            actual, idx_str = path.rsplit("::", 1)
            idx = int(idx_str)
        else:
            actual, idx = path, 0
        result = _cfg.load_data(actual)
        if isinstance(result, list):
            if idx >= len(result):
                raise IndexError(f"case index {idx} >= {len(result)} cases in file")
            return result[idx]
        return result

    def tools(self, instance: dict) -> dict:
        inst = instance  # captured by closure

        def is_feasible(solution: dict):
            if not isinstance(solution, dict):
                return False, f"solution must be dict, got {type(solution).__name__}"
            merged = {**inst, **solution}
            try:
                raw = _filtered_eval_call(merged)
            except Exception as e:
                return False, f"eval_func raised: {type(e).__name__}: {e}"
            if raw is None:
                return False, "eval_func returned None"
            try:
                r = float(raw)
            except Exception as e:
                return False, f"eval_func returned non-numeric: {type(raw).__name__} ({e})"
            if not math.isfinite(r):
                return False, f"eval_func returned non-finite: {r}"
            if r < 0:
                return False, f"eval_func returned negative: {r}"
            return True, None

        def objective(solution: dict) -> float:
            merged = {**inst, **solution}
            try:
                raw = _filtered_eval_call(merged)
            except Exception:
                return 1e10
            try:
                r = float(raw)
            except Exception:
                return 1e10
            if not math.isfinite(r) or r < 0:
                return 1e10
            if _DIRECTION == "max":
                # Higher raw is better; reverse so ISTH lower=better holds.
                # We use 1.0/r when r>0 to keep CPI scale-invariant per problem.
                # (Add eps for stability.)
                return 1.0 / (r + 1e-12)
            return r

        base = {"is_feasible": is_feasible, "objective": objective}
        if _EXTRA_TOOLS_FACTORY is not None:
            try:
                extras = _EXTRA_TOOLS_FACTORY(inst)
            except Exception as e:
                # don't let a faulty extras module break the framework
                extras = {}
                import warnings as _w
                _w.warn(f"extras.extra_tools({type(inst)}) raised: {type(e).__name__}: {e}")
            if isinstance(extras, dict):
                base.update(extras)
        return base


SPEC = _CoBenchSpec()
