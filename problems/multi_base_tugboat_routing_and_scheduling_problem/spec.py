"""ISTH spec for CO-Bench task: 'Multi-Base Tugboat Routing and Scheduling Problem' (auto-generated)."""
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
_mod_spec = importlib.util.spec_from_file_location("_cobench_cfg_multi_base_tugboat_routing_and_scheduling_problem", _CONFIG_PATH)
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
    name = "cobench_multi_base_tugboat_routing_and_scheduling_problem"
    description = _cfg.DESCRIPTION

    starter_code = 'def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:\n    """Solve a CO-Bench \'Multi-Base Tugboat Routing and Scheduling Problem\' instance under the ISTH interface.\n\n    Args:\n      instance: dict containing every keyword argument the original CO-Bench\n                task expects. Access via instance[<key>]. The full schema and\n                semantics from the original CO-Bench task description below:\n\n        Solve a Multi-Base Tugboat Routing and Scheduling Problem (MTRSP-MB) instance.\n\nArgs (keyword arguments — instance dict unpacked):\n\n────── Dimensions ──────\n    num_tasks    (int): n, number of tasks (1-indexed task ids 1..n).\n    num_tugboats (int): m, number of tugboats (0-indexed tug ids 0..m-1).\n    num_bases    (int): p, number of bases (base ids are -1..-p).\n\n────── Per-task arrays (length n, indexed by task id s ∈ {1..n} via s-1) ──────\n    task_max_tugs           (list[int]):   Γ_s — max tugboats per task.\n    task_min_horsepower     (list[float]): H_s^min — minimum HP sum required.\n    task_time_window_lower  (list[float]): a_s — earliest service start (hours).\n    task_time_window_upper  (list[float]): b_s — latest service start (hours).\n    task_service_time       (list[float]): T_s — fixed service duration (hours).\n\n────── Per-tugboat arrays (length m, indexed by tug k ∈ [0, m)) ──────\n    tugboat_horsepower      (list[float]): HP_k — tug horsepower (kW).\n    tugboat_fuel_capacity   (list[float]): F_k^max — fuel tank capacity (kg).\n    tugboat_alpha           (list[float]): α_k — service burn rate (kg/(kW·h)).\n    tugboat_beta            (list[float]): β_k — travel burn rate (kg/(kW·h)).\n    tugboat_base_assignment (list[int]):   home base of tug k (negative int, -1..-p).\n\n────── Per-base arrays (length p, indexed by base index |b|-1) ──────\n    base_capacity (list[int]): δ_b — max tugs that base b can dispatch / receive.\n\n────── Network & system parameters ──────\n    time_matrix       (dict[str, float]): sparse travel time matrix (hours).\n        Keys are STRINGS of the form \'<i>_<j>\' where:\n          • \'<b>_<s>\'         base origin b ∈ {-1..-p} → task s ∈ {1..n}\n          • \'<i>_<j>\'         task i → task j  (i, j ∈ {1..n}, i ≠ j)\n          • \'<s>_<n+|b|>\'     task s → base destination node n+|b| ∈ {n+1..n+p}\n        Use these keys verbatim in your start-time + fuel calculations.\n    big_M             (float): large constant for the original MILP (unused at run time).\n    planning_horizon  (float): T_max — total horizon in hours (typically 24).\n    penalty_weight    (float): W — penalty per unserved task (typically 10000).\n\nReturns:\n    dict with EXACTLY these 3 keys:\n\n      \'routes\' (list[list[int]]): length-m list. routes[k] is the\n          ordered list of task ids (∈ {1..n}) that tugboat k serves.\n          [] means tug k is unused. The implicit full route is\n          [home_base_k, *routes[k], n - home_base_k] but ONLY the task\n          ids are stored.\n\n      \'task_tugboats\' (dict[int, list[int]]): for each served task s,\n          the list of tugboat ids serving it. Keys are task ids ∈ {1..n};\n          values are lists of tug ids ∈ {0..m-1}. Must be CONSISTENT with\n          routes (C6): k ∈ task_tugboats[s]  iff  s ∈ routes[k].\n          Only includes executed (z_s = 1) tasks — unserved tasks are\n          absent from the dict.\n\n      \'task_start_times\' (dict[int, float]): for each served task s,\n          its start time τ_s (hours). Keys must equal task_tugboats.keys().\n          All tugs in task_tugboats[s] are implicitly synchronized to\n          start at this time.\n\nExample (n=4 tasks, m=3 tugs, p=2 bases; serves tasks 1,3 with collab):\n    {\n      \'routes\': [\n          [1, 3],          # tugboat 0 (base -1): -1 → task 1 → task 3 → dest 5\n          [3],             # tugboat 1 (base -1): -1 → task 3       → dest 5\n          [],              # tugboat 2 (base -2): unused\n      ],\n      \'task_tugboats\':    {1: [0],    3: [0, 1]},\n      \'task_start_times\': {1: 1.5,    3: 5.0},\n    }\n    Tasks 2 and 4 are unserved (z_2 = z_4 = 0), each paying W penalty.\n\nThe "do nothing" return below is FEASIBLE — it just pays maximum\npenalty W·n. A strong solver serves as many tasks as it can while\nrespecting all 14 constraints (see eval_func source for the exact check).\n\n      tools:        see the "Available tools" section above.\n      time_limit_s: max wall-clock seconds (self-monitor with time.time()).\n\n    Returns:\n      The solution dict in the shape the task expects. For this task the\n      original CO-Bench solve template returns: below\n      (Implement an algorithm that produces such a dict for the given\n      instance and beats the trivial example.)\n    """\n    # Trivial placeholder; replace with your algorithm.\n    return below\n'

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
