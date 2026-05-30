"""ISTH spec for CO-Bench task: 'Container loading with weight restrictions' (auto-generated)."""
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
_mod_spec = importlib.util.spec_from_file_location("_cobench_cfg_container_loading_with_weight_restrictions", _CONFIG_PATH)
_cfg = importlib.util.module_from_spec(_mod_spec)
_mod_spec.loader.exec_module(_cfg)


_DIRECTION = 'max'   # eval_func returns utilization (0..1, higher=better); framework inverts via 1/raw


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
    name = "cobench_container_loading_with_weight_restrictions"
    description = _cfg.DESCRIPTION

    starter_code = 'def solve(instance: dict, tools: dict, time_limit_s: float) -> dict:\n    """Solve a CO-Bench \'Container loading with weight restrictions\' instance under the ISTH interface.\n\n    Args:\n      instance: dict containing every keyword argument the original CO-Bench\n                task expects. Access via instance[<key>]. The full schema and\n                semantics from the original CO-Bench task description below:\n\n        Solves the Container Loading with Weight Restrictions problem.\n\nInput kwargs (for one test case):\n  - container (tuple of int): (L, W, H) representing the container dimensions in cm.\n  - n (int): the number of box types.\n  - cargo_vol (float): the total cargo volume in m³ (provided for consistency).\n  - box_types (list of dict): one per box type. Each dictionary has the keys:\n        \'length\' (int), \'length_flag\' (int),\n        \'width\' (int),  \'width_flag\' (int),\n        \'height\' (int), \'height_flag\' (int),\n        \'count\' (int),  \'weight\' (float),\n        \'lb1\' (float), \'lb2\' (float), \'lb3\' (float).\n\nThe problem is to select and place boxes (each possibly in one of three allowed orientations)\ninside the container so as to maximize the ratio of the total volume of placed boxes (each based on its original dimensions)\nto the container’s volume, while obeying placement, support, and load–bearing constraints.\n\nEvaluation metric:\n  The score is the container volume utilization (i.e. total placed boxes volume divided by container volume)\n  if the solution is valid according to all constraints; otherwise the score is 0.0.\n\nPlaceholder implementation: No boxes are placed.\n\nReturns a dictionary with keys:\n  - \'instance\': instance number (int),\n  - \'util\': achieved utilization (float),\n  - \'m\': number of placements (int),\n  - \'placements\': a list of placements; each placement is a dict with keys:\n        \'box_type\' (int, 1-indexed), \'orientation\' (int: 1, 2, or 3),\n        \'x\', \'y\', \'z\' (floats for the lower–left–front corner in cm).\n\n      tools:        see the "Available tools" section above.\n      time_limit_s: max wall-clock seconds (self-monitor with time.time()).\n\n    Returns:\n      The solution dict in the shape the task expects. For this task the\n      original CO-Bench solve template returns: an\n      (Implement an algorithm that produces such a dict for the given\n      instance and beats the trivial example.)\n    """\n    # Trivial placeholder; replace with your algorithm.\n    return an\n'

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
