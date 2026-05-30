"""ISTH framework: universal runner for CO problems.

A CO problem is decomposed into 4 components (ISTH):
  I (Instance)  - input data contract; defined by load_data() return shape
  S (Solution)  - output structure contract; defined by starter_code return shape
  T (Tool)      - tool kit; evaluator + objective + domain helpers
  H (Heuristic) - the LLM-generated solve function

Spec authors implement IST. LLM generates H. Framework runs the whole thing.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Any
import os
import time
import importlib.util
import json
from pathlib import Path

# When MACE_NO_TOOLS=1, run_solve passes an empty dict instead of the spec's
# real tools. Used by the no-tools sensitivity experiment. Default off.
_NO_TOOLS = os.environ.get("MACE_NO_TOOLS") == "1"


# ============== problem spec ABC ==============

class ProblemSpec:
    """Base class for a CO problem specification under ISTH."""

    # ---- declarative fields (subclass MUST set) ----
    name: str = ""
    description: str = ""
    starter_code: str = ""

    # ---- optional fields (subclass MAY set; used by Stage Two prompt builder) ----
    # Verbose, human-readable description of what makes a solution feasible.
    # Typically embeds the `is_feasible` source so the LLM can see each
    # constraint inline. Empty string = no special feasibility section.
    feasibility_doc: str = ""

    # List of tool entries to render as a "skill-style" tools description for
    # the LLM. Each entry: {"name": str, "input": str, "output": str, "purpose": str}.
    # Do NOT include `is_feasible` here (that lives in feasibility_doc).
    # DO include `objective` (so the LLM knows it can call it). Empty list =
    # no tools section.
    tools_description: list = []

    # ---- I: input ----
    def load_data(self, path: str) -> dict:
        """Parse an instance file → instance dict.  Schema documented in description / starter_code."""
        raise NotImplementedError

    # ---- T: tools (containing evaluate + helpers) ----
    def tools(self, instance: dict) -> dict[str, Callable]:
        """Return callables. Framework REQUIRES these keys:
              'is_feasible': (solution) -> (bool, str | None)
              'objective':   (solution) -> float
            Plus any domain helpers.
        """
        raise NotImplementedError


# ============== run a single (spec, H, instance) tuple ==============

@dataclass
class RunResult:
    feasible: bool
    objective: float | None
    error_msg: str | None
    elapsed_s: float
    solution: dict | None


def run_solve(
    spec: ProblemSpec,
    instance: dict,
    solve_fn: Callable,
    time_limit_s: float = 60.0,
) -> RunResult:
    """Execute one (spec, instance, solve) triple. Returns scored result.

    NO-TOOLS MODE (env MACE_NO_TOOLS=1): pass empty dict to solve_fn but still
    use spec.tools(instance) internally for is_feasible/objective scoring.
    """
    real_tools = spec.tools(instance)
    is_feasible = real_tools["is_feasible"]
    objective = real_tools["objective"]
    # What the LLM-generated solve actually sees:
    tools_arg = {} if _NO_TOOLS else real_tools

    t0 = time.time()
    try:
        solution = solve_fn(instance, tools_arg, time_limit_s)
    except Exception as e:
        return RunResult(False, None, f"solve() raised: {type(e).__name__}: {e}",
                         time.time() - t0, None)
    elapsed = time.time() - t0

    if not isinstance(solution, dict):
        return RunResult(False, None, f"solve() returned non-dict: {type(solution).__name__}",
                         elapsed, None)

    fea_ok, fea_msg = is_feasible(solution)
    if not fea_ok:
        return RunResult(False, None, fea_msg, elapsed, solution)

    try:
        obj = objective(solution)
    except Exception as e:
        return RunResult(False, None, f"objective() raised: {e}", elapsed, solution)

    return RunResult(True, float(obj), None, elapsed, solution)


# ============== H loader (from .py file) ==============

def load_heuristic(path: str | Path, func_name: str = "solve") -> Callable:
    """Load a heuristic function from a .py file."""
    path = Path(path)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    if not hasattr(m, func_name):
        raise AttributeError(f"{path}: no `{func_name}` function defined")
    return getattr(m, func_name)
