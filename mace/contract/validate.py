"""Shared stage control flow: generate -> reflect -> smoke -> bounded repair."""
from __future__ import annotations
import logging
from typing import Callable, Optional

from mace.evolution.operators._common import extract_python

logger = logging.getLogger(__name__)


class ContractGenerationError(RuntimeError):
    """Raised when a stage cannot pass its smoke test within the repair budget."""


def run_stage(
    llm_client,
    gen_prompt: str,
    reflect_prompt_fn: Callable[[str], Optional[str]],
    smoke_fn: Callable[[str], tuple[bool, Optional[str]]],
    repair_prompt_fn: Optional[Callable[[str, str], str]] = None,
    i_rep: int = 3,
    stage_name: str = "stage",
) -> str:
    """Run one designer stage to a passing artifact. Raises ContractGenerationError on budget exhaustion."""
    draft = extract_python(llm_client.chat(gen_prompt))

    reflect_prompt = reflect_prompt_fn(draft)
    if reflect_prompt:
        draft = extract_python(llm_client.chat(reflect_prompt))

    passed, err = smoke_fn(draft)
    rep = 0
    while not passed and rep < i_rep:
        rep += 1
        if repair_prompt_fn is not None:
            rp = repair_prompt_fn(draft, err or "")
        else:
            rp = (f"# Broken artifact\n```python\n{draft}\n```\n\n"
                  f"# Failure report\n```\n{err}\n```\n\n"
                  f"Fix the artifact so it passes. Output ONLY the corrected "
                  f"Python in one fenced ```python block.")
        draft = extract_python(llm_client.chat(rp))
        passed, err = smoke_fn(draft)

    if not passed:
        raise ContractGenerationError(
            f"[{stage_name}] failed smoke test after {rep} repairs: {err}")
    logger.info("[%s] passed%s", stage_name, f" after {rep} repairs" if rep else "")
    return draft


import importlib.util
import tempfile
import os
from pathlib import Path


def _import_from_source(src: str, mod_name: str):
    """Write src to a temp .py and import it as a module. Returns the module."""
    tmp = Path(tempfile.mkdtemp()) / f"{mod_name}.py"
    tmp.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(mod_name, tmp)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def smoke_input(load_data_src: str, instance_paths: list[str],
                required_keys: list[str]) -> tuple[bool, Optional[str]]:
    """Smoke test I: load_data parses EVERY instance into a dict with required keys."""
    try:
        mod = _import_from_source(load_data_src, "_ctr_input")
    except Exception as e:
        return False, f"load_data source import failed: {type(e).__name__}: {e}"
    if not hasattr(mod, "load_data"):
        return False, "no load_data() defined"
    for p in instance_paths:
        try:
            result = mod.load_data(p)
        except Exception as e:
            return False, f"load_data('{os.path.basename(p)}') raised: {type(e).__name__}: {e}"
        inst = result[0] if isinstance(result, list) else result
        if not isinstance(inst, dict):
            return False, f"load_data returned {type(inst).__name__}, expected dict"
        missing = [k for k in required_keys if k not in inst]
        if missing:
            return False, f"instance missing keys {missing} (file {os.path.basename(p)})"
    return True, None


def smoke_eval(config_src: str, instance_path: str,
               feasible_solution_code: str,
               infeasible_solution_code: str) -> tuple[bool, Optional[str]]:
    """Smoke test T: eval_func returns finite cost on a feasible solution and
    raises/penalizes on a deliberately infeasible one. Each *_solution_code is a
    snippet defining `make_solution(instance) -> dict`."""
    try:
        cfg = _import_from_source(config_src, "_ctr_config")
    except Exception as e:
        return False, f"config import failed: {type(e).__name__}: {e}"
    for fn in ("load_data", "eval_func"):
        if not hasattr(cfg, fn):
            return False, f"config.py missing {fn}()"
    try:
        raw = cfg.load_data(instance_path)
        inst = raw[0] if isinstance(raw, list) else raw
    except Exception as e:
        return False, f"load_data raised in eval smoke: {e}"

    def _mk(code):
        ns = {}
        exec(compile(code, "<sol>", "exec"), ns)
        return ns["make_solution"](inst)

    try:
        feas = _mk(feasible_solution_code)
    except Exception as e:
        return False, f"feasible solution builder raised: {e}"
    try:
        cost = cfg.eval_func(**{**inst, **feas})
        cost = float(cost)
    except Exception as e:
        return False, f"eval_func REJECTED a feasible solution: {type(e).__name__}: {e}"
    if not (cost == cost) or cost < 0:  # NaN / negative
        return False, f"eval_func returned bad cost on feasible solution: {cost}"

    try:
        infeas = _mk(infeasible_solution_code)
        cfg.eval_func(**{**inst, **infeas})
        return False, "eval_func ACCEPTED a deliberately infeasible solution (too lax)"
    except Exception:
        pass  # expected: infeasible should raise
    return True, None
