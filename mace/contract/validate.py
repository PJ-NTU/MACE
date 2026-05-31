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
