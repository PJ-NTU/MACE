"""Helper Designer: generate a few domain helper tools (a handful), then validate
EACH ONE INDIVIDUALLY — an LLM writes a simple heuristic that is required to call
that specific helper, and we instrument the helper to confirm it was actually
invoked and ran without error. A broken or unused helper is rejected.

Helpers are optional: if the model decides none are needed, the stage passes."""
from __future__ import annotations
import ast
import tempfile
from dataclasses import replace
from pathlib import Path

from .validate import run_stage
from .assemble import build_spec
from .heuristic_check import heuristic_passes

_PROMPT = (Path(__file__).parent / "prompts" / "helper_designer.md").read_text(encoding="utf-8")


def _helper_funcs(src: str):
    """Return [(name, purpose)] for every top-level function in the draft."""
    out = []
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef):
            doc = ast.get_docstring(node) or ""
            purpose = doc.strip().splitlines()[0] if doc.strip() else f"helper {node.name}"
            out.append((node.name, purpose))
    return out


def _validate_one_helper(spec, cfg_module, name, llm_client, instance_path, tries=2):
    """A heuristic must CALL tools['name'] and run feasibly. Instrument the
    config-level helper to count invocations so an unused/ignored helper fails."""
    orig = getattr(cfg_module, name)
    calls = {"n": 0}

    def wrapped(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    setattr(cfg_module, name, wrapped)
    try:
        hint = (f"# REQUIREMENT: your solve MUST call `tools['{name}'](...)` at "
                f"least once — exercising that tool is the whole point of this run.")
        err = "no heuristic generated"
        for _ in range(tries):
            calls["n"] = 0
            ok, err, _code = heuristic_passes(spec, llm_client, instance_path,
                                              hint=hint, tries=1)
            if ok and calls["n"] > 0:
                return True, None
            if ok and calls["n"] == 0:
                err = f"heuristic ran but never called tools['{name}']"
        return False, err
    finally:
        setattr(cfg_module, name, orig)


def design_helpers(ctx, llm_client, instance_path, i_rep: int = 2):
    base_tools = list(ctx.tools_description or [])
    gen_prompt = _PROMPT.format(
        nl=ctx.nl_description,
        input_schema=ctx.input_schema or "",
        output_schema=ctx.output_schema or "",
        is_feasible=ctx.is_feasible_code or "",
        objective=ctx.objective_code or "",
    )

    def _tools_desc(funcs):
        return base_tools + [
            {"name": n, "input": "solution/partial args (instance is bound)",
             "output": "...", "purpose": p} for n, p in funcs
        ]

    def smoke(draft):
        try:
            compile(draft, "<helpers_draft>", "exec")
        except SyntaxError as e:
            return False, f"helpers draft invalid syntax: {e.msg} (line {e.lineno})"
        funcs = _helper_funcs(draft)
        if not funcs:
            return True, None  # model judged no helpers needed — acceptable
        names = [n for n, _ in funcs]
        trial = replace(ctx, helpers_code=draft, helper_names=names,
                        tools_description=_tools_desc(funcs))
        try:
            tmp = Path(tempfile.mkdtemp()) / "helper_trial"
            spec = build_spec(trial, "helper_trial", str(tmp))
            cfg_module = spec._cfg_module
        except Exception as e:
            return False, f"contract failed to assemble/import with helpers: {type(e).__name__}: {e}"
        # Validate EACH helper individually (its own heuristic, must be called).
        for name, _purpose in funcs:
            ok, err = _validate_one_helper(spec, cfg_module, name, llm_client, instance_path)
            if not ok:
                return False, f"helper '{name}': {err}"
        return True, None

    src = run_stage(llm_client, gen_prompt, lambda d: None, smoke, i_rep=i_rep,
                    stage_name="Helper Designer")
    funcs = _helper_funcs(src)
    if funcs:
        ctx.helpers_code = src
        ctx.helper_names = [n for n, _ in funcs]
        ctx.tools_description = _tools_desc(funcs)
    else:
        ctx.helpers_code = None
        ctx.helper_names = []
        ctx.tools_description = base_tools
    return ctx
