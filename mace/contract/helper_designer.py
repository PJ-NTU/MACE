"""Helper Designer: generate a few domain helper tools (just a handful), then
validate by having an LLM write a simple heuristic that actually CALLS them and
runs through the contract. If the helpers are not usable, repair.

Helpers are optional: if the model decides none are needed, the stage passes
with an empty set."""
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


def design_helpers(ctx, llm_client, instance_path, i_rep: int = 2):
    base_tools = list(ctx.tools_description or [])
    gen_prompt = _PROMPT.format(
        nl=ctx.nl_description,
        input_schema=ctx.input_schema or "",
        output_schema=ctx.output_schema or "",
        is_feasible=ctx.is_feasible_code or "",
        objective=ctx.objective_code or "",
    )

    def smoke(draft):
        try:
            compile(draft, "<helpers_draft>", "exec")
        except SyntaxError as e:
            return False, f"helpers draft invalid syntax: {e.msg} (line {e.lineno})"
        funcs = _helper_funcs(draft)
        if not funcs:
            return True, None  # model judged no helpers needed — acceptable
        names = [n for n, _ in funcs]
        tools_desc = base_tools + [
            {"name": n, "input": "solution/partial args (instance is bound)",
             "output": "...", "purpose": p} for n, p in funcs
        ]
        trial = replace(ctx, helpers_code=draft, helper_names=names,
                        tools_description=tools_desc)
        try:
            tmp = Path(tempfile.mkdtemp()) / "helper_trial"
            spec = build_spec(trial, "helper_trial", str(tmp))
        except Exception as e:
            return False, f"contract failed to assemble/import with helpers: {type(e).__name__}: {e}"
        hint = ("# You SHOULD call these helper tools in your solve:\n"
                + "\n".join(f"  - tools['{n}']  ({p})" for n, p in funcs))
        ok, err, _ = heuristic_passes(spec, llm_client, instance_path, hint=hint, tries=2)
        if not ok:
            return False, f"a heuristic could not use the helpers: {err}"
        return True, None

    src = run_stage(llm_client, gen_prompt, lambda d: None, smoke, i_rep=i_rep,
                    stage_name="Helper Designer")
    funcs = _helper_funcs(src)
    if funcs:
        ctx.helpers_code = src
        ctx.helper_names = [n for n, _ in funcs]
        ctx.tools_description = base_tools + [
            {"name": n, "input": "solution/partial args (instance is bound)",
             "output": "...", "purpose": p} for n, p in funcs
        ]
    else:
        ctx.helpers_code = None
        ctx.helper_names = []
        ctx.tools_description = base_tools
    return ctx
