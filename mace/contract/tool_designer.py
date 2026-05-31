"""Tool Designer — core T: generate is_feasible(instance, solution) and
objective(instance, solution). NO eval_func (a new CO problem has none).

Validation follows the user's design: build a temp spec from (I + this T) and
have an LLM generate a real heuristic that runs through I -> O -> T. If a solver
can produce a solution that is_feasible accepts and objective scores, the I/O/T
core works together. Failure triggers bounded repair of T."""
from __future__ import annotations
import ast
import math
import tempfile
from dataclasses import replace
from pathlib import Path

from .validate import run_stage
from .assemble import build_spec
from .heuristic_check import heuristic_passes

_PROMPT = (Path(__file__).parent / "prompts" / "tool_designer.md").read_text(encoding="utf-8")


def _placeholder_is_feasible(spec, instance_path) -> bool:
    """True if the O placeholder make_solution is accepted by is_feasible and
    scored by objective — a guaranteed-feasible witness that the T core runs
    correctly, without needing a heuristic to FIND feasibility."""
    try:
        inst = spec.load_data(instance_path)
        cfg = getattr(spec, "_cfg_module", None)
        if cfg is None or not hasattr(cfg, "make_solution"):
            return False
        sol = cfg.make_solution(inst)
        tools = spec.tools(inst)
        ok, _msg = tools["is_feasible"](sol)
        if not ok:
            return False
        val = tools["objective"](sol)
        return isinstance(val, (int, float)) and math.isfinite(val) and val < 1e10
    except Exception:
        return False


def _extract_func_src(src: str, name: str) -> str | None:
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(src, node)
    return None


def _extract_imports(src: str) -> str:
    lines = []
    for node in ast.parse(src).body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            seg = ast.get_source_segment(src, node)
            if seg:
                lines.append(seg)
    return "\n".join(lines)


def _func_with_imports(src: str, name: str) -> str | None:
    func = _extract_func_src(src, name)
    if func is None:
        return None
    imports = _extract_imports(src)
    return (imports + "\n\n" + func).strip() if imports else func


def design_tools(ctx, llm_client, instance_path, example_slug=None, i_rep: int = 3):
    gen_prompt = _PROMPT.format(
        nl=ctx.nl_description,
        input_schema=ctx.input_schema or "",
        output_schema=ctx.output_schema or "",
        placeholder=ctx.placeholder_solution_code or "",
    )

    def smoke(draft):
        try:
            compile(draft, "<tool_draft>", "exec")
        except SyntaxError as e:
            return False, f"draft has invalid Python syntax: {e.msg} (line {e.lineno})"
        is_feas = _func_with_imports(draft, "is_feasible")
        obj = _func_with_imports(draft, "objective")
        if is_feas is None:
            return False, "missing is_feasible(instance, solution)"
        if obj is None:
            return False, "missing objective(instance, solution)"
        # Build a temp spec from (I + this T).
        trial = replace(ctx, is_feasible_code=is_feas, objective_code=obj,
                        helpers_code=None, helper_names=[], tools_description=[])
        try:
            tmp = Path(tempfile.mkdtemp()) / "tool_trial"
            spec = build_spec(trial, "tool_trial", str(tmp))
        except Exception as e:
            return False, f"contract failed to assemble/import: {type(e).__name__}: {e}"
        # Fast path: the O placeholder make_solution is meant to be feasible by
        # construction. If is_feasible accepts it and objective scores it, the T
        # core demonstrably runs correctly on a known-feasible witness — no need
        # to depend on a generated heuristic FINDING feasibility (which is itself
        # hard for tightly-constrained problems).
        if _placeholder_is_feasible(spec, instance_path):
            return True, None
        # Otherwise fall back to having an LLM write a real heuristic that must
        # produce a feasible, scored solution through I -> O -> T.
        ok, err, _ = heuristic_passes(spec, llm_client, instance_path,
                                      hint="", tries=3)
        if not ok:
            return False, f"a heuristic could not solve through I/O/T: {err}"
        return True, None

    src = run_stage(llm_client, gen_prompt, lambda d: None, smoke, i_rep=i_rep,
                    stage_name="Tool Designer (core)")
    ctx.is_feasible_code = _func_with_imports(src, "is_feasible")
    ctx.objective_code = _func_with_imports(src, "objective")
    # Always advertise objective to the heuristic generator.
    ctx.tools_description = [{
        "name": "objective", "input": "solution: dict", "output": "float",
        "purpose": "Objective value of a solution (LOWER IS BETTER).",
    }]
    return ctx
