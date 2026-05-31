"""Tool Designer (T): generate eval_func + FEASIBILITY_STEPS_PY + optional extras."""
from __future__ import annotations
import ast
from pathlib import Path

from .validate import run_stage, smoke_eval

_PROMPT = (Path(__file__).parent / "prompts" / "tool_designer.md").read_text(encoding="utf-8")
_EXAMPLE_ROOT = Path(__file__).resolve().parents[2] / "problems"


def _example_block(slug: str) -> str:
    cfg = (_EXAMPLE_ROOT / slug / "config.py").read_text(encoding="utf-8")
    try:
        fs = (_EXAMPLE_ROOT / slug / "feasibility_steps.py").read_text(encoding="utf-8")
    except FileNotFoundError:
        fs = ""
    return cfg + "\n\n# feasibility_steps.py:\n" + fs


def _extract_str_const(src: str, name: str) -> str | None:
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets):
            if isinstance(node.value, ast.Constant):
                return node.value.value
    return None


def _extract_func_src(src: str, name: str) -> str | None:
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(src, node)
    return None


def design_tools(ctx, llm_client, instance_path, example_slug, i_rep: int = 3):
    gen_prompt = _PROMPT.format(
        nl=ctx.nl_description,
        input_schema=ctx.input_schema or "",
        output_schema=ctx.output_schema or "",
        placeholder=ctx.placeholder_solution_code or "",
        example=_example_block(example_slug),
    )

    def reflect(draft):
        return (f"{gen_prompt}\n\n# Your draft\n```python\n{draft}\n```\n\n"
                f"# Reflection\nEnumerate EVERY constraint in the problem description. "
                f"For each, point to the branch in eval_func that enforces it and the "
                f"labelled step in FEASIBILITY_STEPS_PY. Add any missing constraint, "
                f"then re-output one ```python block.")

    def smoke(draft):
        config_src = (ctx.load_data_code or "") + "\n" + (_extract_func_src(draft, "eval_func") or "")
        feasible = ctx.placeholder_solution_code  # defines make_solution
        infeasible = _extract_func_src(draft, "infeasible_make_solution")
        if infeasible is None:
            return False, "missing infeasible_make_solution() for smoke test"
        infeasible = infeasible.replace("def infeasible_make_solution", "def make_solution")
        steps = _extract_str_const(draft, "FEASIBILITY_STEPS_PY")
        if steps is None:
            return False, "missing FEASIBILITY_STEPS_PY string"
        try:
            compile(steps, "<fs>", "exec")
        except SyntaxError as e:
            return False, f"FEASIBILITY_STEPS_PY does not compile: {e}"
        return smoke_eval(config_src, instance_path, feasible, infeasible)

    src = run_stage(llm_client, gen_prompt, reflect, smoke, i_rep=i_rep,
                    stage_name="Tool Designer")
    ctx.eval_func_code = _extract_func_src(src, "eval_func")
    ctx.feasibility_steps = _extract_str_const(src, "FEASIBILITY_STEPS_PY")
    return ctx
