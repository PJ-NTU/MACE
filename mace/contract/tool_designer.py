"""Tool Designer (T): generate the eval_func (feasibility + objective ground truth).

Slim version: one prompt produces eval_func + a deliberately-infeasible solution
builder used only to smoke-test that eval_func rejects bad solutions. No separate
FEASIBILITY_STEPS_PY artifact (the eval_func source is shown to the heuristic
generator directly via the spec's feasibility_doc)."""
from __future__ import annotations
import ast
from pathlib import Path

from .validate import run_stage, smoke_eval

_PROMPT = (Path(__file__).parent / "prompts" / "tool_designer.md").read_text(encoding="utf-8")
_EXAMPLE_ROOT = Path(__file__).resolve().parents[2] / "problems"


def _example_block(slug: str) -> str:
    return (_EXAMPLE_ROOT / slug / "config.py").read_text(encoding="utf-8")


def _extract_func_src(src: str, name: str) -> str | None:
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(src, node)
    return None


def _extract_imports(src: str) -> str:
    """Module-level import statements from the draft. eval_func often depends on
    `import math` / numpy written at module scope; extracting only the function
    body would drop them and raise NameError at run time."""
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


def design_tools(ctx, llm_client, instance_path, example_slug, i_rep: int = 3):
    gen_prompt = _PROMPT.format(
        nl=ctx.nl_description,
        input_schema=ctx.input_schema or "",
        output_schema=ctx.output_schema or "",
        placeholder=ctx.placeholder_solution_code or "",
        example=_example_block(example_slug),
    )

    def smoke(draft):
        # Guard: a non-Python draft would make the ast extractor below crash the
        # whole stage instead of triggering repair. Turn that into a smoke failure.
        try:
            compile(draft, "<tool_draft>", "exec")
        except SyntaxError as e:
            return False, f"draft has invalid Python syntax: {e.msg} (line {e.lineno})"
        imports = _extract_imports(draft)
        config_src = (ctx.load_data_code or "") + "\n" + (_func_with_imports(draft, "eval_func") or "")
        feasible = ctx.placeholder_solution_code  # defines make_solution
        infeasible = _extract_func_src(draft, "infeasible_make_solution")
        if infeasible is None:
            return False, "missing infeasible_make_solution() for smoke test"
        infeasible = infeasible.replace("def infeasible_make_solution", "def make_solution")
        if imports:  # the builder may also need the draft's imports (e.g. math)
            infeasible = imports + "\n\n" + infeasible
        return smoke_eval(config_src, instance_path, feasible, infeasible)

    src = run_stage(llm_client, gen_prompt, lambda d: None, smoke, i_rep=i_rep,
                    stage_name="Tool Designer")
    ctx.eval_func_code = _func_with_imports(src, "eval_func")
    ctx.feasibility_steps = None
    return ctx
