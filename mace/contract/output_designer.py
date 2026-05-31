"""Output Designer (O): define solution schema + trivial feasible placeholder."""
from __future__ import annotations
import ast
from pathlib import Path

from .validate import run_stage, _import_from_source

_PROMPT = (Path(__file__).parent / "prompts" / "output_designer.md").read_text(encoding="utf-8")
_EXAMPLE_ROOT = Path(__file__).resolve().parents[2] / "problems"


def _starter_example(slug: str) -> str:
    return (_EXAMPLE_ROOT / slug / "spec.py").read_text(encoding="utf-8")


def _extract_str_const(src: str, name: str) -> str | None:
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets):
            if isinstance(node.value, ast.Constant):
                return node.value.value
    return None


def design_output(ctx, llm_client, example_slug, i_rep: int = 3):
    gen_prompt = _PROMPT.format(
        nl=ctx.nl_description,
        input_schema=ctx.input_schema or "",
        load_data_code=ctx.load_data_code or "",
        example=_starter_example(example_slug),
    )

    def smoke(draft):
        try:
            mod = _import_from_source(
                (ctx.load_data_code or "") + "\n" + draft, "_ctr_output")
        except Exception as e:
            return False, f"output draft import failed: {type(e).__name__}: {e}"
        if not hasattr(mod, "make_solution"):
            return False, "no make_solution() defined"
        schema = _extract_str_const(draft, "OUTPUT_SCHEMA")
        return (schema is not None,
                None if schema else "missing OUTPUT_SCHEMA string")

    src = run_stage(llm_client, gen_prompt, lambda d: None, smoke, i_rep=i_rep,
                    stage_name="Output Designer")
    ctx.output_schema = _extract_str_const(src, "OUTPUT_SCHEMA") or ""
    ctx.placeholder_solution_code = src
    return ctx
