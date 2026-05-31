"""Output Designer (O): define solution schema + trivial feasible placeholder.

Validation = shape check (make_solution + OUTPUT_SCHEMA present, importable) AND
an independent reviewer LLM judging whether the solution shape fits the problem."""
from __future__ import annotations
import ast
import logging
from pathlib import Path

from .validate import run_stage, _import_from_source
from .reviewer import review

logger = logging.getLogger(__name__)

_PROMPT = (Path(__file__).parent / "prompts" / "output_designer.md").read_text(encoding="utf-8")
_EXAMPLE_ROOT = Path(__file__).resolve().parents[2] / "problems"


def _starter_example(slug: str) -> str:
    return (_EXAMPLE_ROOT / slug / "spec.py").read_text(encoding="utf-8")


def _extract_str_const(src: str, name: str) -> str | None:
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets) \
                and isinstance(node.value, ast.Constant):
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
            mod = _import_from_source((ctx.load_data_code or "") + "\n" + draft, "_ctr_output")
        except Exception as e:
            return False, f"[machine] output draft import failed: {type(e).__name__}: {e}"
        if not hasattr(mod, "make_solution"):
            return False, "[machine] no make_solution() defined"
        schema = _extract_str_const(draft, "OUTPUT_SCHEMA")
        if schema is None:
            return False, "[machine] missing OUTPUT_SCHEMA string"
        # ADVISORY reviewer (logged, NON-blocking). The output schema's real
        # correctness is validated downstream: the Tool stage + final gate require
        # a real heuristic to produce an O-conforming solution that is_feasible
        # accepts and objective scores. A blocking LLM reviewer here (at this model
        # tier) mostly produces false rejections on naming/wording nuances, so its
        # opinion is recorded but does not block. (Schema presence/import are still
        # hard machine checks above.)
        try:
            ok2, fb = review(
                llm_client, "output (solution) schema",
                ctx.nl_description, f'OUTPUT_SCHEMA = """{schema}"""',
                extra_context=f"# Locked input schema\n{ctx.input_schema or ''}")
            if not ok2:
                logger.info("[Output Designer] advisory reviewer note (not blocking): %s",
                            (fb or "")[:200])
        except Exception:
            pass
        return True, None

    src = run_stage(llm_client, gen_prompt, lambda d: None, smoke, i_rep=i_rep,
                    stage_name="Output Designer")
    ctx.output_schema = _extract_str_const(src, "OUTPUT_SCHEMA") or ""
    ctx.placeholder_solution_code = src
    return ctx
