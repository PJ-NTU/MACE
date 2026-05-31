"""Input Designer (I): generate DESCRIPTION + load_data + input schema.

Validation = machine smoke (load_data actually parses every instance) AND an
independent reviewer LLM that judges whether the schema correctly captures the
problem. Either failing triggers bounded repair."""
from __future__ import annotations
import ast
import logging
from pathlib import Path

from .validate import run_stage, smoke_input, ContractGenerationError
from .reviewer import review

logger = logging.getLogger(__name__)

_PROMPT = (Path(__file__).parent / "prompts" / "input_designer.md").read_text(encoding="utf-8")
_EXAMPLE_ROOT = Path(__file__).resolve().parents[2] / "problems"


def _example_block(slug: str) -> str:
    return (_EXAMPLE_ROOT / slug / "config.py").read_text(encoding="utf-8")


def _extract_description(src: str) -> str | None:
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "DESCRIPTION" \
                        and isinstance(node.value, ast.Constant):
                    return node.value.value
    return None


def _extract_schema_comment(src: str) -> str:
    lines = [ln[1:].strip() for ln in src.splitlines() if ln.lstrip().startswith("#")]
    return "\n".join(lines)


def _extract_load_data_with_imports(src: str) -> str | None:
    """Pull just the module-level imports + the load_data function out of `src`,
    so a whole config.py (eval_func, solve, etc.) can be passed in and only its
    load_data is adopted."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    imports, func = [], None
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            seg = ast.get_source_segment(src, node)
            if seg:
                imports.append(seg)
        elif isinstance(node, ast.FunctionDef) and node.name == "load_data":
            func = ast.get_source_segment(src, node)
    if func is None:
        return None
    return ("\n".join(imports) + "\n\n" + func).strip() if imports else func


def adopt_input(ctx, load_data_code, instance_paths):
    """Adopt a user-supplied load_data verbatim instead of generating one.

    Skips the LLM generator AND the reviewer — the user already knows the data
    format. The machine smoke check still runs (load_data must actually parse the
    real instances), so a wrong parser is still caught."""
    clean = _extract_load_data_with_imports(load_data_code)
    if clean is None:
        raise ContractGenerationError(
            "[Input Designer] user-supplied source defines no load_data()")
    ok, err = smoke_input(clean, instance_paths, [])
    if not ok:
        raise ContractGenerationError(
            f"[Input Designer] user-supplied load_data failed machine parse: {err}")
    ctx.load_data_code = clean
    # Use the user's own --description here, NOT any DESCRIPTION embedded in the
    # adopted file: the latter may promise tools/helpers that don't exist yet and
    # would mislead the downstream T/H heuristic-validation prompts.
    ctx.description = ctx.nl_description
    ctx.input_schema = _extract_schema_comment(clean) or ctx.nl_description
    logger.info("[Input Designer] adopted user-supplied load_data "
                "(machine-checked; reviewer skipped)")
    return ctx


def design_input(ctx, llm_client, instance_paths, example_slug, i_rep: int = 3):
    gen_prompt = _PROMPT.format(
        nl=ctx.nl_description,
        sample=ctx.sample_instance_text,
        example=_example_block(example_slug),
    )

    def smoke(draft):
        ok, err = smoke_input(draft, instance_paths, [])
        if not ok:
            return False, f"[machine] {err}"
        ok2, fb = review(
            llm_client, "input schema + load_data parser", ctx.nl_description, draft,
            extra_context=f"# Raw sample instance bytes\n```\n{ctx.sample_instance_text[:4000]}\n```",
        )
        if not ok2:
            return False, f"[reviewer] {fb}"
        return True, None

    src = run_stage(llm_client, gen_prompt, lambda d: None, smoke, i_rep=i_rep,
                    stage_name="Input Designer")
    ctx.load_data_code = src
    ctx.description = _extract_description(src) or ctx.nl_description
    ctx.input_schema = _extract_schema_comment(src) or ""
    return ctx
