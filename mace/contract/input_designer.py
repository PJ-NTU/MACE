"""Input Designer (I): generate DESCRIPTION + load_data + input schema."""
from __future__ import annotations
import ast
from pathlib import Path

from .validate import run_stage, smoke_input

_PROMPT = (Path(__file__).parent / "prompts" / "input_designer.md").read_text(encoding="utf-8")
_EXAMPLE_ROOT = Path(__file__).resolve().parents[2] / "problems"


def _example_block(slug: str) -> str:
    return (_EXAMPLE_ROOT / slug / "config.py").read_text(encoding="utf-8")


def _extract_description(src: str) -> str | None:
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "DESCRIPTION":
                    if isinstance(node.value, ast.Constant):
                        return node.value.value
    return None


def _extract_schema_comment(src: str) -> str:
    lines = [ln[1:].strip() for ln in src.splitlines() if ln.lstrip().startswith("#")]
    return "\n".join(lines)


def design_input(ctx, llm_client, instance_paths, required_keys, example_slug,
                 i_rep: int = 3):
    gen_prompt = _PROMPT.format(
        nl=ctx.nl_description,
        sample=ctx.sample_instance_text,
        example=_example_block(example_slug),
    )

    def reflect(draft):
        return (f"{gen_prompt}\n\n# Your draft\n```python\n{draft}\n```\n\n"
                f"# Reflection\nCompare your load_data output dict against the raw "
                f"sample instance. List any field you missed, mis-typed, or invented, "
                f"then output the corrected DESCRIPTION + load_data in one ```python block.")

    def smoke(draft):
        return smoke_input(draft, instance_paths, required_keys)

    src = run_stage(llm_client, gen_prompt, reflect, smoke, i_rep=i_rep,
                    stage_name="Input Designer")
    ctx.load_data_code = src
    ctx.description = _extract_description(src) or ctx.nl_description
    ctx.input_schema = _extract_schema_comment(src) or ", ".join(required_keys)
    return ctx
