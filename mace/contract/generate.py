"""Stage-Zero orchestrator: NL description + instances -> drop-in contract.

Flow (per the user's design):
  I  : generate input schema + load_data; machine-parse all instances + reviewer LLM.
  O  : generate solution schema + trivial make_solution; shape check + reviewer LLM.
  T  : generate is_feasible + objective (no eval_func); validate by having an LLM
       write a real heuristic that runs through I -> O -> T.
  H  : generate a few domain helpers; validate by a heuristic that calls them.
Then assemble the drop-in files and run one final heuristic gate before promoting.
"""
from __future__ import annotations
import logging
import shutil
import tempfile
from pathlib import Path

from .context import ContractContext
from .input_designer import design_input
from .output_designer import design_output
from .tool_designer import design_tools
from .helper_designer import design_helpers
from .assemble import assemble_contract, build_spec
from .heuristic_check import heuristic_passes
from .validate import ContractGenerationError

logger = logging.getLogger(__name__)


def _read_sample(instance_paths, max_files: int = 2, max_chars: int = 12000) -> str:
    """Show the Input Designer the real byte layout of the SMALLEST instance
    files (a truncated slice of a huge file is misleading). Each file is capped
    at `max_chars`, truncated only at a whitespace boundary."""
    by_size = sorted(instance_paths, key=lambda p: Path(p).stat().st_size)
    blocks = []
    for p in by_size[:max_files]:
        txt = Path(p).read_text(encoding="utf-8", errors="replace")
        if len(txt) > max_chars:
            head = txt[:max_chars]
            ws = max(head.rfind(" "), head.rfind("\n"))
            if ws > 0:
                head = head[:ws]
            txt = head + "\n... [truncated — the rest of the file continues in the SAME format]"
        blocks.append(f"# ===== file: {Path(p).name} =====\n{txt}")
    return "\n\n".join(blocks)


def _find_instances(instances_dir: Path) -> list[str]:
    paths = sorted(str(p) for p in instances_dir.glob("*") if p.is_file())
    if not paths:  # instances may be nested in train/ test/ subdirs
        paths = sorted(str(p) for p in instances_dir.rglob("*") if p.is_file())
    return paths


def generate_contract(slug, nl_description, instances_dir, out_dir, llm_client,
                      example_slug="aircraft_landing", direction="min",
                      i_rep=3, smoke_time_limit_s=30.0):
    instances_dir = Path(instances_dir)
    instance_paths = _find_instances(instances_dir)
    if not instance_paths:
        raise ContractGenerationError(f"no instance files found in {instances_dir}")
    # All instances of one problem share the same format, so validating the
    # parser on the SMALLEST one or two is enough (and avoids tripping/slowing on
    # huge instances). The smallest is also used for the T / helper / final gates.
    by_size = sorted(instance_paths, key=lambda p: Path(p).stat().st_size)
    val_paths = by_size[:2]
    smallest = by_size[0]

    ctx = ContractContext(nl_description=nl_description,
                          sample_instance_text=_read_sample(instance_paths))
    ctx.direction = direction

    logger.info("Stage Zero: Input Designer (I) — generate + reviewer")
    design_input(ctx, llm_client, val_paths, example_slug, i_rep)
    logger.info("Stage Zero: Output Designer (O) — generate + reviewer")
    design_output(ctx, llm_client, example_slug, i_rep)
    logger.info("Stage Zero: Tool Designer (T core) — is_feasible + objective, validated by a heuristic")
    design_tools(ctx, llm_client, smallest, i_rep=i_rep)
    logger.info("Stage Zero: Helper Designer — a few helpers, validated by a heuristic that calls them")
    design_helpers(ctx, llm_client, smallest, i_rep=min(i_rep, 2))

    # Final gate: assemble the full contract and run one more real heuristic
    # through it before promoting to the destination.
    tmp = Path(tempfile.mkdtemp()) / slug
    spec = build_spec(ctx, slug, str(tmp))
    ok, err, _ = heuristic_passes(spec, llm_client, smallest, tries=2,
                                  time_limit_s=smoke_time_limit_s)
    if not ok:
        raise ContractGenerationError(f"final heuristic gate failed: {err}")

    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(tmp, out)
    logger.info("Stage Zero complete -> %s", out)
    return str(out)
