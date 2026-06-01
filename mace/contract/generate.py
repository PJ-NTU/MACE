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
from .input_designer import design_input, adopt_input
from .output_designer import design_output
from .tool_designer import design_tools, _placeholder_is_feasible
from .helper_designer import design_helpers
from .assemble import assemble_contract, build_spec
from .validate import ContractGenerationError, _import_from_source

logger = logging.getLogger(__name__)


def _shape(v) -> str:
    if isinstance(v, bool):
        return f"bool (e.g. {v})"
    if isinstance(v, int):
        return f"int (e.g. {v})"
    if isinstance(v, float):
        return f"float (e.g. {v})"
    if isinstance(v, str):
        return f"str (e.g. {v[:25]!r})"
    if isinstance(v, dict):
        return f"dict with keys {list(v.keys())[:12]}"
    if isinstance(v, (list, tuple)):
        n = len(v)
        if v and isinstance(v[0], (list, tuple)):
            return f"list of {n} rows (≈ {n}x{len(v[0])} matrix of {type(v[0][0]).__name__ if v[0] else '?'})"
        if v and isinstance(v[0], dict):
            return f"list of {n} dicts, each with keys {list(v[0].keys())[:12]}"
        if v:
            return f"list of {n} {type(v[0]).__name__}"
        return "empty list"
    return type(v).__name__


def _describe_instance(inst: dict) -> str:
    """A precise, human-readable view of a REAL parsed instance dict: every
    top-level key with its type/shape and a sample value. Fed to the O/T/helper
    designers so they use the ACTUAL instance keys instead of guessing names."""
    lines = ["The `instance` dict (obtained by actually parsing a sample instance) "
             "has EXACTLY these top-level keys — use these names verbatim:"]
    for k, v in inst.items():
        lines.append(f"  - instance[{k!r}]: {_shape(v)}")
        if isinstance(v, (list, tuple)) and v and isinstance(v[0], dict):
            lines.append(f"      (each element dict keys: {list(v[0].keys())})")
    return "\n".join(lines)


def _derive_input_schema(load_data_code: str, instance_path: str) -> str | None:
    try:
        mod = _import_from_source(load_data_code, "_ctr_introspect")
        raw = mod.load_data(instance_path)
        inst = raw[0] if isinstance(raw, list) else raw
        if isinstance(inst, dict):
            return _describe_instance(inst)
    except Exception:
        return None
    return None


def _read_sample(instance_paths, max_files: int = 2, max_chars: int = 16000) -> str:
    """Show the Input Designer the SMALLEST instance files, each prefixed with its
    exact TOTAL line and whitespace-token count. That global count is the key the
    model needs to infer matrix-shaped formats (e.g. an N followed by an N*N matrix
    whose rows wrap across physical lines) even when the body is truncated."""
    by_size = sorted(instance_paths, key=lambda p: Path(p).stat().st_size)
    blocks = []
    for p in by_size[:max_files]:
        full = Path(p).read_text(encoding="utf-8", errors="replace")
        ntok = len(full.split())
        nlines = full.count("\n") + 1
        header = (f"# ===== file: {Path(p).name} — {nlines} physical lines, {ntok} "
                  f"whitespace-separated tokens TOTAL. Use this exact count to deduce the "
                  f"structure (e.g. a header value then an N*N matrix); records may wrap "
                  f"across physical lines, so parse the flat token stream. =====")
        txt = full
        if len(full) > max_chars:
            head = full[:max_chars]
            ws = max(head.rfind(" "), head.rfind("\n"))
            if ws > 0:
                head = head[:ws]
            txt = head + "\n... [truncated — same format continues; the TOTAL token count above is exact]"
        blocks.append(header + "\n" + txt)
    return "\n\n".join(blocks)


def _find_instances(instances_dir: Path) -> list[str]:
    paths = sorted(str(p) for p in instances_dir.glob("*") if p.is_file())
    if not paths:  # instances may be nested in train/ test/ subdirs
        paths = sorted(str(p) for p in instances_dir.rglob("*") if p.is_file())
    return paths


def generate_contract(slug, nl_description, instances_dir, out_dir, llm_client,
                      example_slug="aircraft_landing", direction="min",
                      i_rep=3, smoke_time_limit_s=30.0, load_data_code=None):
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

    if load_data_code:
        logger.info("Stage Zero: Input Designer (I) — adopting user-supplied load_data (machine check, no reviewer)")
        adopt_input(ctx, load_data_code, val_paths)
    else:
        logger.info("Stage Zero: Input Designer (I) — generate + reviewer")
        design_input(ctx, llm_client, val_paths, example_slug, i_rep)
    # Feed the O/T/helper designers the REAL instance structure (actual keys +
    # shapes from a parsed sample) so they never guess wrong key names.
    derived = _derive_input_schema(ctx.load_data_code, smallest)
    if derived:
        ctx.input_schema = derived
        logger.info("Stage Zero: derived precise input structure from a parsed instance")
    logger.info("Stage Zero: Output Designer (O) — generate + reviewer")
    design_output(ctx, llm_client, example_slug, i_rep)
    logger.info("Stage Zero: Tool Designer (T core) — is_feasible + objective, validated by a heuristic")
    design_tools(ctx, llm_client, smallest, i_rep=i_rep)
    logger.info("Stage Zero: Helper Designer — a few helpers, validated by a heuristic that calls them")
    design_helpers(ctx, llm_client, smallest, i_rep=min(i_rep, 2))

    # Final self-check: assemble the full contract and verify it end to end with
    # a DETERMINISTIC witness — the O placeholder make_solution must be accepted
    # by is_feasible and scored by objective on the fully assembled spec
    # (I + O + T + helpers). The I/O/T/helper stages each already validated end to
    # end via a generated heuristic, so this is a cheap assembly sanity check, not
    # a gate that should discard an otherwise-verified contract on an unlucky LLM
    # sample. If the placeholder is not accepted we only warn (the placeholder may
    # simply be too naive) and still promote.
    tmp = Path(tempfile.mkdtemp()) / slug
    spec = build_spec(ctx, slug, str(tmp))
    if _placeholder_is_feasible(spec, smallest):
        logger.info("Stage Zero: final self-check passed (placeholder solution "
                    "feasible + scored on the full contract)")
    else:
        logger.warning("Stage Zero: final self-check inconclusive — the O placeholder "
                       "solution was not accepted by the assembled contract; promoting "
                       "anyway since I/O/T/helpers each passed their own validation")

    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(tmp, out)
    logger.info("Stage Zero complete -> %s", out)
    return str(out)
