"""Stage-Zero orchestrator: NL description + instances -> drop-in contract."""
from __future__ import annotations
import logging
import shutil
import tempfile
from pathlib import Path

from .context import ContractContext
from .input_designer import design_input
from .output_designer import design_output
from .tool_designer import design_tools
from .assemble import assemble_contract
from .validate import ContractGenerationError

logger = logging.getLogger(__name__)


def _read_sample(instance_paths, max_files: int = 2, max_chars: int = 16000) -> str:
    """Concatenate the full text of up to `max_files` instance files, each
    labelled with its filename, so the Input Designer sees the real byte layout
    (not just a truncated first file). Capped at `max_chars` total."""
    blocks = []
    for p in instance_paths[:max_files]:
        txt = Path(p).read_text(encoding="utf-8", errors="replace")
        blocks.append(f"# ===== file: {Path(p).name} =====\n{txt}")
    joined = "\n\n".join(blocks)
    return joined[:max_chars]


def generate_contract(slug, nl_description, instances_dir, out_dir, llm_client,
                      required_keys=None, example_slug="aircraft_landing",
                      direction="min", i_rep=3, smoke_time_limit_s=30.0):
    instances_dir = Path(instances_dir)
    instance_paths = sorted(str(p) for p in instances_dir.glob("*") if p.is_file())
    if not instance_paths:
        # Instance files may live in subdirectories (e.g. train/ test/). Recurse.
        instance_paths = sorted(str(p) for p in instances_dir.rglob("*") if p.is_file())
    if not instance_paths:
        raise ContractGenerationError(f"no instance files found in {instances_dir}")
    required_keys = required_keys or []

    ctx = ContractContext(nl_description=nl_description,
                          sample_instance_text=_read_sample(instance_paths))
    ctx.direction = direction

    logger.info("Stage Zero: Input Designer (I)")
    design_input(ctx, llm_client, instance_paths, required_keys, example_slug, i_rep)
    logger.info("Stage Zero: Output Designer (O)")
    design_output(ctx, llm_client, example_slug, i_rep)
    logger.info("Stage Zero: Tool Designer (T)")
    design_tools(ctx, llm_client, instance_paths[0], example_slug, i_rep)

    tmp = Path(tempfile.mkdtemp()) / slug
    assemble_contract(ctx, slug=slug, out_dir=str(tmp))
    _end_to_end_gate(tmp, instance_paths[0], smoke_time_limit_s)

    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(tmp, out)
    logger.info("Stage Zero complete -> %s", out)
    return str(out)


def _end_to_end_gate(contract_dir: Path, smoke_instance_path: str, time_limit_s: float):
    """Import the generated spec and run the placeholder solve through the
    existing framework smoke_test. Reuses Stage-Two machinery unchanged."""
    import importlib.util
    from mace.evolution.smoke_test import smoke_test

    spec_path = contract_dir / "spec.py"
    s = importlib.util.spec_from_file_location(f"_ctr_spec_{contract_dir.name}", spec_path)
    mod = importlib.util.module_from_spec(s)
    s.loader.exec_module(mod)
    spec_obj = mod.SPEC

    cfg_path_literal = repr(str(contract_dir / "config.py"))
    placeholder = (
        "def solve(instance, tools, time_limit_s):\n"
        "    import importlib.util\n"
        f"    cfgp = {cfg_path_literal}\n"
        "    sp = importlib.util.spec_from_file_location('_cfg_gate', cfgp)\n"
        "    cfg = importlib.util.module_from_spec(sp); sp.loader.exec_module(cfg)\n"
        "    return cfg.make_solution(instance)\n"
    )
    passed, err = smoke_test(placeholder, spec_obj, smoke_instance_path,
                             time_limit_s=time_limit_s)
    if not passed:
        raise ContractGenerationError(f"end-to-end gate failed: {err}")
