import os
import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENROUTER_API_KEY"),
    reason="needs OPENROUTER_API_KEY; regenerates a real contract")


def test_regenerate_existing_problem_passes_gate(tmp_path):
    """Regenerate an EXISTING problem's contract from its DESCRIPTION using a
    DIFFERENT reference example, and confirm it passes the end-to-end gate.
    Reproducibility evidence for the paper's automation claim."""
    import sys
    sys.path.insert(0, ".")
    import problems.aircraft_landing.config as cfg
    from mace.contract import generate_contract
    from mace.evolution.llm_client import OpenRouterClient

    llm = OpenRouterClient(
        api_key=os.environ["OPENROUTER_API_KEY"],
        model=os.environ.get("MACE_TEST_MODEL", "google/gemini-3.1-flash-lite"))
    out = tmp_path / "regen"
    generate_contract(
        slug="aircraft_landing_regen", nl_description=cfg.DESCRIPTION,
        instances_dir="problems/aircraft_landing/instances", out_dir=str(out),
        llm_client=llm, required_keys=["num_planes", "planes", "separation"],
        example_slug="port_scheduling_problem",
    )
    assert (out / "spec.py").exists()  # gate already ran inside generate_contract
