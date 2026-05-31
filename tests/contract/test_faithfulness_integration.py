import os
import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENROUTER_API_KEY"),
    reason="needs OPENROUTER_API_KEY; regenerates a real contract via live LLM")


def test_regenerate_problem_passes_gate(tmp_path):
    """Regenerate a problem's contract from its natural-language DESCRIPTION using
    a DIFFERENT reference example, and confirm the full I->O->T->helpers pipeline
    plus the heuristic gate passes. Reproducibility evidence for the paper's
    'the contract is constructed automatically by an LLM' claim."""
    import sys
    sys.path.insert(0, ".")
    import problems.travelling_salesman_problem.config as cfg
    from mace.contract import generate_contract
    from mace.evolution.llm_client import OpenRouterClient

    llm = OpenRouterClient(
        api_key=os.environ["OPENROUTER_API_KEY"],
        model=os.environ.get("MACE_TEST_MODEL", "google/gemini-2.5-flash"),
        max_tokens=16384)  # eval/tool code can be long
    out = tmp_path / "regen"
    generate_contract(
        slug="tsp_regen", nl_description=cfg.DESCRIPTION,
        instances_dir="problems/travelling_salesman_problem/instances",
        out_dir=str(out), llm_client=llm, example_slug="aircraft_landing")
    assert (out / "spec.py").exists()      # gate already ran inside generate_contract
    assert (out / "config.py").exists()
