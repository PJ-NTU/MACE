from mace.contract.context import ContractContext


def test_context_defaults_and_thread():
    ctx = ContractContext(nl_description="desc", sample_instance_text="raw")
    assert ctx.nl_description == "desc"
    assert ctx.direction == "min"
    assert ctx.is_feasible_code is None
    assert ctx.objective_code is None
    assert ctx.helper_names is None
    ctx.objective_code = "def objective(i, s): return 0.0"
    assert ctx.objective_code.startswith("def objective")
