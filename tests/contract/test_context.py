from mace.contract.context import ContractContext

def test_context_defaults_and_thread():
    ctx = ContractContext(nl_description="desc", sample_instance_text="raw")
    assert ctx.nl_description == "desc"
    assert ctx.direction == "min"
    assert ctx.load_data_code is None
    ctx.input_schema = "field: int"
    assert ctx.input_schema == "field: int"

def test_spec_template_fills_four_slots():
    from mace.contract.spec_template import render_spec
    src = render_spec(slug="demo_problem", direction="min")
    assert "_DIRECTION = 'min'" in src
    assert "cobench_demo_problem" in src
    assert "class _CoBenchSpec(ProblemSpec)" in src
    compile(src, "<spec>", "exec")  # must be valid Python
