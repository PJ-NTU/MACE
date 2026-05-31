from mace.contract.input_designer import design_input
from mace.contract.context import ContractContext

I_GEN = '''```python
# n: int, number of items
# weights: list[int]
DESCRIPTION = "pick items"
def load_data(path):
    return {"n": 3, "weights": [5, 2, 8]}
```'''


def test_design_input_fills_context(fake_llm, tmp_path):
    f = tmp_path / "inst1.txt"; f.write_text("3\n5 2 8")
    llm = fake_llm([I_GEN, "APPROVED"])   # generate + reviewer
    ctx = ContractContext(nl_description="pick items", sample_instance_text="raw")
    design_input(ctx, llm, instance_paths=[str(f)], example_slug="aircraft_landing")
    assert ctx.load_data_code is not None and "load_data" in ctx.load_data_code
    assert ctx.description == "pick items"
    assert "n" in (ctx.input_schema or "")
