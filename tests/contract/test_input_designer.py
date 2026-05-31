from mace.contract.input_designer import design_input
from mace.contract.context import ContractContext

CANNED = '''```python
# cap: int
# items: list[int]
DESCRIPTION = "knapsack-like"
def load_data(path):
    return {"cap": 10, "items": [3, 4, 5]}
```'''

def test_design_input_fills_context(fake_llm, tmp_path):
    f = tmp_path / "inst1.txt"; f.write_text("raw bytes")
    llm = fake_llm([CANNED])  # slim: one generation call, then smoke
    ctx = ContractContext(nl_description="a knapsack", sample_instance_text="raw bytes")
    design_input(ctx, llm, instance_paths=[str(f)],
                 required_keys=["cap", "items"], example_slug="aircraft_landing")
    assert ctx.load_data_code is not None
    assert "load_data" in ctx.load_data_code
    assert ctx.description == "knapsack-like"
    assert "cap" in (ctx.input_schema or "")
