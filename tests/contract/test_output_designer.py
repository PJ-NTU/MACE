from mace.contract.output_designer import design_output
from mace.contract.context import ContractContext

CANNED = '''```python
OUTPUT_SCHEMA = "dict with key 'chosen': list[int]"
def make_solution(instance):
    return {"chosen": []}
```'''

def test_design_output_fills_context(fake_llm):
    llm = fake_llm([CANNED])  # slim: one generation call, then smoke
    ctx = ContractContext(nl_description="knapsack", sample_instance_text="raw")
    ctx.input_schema = "cap:int, items:list"
    ctx.load_data_code = "def load_data(p):\n    return {'cap':10,'items':[3,4]}"
    design_output(ctx, llm, example_slug="aircraft_landing")
    assert "chosen" in ctx.output_schema
    assert "make_solution" in ctx.placeholder_solution_code
