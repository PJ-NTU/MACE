from mace.contract.output_designer import design_output
from mace.contract.context import ContractContext

O_GEN = '''```python
OUTPUT_SCHEMA = "dict with key 'picked': list[int] of selected item indices"
def make_solution(instance):
    return {"picked": [0]}
```'''


def test_design_output_fills_context(fake_llm):
    llm = fake_llm([O_GEN, "APPROVED"])   # generate + reviewer
    ctx = ContractContext(nl_description="pick items", sample_instance_text="raw")
    ctx.input_schema = "n:int, weights:list[int]"
    ctx.load_data_code = "def load_data(p):\n    return {'n':3,'weights':[5,2,8]}"
    design_output(ctx, llm, example_slug="aircraft_landing")
    assert "picked" in ctx.output_schema
    assert "make_solution" in ctx.placeholder_solution_code
