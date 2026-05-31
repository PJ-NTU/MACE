from mace.contract.tool_designer import design_tools
from mace.contract.context import ContractContext

CANNED = '''```python
def eval_func(**kw):
    chosen = kw["chosen"]
    if len(set(chosen)) != len(chosen):
        raise ValueError("C1 duplicate item")
    total = sum(kw["items"][i] for i in chosen)
    if total > kw["cap"]:
        raise ValueError("C2 capacity exceeded")
    return float(kw["cap"] - total)
FEASIBILITY_STEPS_PY = "def is_feasible(solution):\\n    return True, None"
def infeasible_make_solution(instance):
    n = len(instance["items"])
    return {"chosen": list(range(n)) + list(range(n))}
```'''

def test_design_tools_fills_context(fake_llm, tmp_path):
    f = tmp_path / "i.txt"; f.write_text("x")
    llm = fake_llm([CANNED, CANNED])  # generate + reflect
    ctx = ContractContext(nl_description="knapsack", sample_instance_text="raw")
    ctx.load_data_code = "def load_data(p):\n    return {'cap':10,'items':[3,4,5]}"
    ctx.output_schema = "chosen: list[int]"
    ctx.placeholder_solution_code = 'def make_solution(instance):\n    return {"chosen": []}'
    design_tools(ctx, llm, instance_path=str(f), example_slug="aircraft_landing")
    assert "eval_func" in ctx.eval_func_code
    assert "is_feasible" in ctx.feasibility_steps
