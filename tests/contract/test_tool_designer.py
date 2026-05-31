from mace.contract.tool_designer import design_tools
from mace.contract.context import ContractContext

# T core: is_feasible + objective (no eval_func)
T_GEN = '''```python
def is_feasible(instance, solution):
    picked = solution.get("picked", [])
    if len(picked) < 1:
        return False, "C1: must pick at least one item"
    return True, None

def objective(instance, solution):
    return float(sum(instance["weights"][i] for i in solution["picked"]))
```'''

# the validating heuristic that runs through I/O/T
HEUR = '''```python
def solve(instance, tools, time_limit_s):
    return {"picked": [0]}
```'''


def test_design_tools_validated_by_heuristic(fake_llm, tmp_path):
    f = tmp_path / "i.txt"; f.write_text("3\n5 2 8")
    llm = fake_llm([T_GEN, HEUR])   # T gen + one heuristic that solves through I/O/T
    ctx = ContractContext(nl_description="pick items, min weight", sample_instance_text="raw")
    ctx.load_data_code = 'def load_data(path):\n    return {"n": 3, "weights": [5, 2, 8]}'
    ctx.output_schema = "picked: list[int]"
    ctx.placeholder_solution_code = 'def make_solution(instance):\n    return {"picked": [0]}'
    design_tools(ctx, llm, instance_path=str(f))
    assert "is_feasible" in ctx.is_feasible_code
    assert "objective" in ctx.objective_code
    assert any(t.get("name") == "objective" for t in ctx.tools_description)
