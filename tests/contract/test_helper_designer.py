from mace.contract.helper_designer import design_helpers
from mace.contract.context import ContractContext


def _ctx():
    ctx = ContractContext(nl_description="pick items, min weight", sample_instance_text="raw")
    ctx.load_data_code = 'def load_data(path):\n    return {"n": 3, "weights": [5, 2, 8]}'
    ctx.input_schema = "n:int, weights:list[int]"
    ctx.output_schema = "picked: list[int]"
    ctx.placeholder_solution_code = 'def make_solution(instance):\n    return {"picked": [0]}'
    ctx.is_feasible_code = ('def is_feasible(instance, solution):\n'
                            '    return (len(solution.get("picked", [])) >= 1), None')
    ctx.objective_code = ('def objective(instance, solution):\n'
                          '    return float(sum(instance["weights"][i] for i in solution["picked"]))')
    ctx.tools_description = [{"name": "objective", "purpose": "min weight"}]
    return ctx


PLAN_GOOD = '''```python
HELPERS_PLAN = [dict(name="lightest_item", purpose="Index of the lightest item.")]
```'''
IMPL_LIGHTEST = '''```python
def lightest_item(instance):
    """Index of the lightest item."""
    w = instance["weights"]
    return min(range(len(w)), key=lambda i: w[i])
```'''
HEUR_USES = '''```python
def solve(instance, tools, time_limit_s):
    return {"picked": [tools['lightest_item']()]}
```'''


def test_good_helper_accepted(fake_llm, tmp_path):
    f = tmp_path / "i.txt"; f.write_text("3\n5 2 8")
    # phase1 plan + phase2 implement + one validating heuristic
    llm = fake_llm([PLAN_GOOD, IMPL_LIGHTEST, HEUR_USES])
    ctx = _ctx()
    design_helpers(ctx, llm, instance_path=str(f), i_rep=1)
    assert ctx.helper_names == ["lightest_item"]
    assert "lightest_item" in ctx.helpers_code


PLAN_BAD = '''```python
HELPERS_PLAN = [dict(name="bad_helper", purpose="always raises")]
```'''
IMPL_BAD = '''```python
def bad_helper(instance):
    """Always raises."""
    raise ValueError("boom")
```'''
HEUR_CALLS_BAD = '''```python
def solve(instance, tools, time_limit_s):
    tools['bad_helper']()
    return {"picked": [0]}
```'''


def test_broken_helper_discarded_not_fatal(fake_llm, tmp_path):
    f = tmp_path / "i.txt"; f.write_text("3\n5 2 8")
    # plan + impl + attempt0(2 heuristics) + repair + attempt1(2 heuristics) = 7 calls
    llm = fake_llm([PLAN_BAD, IMPL_BAD, HEUR_CALLS_BAD, HEUR_CALLS_BAD,
                    IMPL_BAD, HEUR_CALLS_BAD, HEUR_CALLS_BAD])
    ctx = _ctx()
    design_helpers(ctx, llm, instance_path=str(f), i_rep=1)
    # broken helper DISCARDED; pipeline does NOT raise; helpers empty
    assert ctx.helper_names == []
    assert ctx.helpers_code is None
