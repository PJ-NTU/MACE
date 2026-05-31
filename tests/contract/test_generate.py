from mace.contract.generate import generate_contract

I_RESP = '''```python
# cap: int
# items: list[int]
DESCRIPTION = "0/1 knapsack: pick items without exceeding capacity"
def load_data(path):
    with open(path) as f:
        cap, *items = [int(x) for x in f.read().split()]
    return {"cap": cap, "items": items}
```'''
O_RESP = '''```python
OUTPUT_SCHEMA = "dict with key 'chosen': list[int] of selected item indices"
def make_solution(instance):
    return {"chosen": []}
```'''
T_RESP = '''```python
def eval_func(**kw):
    chosen = kw["chosen"]
    if len(set(chosen)) != len(chosen):
        raise ValueError("duplicate item")
    total = sum(kw["items"][i] for i in chosen)
    if total > kw["cap"]:
        raise ValueError("capacity exceeded")
    return float(kw["cap"] - total)
FEASIBILITY_STEPS_PY = "def is_feasible(solution):\\n    return True, None"
def infeasible_make_solution(instance):
    n = len(instance["items"])
    return {"chosen": list(range(n)) + list(range(n))}
```'''

def test_generate_contract_end_to_end(fake_llm, tmp_path):
    inst = tmp_path / "instances"; inst.mkdir()
    (inst / "k1.txt").write_text("10 3 4 5 6")
    out = tmp_path / "problems" / "knap"
    # each designer calls LLM twice (generate + reflect)
    llm = fake_llm([I_RESP, I_RESP, O_RESP, O_RESP, T_RESP, T_RESP])
    result_path = generate_contract(
        slug="knap", nl_description="0/1 knapsack",
        instances_dir=str(inst), out_dir=str(out),
        llm_client=llm, required_keys=["cap", "items"],
        example_slug="aircraft_landing",
    )
    assert (out / "spec.py").exists()
    assert (out / "config.py").exists()
    assert (out / "feasibility_steps.py").exists()
