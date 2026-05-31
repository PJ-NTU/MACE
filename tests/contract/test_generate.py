from mace.contract.generate import generate_contract

I_GEN = '''```python
# n: int, number of items
# weights: list[int]
DESCRIPTION = "Pick at least one item to minimise total selected weight."
def load_data(path):
    return {"n": 3, "weights": [5, 2, 8]}
```'''
O_GEN = '''```python
OUTPUT_SCHEMA = "dict with key 'picked': list[int]"
def make_solution(instance):
    return {"picked": [0]}
```'''
T_GEN = '''```python
def is_feasible(instance, solution):
    if len(solution.get("picked", [])) < 1:
        return False, "C1: need at least one"
    return True, None
def objective(instance, solution):
    return float(sum(instance["weights"][i] for i in solution["picked"]))
```'''
HELPER_PLAN = '''```python
HELPERS_PLAN = [dict(name="lightest_item", purpose="Index of the lightest item.")]
```'''
HELPER_IMPL = '''```python
def lightest_item(instance):
    """Index of the lightest item."""
    w = instance["weights"]
    return min(range(len(w)), key=lambda i: w[i])
```'''
HEUR = '''```python
def solve(instance, tools, time_limit_s):
    return {"picked": [0]}
```'''
# helper validation requires the heuristic to actually CALL the helper tool
HEUR_USES_HELPER = '''```python
def solve(instance, tools, time_limit_s):
    i = tools['lightest_item']()
    return {"picked": [i]}
```'''


def test_generate_contract_end_to_end(fake_llm, tmp_path):
    inst = tmp_path / "instances"; inst.mkdir()
    (inst / "k1.txt").write_text("3\n5 2 8")
    out = tmp_path / "problems" / "knap"
    # I(gen+review) O(gen+review) T(gen; passes via feasible make_solution fast path)
    # helper(plan+impl+heuristic) final(heuristic)
    llm = fake_llm([I_GEN, "APPROVED", O_GEN, "APPROVED",
                    T_GEN, HELPER_PLAN, HELPER_IMPL, HEUR_USES_HELPER, HEUR])
    generate_contract(slug="knap", nl_description="pick items, min weight",
                      instances_dir=str(inst), out_dir=str(out),
                      llm_client=llm, example_slug="aircraft_landing")
    assert (out / "spec.py").exists()
    assert (out / "config.py").exists()
    assert not (out / "feasibility_steps.py").exists()  # native: no eval_func / steps

    import importlib.util
    s = importlib.util.spec_from_file_location("knapcfg2", out / "config.py")
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    assert not hasattr(m, "eval_func")
    assert hasattr(m, "is_feasible") and hasattr(m, "objective")
