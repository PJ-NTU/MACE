from mace.contract.assemble import assemble_contract
from mace.contract.context import ContractContext

def _full_ctx():
    ctx = ContractContext(nl_description="knapsack", sample_instance_text="raw")
    ctx.description = "knapsack"
    ctx.load_data_code = 'def load_data(path):\n    return {"cap": 10, "items": [3, 4, 5]}'
    ctx.input_schema = "cap:int, items:list[int]"
    ctx.output_schema = "chosen: list[int]"
    ctx.placeholder_solution_code = 'def make_solution(instance):\n    return {"chosen": []}'
    ctx.eval_func_code = ('def eval_func(**kw):\n'
                          '    t = sum(kw["items"][i] for i in kw["chosen"])\n'
                          '    if t > kw["cap"]:\n        raise ValueError("cap")\n'
                          '    return float(kw["cap"] - t)')
    ctx.feasibility_steps = 'def is_feasible(solution):\n    return True, None'
    ctx.direction = "min"
    return ctx

def test_assemble_writes_importable_contract(tmp_path):
    out = tmp_path / "knap"
    assemble_contract(_full_ctx(), slug="knap", out_dir=str(out))
    for f in ("config.py", "feasibility_steps.py", "spec.py", "__init__.py"):
        assert (out / f).exists()
    import importlib.util
    spec = importlib.util.spec_from_file_location("knapcfg", out / "config.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    assert hasattr(mod, "load_data") and hasattr(mod, "eval_func")
    assert hasattr(mod, "make_solution")
    assert mod.eval_func(cap=10, items=[3, 4, 5], chosen=[0]) == 7.0
    # feasibility_steps module exposes the string
    fspec = importlib.util.spec_from_file_location("knapfs", out / "feasibility_steps.py")
    fmod = importlib.util.module_from_spec(fspec); fspec.loader.exec_module(fmod)
    assert "is_feasible" in fmod.FEASIBILITY_STEPS_PY
