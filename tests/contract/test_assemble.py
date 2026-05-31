import importlib.util

from mace.contract.assemble import assemble_contract
from mace.contract.context import ContractContext


def _ctx():
    ctx = ContractContext(nl_description="pick items, min weight", sample_instance_text="raw")
    ctx.load_data_code = ('# n: int\n# weights: list[int]\n'
                          'DESCRIPTION = "pick items"\n'
                          'def load_data(path):\n    return {"n": 3, "weights": [5, 2, 8]}')
    ctx.input_schema = "n:int, weights:list[int]"
    ctx.output_schema = "picked: list[int]"
    ctx.placeholder_solution_code = 'def make_solution(instance):\n    return {"picked": [0]}'
    ctx.is_feasible_code = ('def is_feasible(instance, solution):\n'
                            '    return (len(solution.get("picked", [])) >= 1), None')
    ctx.objective_code = ('def objective(instance, solution):\n'
                          '    return float(sum(instance["weights"][i] for i in solution["picked"]))')
    ctx.helpers_code = ('def lightest(instance):\n    """lightest item index."""\n'
                        '    w = instance["weights"]\n    return min(range(len(w)), key=lambda i: w[i])')
    ctx.helper_names = ["lightest"]
    ctx.tools_description = [{"name": "objective", "purpose": "min weight"}]
    return ctx


def _import(path, name):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


def test_assemble_native_contract(tmp_path):
    out = tmp_path / "knap"
    assemble_contract(_ctx(), slug="knap", out_dir=str(out))
    for f in ("config.py", "spec.py", "__init__.py"):
        assert (out / f).exists()
    assert not (out / "feasibility_steps.py").exists()  # native: no such file

    cfg = _import(out / "config.py", "knapcfg")
    assert hasattr(cfg, "load_data") and hasattr(cfg, "is_feasible")
    assert hasattr(cfg, "objective") and hasattr(cfg, "make_solution")
    assert not hasattr(cfg, "eval_func")   # no eval_func at all
    inst = cfg.load_data("x")
    assert cfg.is_feasible(inst, {"picked": [0]})[0] is True
    assert cfg.objective(inst, {"picked": [0]}) == 5.0
    assert cfg.lightest(inst) == 1

    spec = _import(out / "spec.py", "knapspec").SPEC
    tools = spec.tools(inst)
    assert {"is_feasible", "objective", "lightest"}.issubset(tools.keys())
    # instance is pre-bound: tools take only the solution / extra args
    assert tools["objective"]({"picked": [0]}) == 5.0
    assert tools["is_feasible"]({"picked": [0]}) == (True, None)
    assert tools["lightest"]() == 1
