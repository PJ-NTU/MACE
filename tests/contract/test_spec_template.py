from mace.contract.spec_template import render_spec


def test_render_native_spec_compiles():
    src = render_spec(slug="demo", direction="min", starter_code="def solve(): pass")
    assert "_DIRECTION = 'min'" in src
    assert "is_feasible" in src and "objective" in src
    assert "eval_func(" not in src and "_cfg.eval_func" not in src  # native: no eval_func call
    assert "class _Spec(ProblemSpec)" in src
    compile(src, "<spec>", "exec")          # must be valid Python
