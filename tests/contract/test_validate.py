from mace.contract.validate import run_stage, smoke_input


def test_run_stage_passes_first_try(fake_llm):
    llm = fake_llm(["```python\nGOOD\n```"])
    def smoke(draft): return (draft == "GOOD", None if draft == "GOOD" else "bad")
    out = run_stage(llm, gen_prompt="g", reflect_prompt_fn=lambda d: None,
                    smoke_fn=smoke, i_rep=3)
    assert out == "GOOD"


def test_run_stage_repairs_then_passes(fake_llm):
    llm = fake_llm(["```python\nBAD\n```", "```python\nGOOD\n```"])
    def smoke(draft): return (draft == "GOOD", None if draft == "GOOD" else "still bad")
    out = run_stage(llm, gen_prompt="g", reflect_prompt_fn=lambda d: None,
                    smoke_fn=smoke, i_rep=3)
    assert out == "GOOD"


def test_run_stage_raises_after_budget(fake_llm):
    import pytest
    llm = fake_llm(["```python\nBAD\n```"] * 10)
    def smoke(draft): return (False, "nope")
    with pytest.raises(Exception):
        run_stage(llm, gen_prompt="g", reflect_prompt_fn=lambda d: None,
                  smoke_fn=smoke, i_rep=3)


GOOD_LOADER = '''
def load_data(path):
    return {"n": 3, "w": [1, 2, 3]}
'''


def test_smoke_input_ok(tmp_path):
    f = tmp_path / "inst1.txt"; f.write_text("ignored")
    ok, err = smoke_input(GOOD_LOADER, [str(f)], ["n", "w"])
    assert ok, err


def test_smoke_input_missing_key(tmp_path):
    f = tmp_path / "inst1.txt"; f.write_text("ignored")
    ok, err = smoke_input(GOOD_LOADER, [str(f)], ["n", "missing"])
    assert not ok and "missing" in err
