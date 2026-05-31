"""Helper Designer (two-phase):

  Phase 1 (plan): ONE LLM call lists a small, non-overlapping set of helper tools
    (names + purposes only). Planning the whole set at once avoids duplicates that
    independent per-helper generations would produce (the model has no memory).
  Phase 2 (implement + validate each): for each planned helper, generate its code,
    then validate it with a heuristic that is REQUIRED to call it (instrumented to
    confirm invocation). If it cannot be made to work within the repair budget, the
    helper is DISCARDED — helpers are optional and never fail the pipeline.

Contrast with the T core (is_feasible / objective), which is MANDATORY."""
from __future__ import annotations
import ast
import logging
import tempfile
from dataclasses import replace
from pathlib import Path

from mace.evolution.operators._common import extract_python
from .assemble import build_spec
from .heuristic_check import heuristic_passes

logger = logging.getLogger(__name__)

_PLAN_PROMPT = (Path(__file__).parent / "prompts" / "helper_designer.md").read_text(encoding="utf-8")
_IMPL_PROMPT = (Path(__file__).parent / "prompts" / "helper_impl.md").read_text(encoding="utf-8")


def _func_src(src: str, name: str) -> str | None:
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(src, node)
    return None


def _entry_to_dict(node):
    """Parse one plan entry, accepting either a {..} dict literal or a dict(..) call."""
    if isinstance(node, ast.Dict):
        d = {}
        for k, v in zip(node.keys, node.values):
            if isinstance(k, ast.Constant) and isinstance(v, ast.Constant):
                d[k.value] = v.value
        return d
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "dict":
        return {kw.arg: kw.value.value for kw in node.keywords
                if isinstance(kw.value, ast.Constant)}
    return {}


def _plan_helpers(llm_client, ctx):
    """Phase 1: return [(name, purpose)] — at most 3, non-overlapping."""
    prompt = _PLAN_PROMPT.format(nl=ctx.nl_description, input_schema=ctx.input_schema or "",
                                 output_schema=ctx.output_schema or "")
    code = extract_python(llm_client.chat(prompt))
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "HELPERS_PLAN" for t in node.targets) \
                and isinstance(node.value, (ast.List, ast.Tuple)):
            out = []
            for elt in node.value.elts[:3]:
                d = _entry_to_dict(elt)
                name = d.get("name")
                if isinstance(name, str) and name.isidentifier():
                    out.append((name, str(d.get("purpose", ""))))
            return out
    return []


def _implement_helper(llm_client, ctx, name, purpose) -> str:
    prompt = _IMPL_PROMPT.format(
        nl=ctx.nl_description, input_schema=ctx.input_schema or "",
        output_schema=ctx.output_schema or "", is_feasible=ctx.is_feasible_code or "",
        objective=ctx.objective_code or "", name=name, purpose=purpose)
    code = extract_python(llm_client.chat(prompt))
    return _func_src(code, name) or code


def _repair_helper(llm_client, name, src, err) -> str:
    body = (
        f"# Broken helper `{name}`\n```python\n{src}\n```\n\n"
        f"# Failure report\n```\n{err}\n```\n\n"
        f"Fix this helper so a solver can call `tools['{name}'](...)` and it runs "
        f"correctly. Keep the signature `def {name}(instance, ...)` (instance first) "
        f"and keep any imports inside the function. Output ONLY the corrected function "
        f"in one fenced ```python block."
    )
    code = extract_python(llm_client.chat(body))
    return _func_src(code, name) or code


def _validate_one_helper(spec, cfg_module, name, llm_client, instance_path, tries=2):
    """A heuristic must CALL tools['name'] and run feasibly. The helper is
    instrumented (call counter) to confirm it was actually invoked."""
    orig = getattr(cfg_module, name)
    calls = {"n": 0}

    def wrapped(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    setattr(cfg_module, name, wrapped)
    try:
        hint = (f"# REQUIREMENT: your solve MUST call `tools['{name}'](...)` at "
                f"least once — exercising that tool is the whole point of this run.")
        err = "no heuristic generated"
        for _ in range(tries):
            calls["n"] = 0
            ok, err, _code = heuristic_passes(spec, llm_client, instance_path,
                                              hint=hint, tries=1)
            if ok and calls["n"] > 0:
                return True, None
            if ok and calls["n"] == 0:
                err = f"heuristic ran but never called tools['{name}']"
        return False, err
    finally:
        setattr(cfg_module, name, orig)


def design_helpers(ctx, llm_client, instance_path, i_rep: int = 3):
    base_tools = list(ctx.tools_description or [])

    def _tools_desc(pairs):
        return base_tools + [
            {"name": n, "input": "solution/partial args (instance is bound)",
             "output": "...", "purpose": p} for n, p in pairs
        ]

    plan = _plan_helpers(llm_client, ctx)
    purpose_of = dict(plan)
    accepted: dict[str, str] = {}  # name -> validated source

    for name, purpose in plan:
        try:
            src = _implement_helper(llm_client, ctx, name, purpose)
        except Exception as e:
            logger.info("Helper Designer: DISCARDING '%s' (implement raised: %s)", name, e)
            continue
        ok, err = False, "not validated"
        for attempt in range(i_rep + 1):
            trial_sources = {**accepted, name: src}
            trial = replace(
                ctx, helpers_code="\n\n".join(trial_sources.values()),
                helper_names=list(trial_sources.keys()),
                tools_description=_tools_desc([(n, purpose_of.get(n, "")) for n in trial_sources]))
            try:
                tmp = Path(tempfile.mkdtemp()) / "helper_trial"
                spec = build_spec(trial, "helper_trial", str(tmp))
                cfg_module = spec._cfg_module
            except Exception as e:
                err = f"assemble/import failed: {type(e).__name__}: {e}"
            else:
                ok, err = _validate_one_helper(spec, cfg_module, name, llm_client, instance_path)
                if ok:
                    break
            if attempt < i_rep:
                try:
                    src = _repair_helper(llm_client, name, src, err)
                except Exception as e:
                    err = f"repair raised: {type(e).__name__}: {e}"
        if ok:
            accepted[name] = src
            logger.info("Helper Designer: helper '%s' accepted", name)
        else:
            logger.info("Helper Designer: DISCARDING helper '%s' after %d repairs: %s",
                        name, i_rep, err)

    if accepted:
        ctx.helpers_code = "\n\n".join(accepted.values())
        ctx.helper_names = list(accepted.keys())
        ctx.tools_description = _tools_desc([(n, purpose_of.get(n, "")) for n in accepted])
    else:
        ctx.helpers_code, ctx.helper_names, ctx.tools_description = None, [], base_tools
    return ctx
