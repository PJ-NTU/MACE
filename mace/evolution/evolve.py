"""Stage Two — main evolution loop.

Pipeline (one iteration):
  1. Compute rank matrix R from cost matrix F (M training instances × N heuristics).
  2. Generate N new candidates via uniform sampling of O1..O5; each candidate
     is smoke-tested, repaired via O6 on failure (up to I_rep), then evaluated
     on the M training instances. If timeouts occur, O7 repair (up to I_eff)
     is attempted. Candidates that still fail smoke after repair are discarded
     and resampled.
  3. MILP-select N heuristics from the 2N candidate pool (Π ∪ new) using the
     rank-based objective from STAGE_TWO_DESIGN §5.
  4. Log per-iteration CPI, operator usage, and repair invocation counts.
"""
from __future__ import annotations
import json
import logging
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional
import numpy as np

from .cpi import compute_cpi, PENALTY
from .rank_matrix import compute_rank_matrix
from .milp_selection import milp_select
from .smoke_test import smoke_test
from .operators import (
    GENERATION_OPERATORS,
    o6_error_repair,
    o7_efficiency_repair,
)
from mace.framework import run_solve

logger = logging.getLogger(__name__)


# ---------- single-candidate evaluation ----------

def _exec_solve(code: str):
    ns: dict = {}
    exec(compile(code, "<candidate>", "exec"), ns)
    fn = ns.get("solve")
    if fn is None or not callable(fn):
        raise RuntimeError("no callable `solve` defined")
    return fn


def evaluate_one(
    spec,
    code: str,
    instance_path: str,
    T_max: float,
    timeout_slack: float = 1.5,
) -> tuple[float, dict]:
    """Return (cost_or_PENALTY, info_dict).

    Status values: 'ok', 'exec_error', 'load_error', 'infeasible', 'timeout'.
    A solve that overshoots T_max by more than `timeout_slack` is flagged as
    'timeout' (since the framework does not hard-kill).
    """
    try:
        solve_fn = _exec_solve(code)
    except Exception as e:
        return PENALTY, {"status": "exec_error", "msg": f"{type(e).__name__}: {e}"}

    try:
        instance = spec.load_data(instance_path)
    except Exception as e:
        return PENALTY, {"status": "load_error", "msg": f"{type(e).__name__}: {e}"}

    result = run_solve(spec, instance, solve_fn, time_limit_s=T_max)

    if not result.feasible:
        return PENALTY, {
            "status": "infeasible",
            "msg": result.error_msg,
            "elapsed": result.elapsed_s,
        }
    if result.elapsed_s > T_max * timeout_slack:
        return PENALTY, {
            "status": "timeout",
            "elapsed": result.elapsed_s,
            "instance_path": instance_path,
        }
    return float(result.objective), {"status": "ok", "elapsed": result.elapsed_s}


def evaluate_one_subprocess(
    spec_module_path: str,
    code: str,
    instance_path: str,
    T_max: float,
    hard_kill_slack: float = 2.0,
) -> tuple[float, dict]:
    """Run evaluate_one in a fresh subprocess; hard-kill on T_max * slack.

    Guarantees wall-clock <= T_max * hard_kill_slack + small overhead.
    Cost of subprocess startup: ~500ms on Windows. Use when the heuristic
    code may not honor `time_limit_s`.
    """
    payload = json.dumps({
        "spec_module_path": spec_module_path,
        "code": code,
        "instance_path": instance_path,
        "T_max": T_max,
    })
    timeout = max(5.0, T_max * hard_kill_slack)
    cmd = [sys.executable, "-m", "mace.evolution._eval_worker_subprocess"]
    try:
        result = subprocess.run(
            cmd, input=payload, capture_output=True, text=True,
            timeout=timeout, encoding="utf-8", errors="replace",
            cwd=str(Path(__file__).resolve().parents[2]),
        )
    except subprocess.TimeoutExpired:
        return PENALTY, {"status": "hard_timeout", "msg": f">{timeout:.0f}s"}
    except Exception as e:
        return PENALTY, {"status": "subprocess_launch_error", "msg": f"{type(e).__name__}: {e}"}

    if result.returncode != 0:
        return PENALTY, {
            "status": "subprocess_exit",
            "code": result.returncode,
            "stderr": (result.stderr or "")[:500],
        }

    try:
        # Take the last non-empty stdout line in case worker emits other output first.
        lines = [ln for ln in (result.stdout or "").splitlines() if ln.strip()]
        if not lines:
            return PENALTY, {"status": "empty_stdout", "stderr": (result.stderr or "")[:500]}
        data = json.loads(lines[-1])
        return float(data["cost"]), data["info"]
    except Exception as e:
        return PENALTY, {
            "status": "stdout_parse_error",
            "msg": f"{type(e).__name__}: {e}",
            "stdout": (result.stdout or "")[:500],
        }


def evaluate_code_on_instances(
    spec,
    code: str,
    instance_paths: list[str],
    T_max: float,
    spec_module_path: Optional[str] = None,
    n_workers: int = 1,
    hard_kill_slack: float = 1.5,
    use_subprocess: bool = False,
    remote_eval_cfg=None,
) -> tuple[np.ndarray, list[dict]]:
    """Evaluate one code on all instances.

    Dispatch priority:
      0. remote_eval_cfg set                -> dispatch to remote host (batch SSH)
      1. use_subprocess + spec_module_path  -> per-instance subprocess with hard-kill
      2. n_workers > 1 + spec_module_path   -> ProcessPoolExecutor
      3. fallback                           -> in-process serial
    """
    if remote_eval_cfg is not None and spec_module_path:
        from .remote_eval import remote_evaluate_code_on_instances
        return remote_evaluate_code_on_instances(
            remote_eval_cfg, spec_module_path, code, instance_paths, T_max,
        )
    if use_subprocess and spec_module_path:
        M = len(instance_paths)
        perf = np.full(M, PENALTY, dtype=np.float64)
        infos: list[dict] = []
        for i, p in enumerate(instance_paths):
            cost, info = evaluate_one_subprocess(
                spec_module_path, code, p, T_max,
                hard_kill_slack=max(hard_kill_slack, 2.0),
            )
            perf[i] = cost
            infos.append(info)
        return perf, infos
    if n_workers > 1 and spec_module_path:
        from .parallel_eval import parallel_evaluate_code
        return parallel_evaluate_code(
            spec_module_path, code, instance_paths, T_max,
            n_workers=n_workers, hard_kill_slack=hard_kill_slack,
        )
    M = len(instance_paths)
    perf = np.full(M, PENALTY, dtype=np.float64)
    infos: list[dict] = []
    for i, p in enumerate(instance_paths):
        cost, info = evaluate_one(spec, code, p, T_max)
        perf[i] = cost
        infos.append(info)
    return perf, infos


def evaluate_portfolio(
    spec,
    codes: list[str],
    instance_paths: list[str],
    T_max: float,
    spec_module_path: Optional[str] = None,
    n_workers: int = 1,
    hard_kill_slack: float = 1.5,
    use_subprocess: bool = False,
    remote_eval_cfg=None,
) -> tuple[np.ndarray, list[list[dict]]]:
    """Returns F (M x N) and infos[i_inst][h] for each cell.

    Dispatch priority:
      1. use_subprocess + spec_module_path -> per-cell subprocess with hard-kill
      2. n_workers > 1 + spec_module_path  -> ProcessPoolExecutor
      3. fallback                           -> in-process serial
    """
    # Remote-eval path: one SSH batch per code (each batch parallelises on remote)
    if remote_eval_cfg is not None and spec_module_path:
        M, N = len(instance_paths), len(codes)
        F = np.full((M, N), PENALTY, dtype=np.float64)
        infos: list[list[dict]] = [[{} for _ in range(N)] for _ in range(M)]
        for h, code in enumerate(codes):
            perf_vec, info_vec = evaluate_code_on_instances(
                spec, code, instance_paths, T_max,
                spec_module_path=spec_module_path,
                remote_eval_cfg=remote_eval_cfg,
            )
            F[:, h] = perf_vec
            for i, info in enumerate(info_vec):
                infos[i][h] = info
        return F, infos
    if use_subprocess and spec_module_path:
        M, N = len(instance_paths), len(codes)
        F = np.full((M, N), PENALTY, dtype=np.float64)
        infos: list[list[dict]] = [[{} for _ in range(N)] for _ in range(M)]
        for h, code in enumerate(codes):
            perf_vec, info_vec = evaluate_code_on_instances(
                spec, code, instance_paths, T_max,
                spec_module_path=spec_module_path, use_subprocess=True,
                hard_kill_slack=hard_kill_slack,
            )
            F[:, h] = perf_vec
            for i, info in enumerate(info_vec):
                infos[i][h] = info
        return F, infos
    if n_workers > 1 and spec_module_path:
        from .parallel_eval import parallel_evaluate_portfolio
        return parallel_evaluate_portfolio(
            spec_module_path, codes, instance_paths, T_max,
            n_workers=n_workers, hard_kill_slack=hard_kill_slack,
        )
    M, N = len(instance_paths), len(codes)
    F = np.full((M, N), PENALTY, dtype=np.float64)
    infos: list[list[dict]] = [[{} for _ in range(N)] for _ in range(M)]
    for h, code in enumerate(codes):
        perf_vec, info_vec = evaluate_code_on_instances(spec, code, instance_paths, T_max)
        F[:, h] = perf_vec
        for i, info in enumerate(info_vec):
            infos[i][h] = info
    return F, infos


# ---------- run a new candidate with O6/O7 repair ----------

def run_with_repair(
    spec,
    code: str,
    instance_paths: list[str],
    T_max: float,
    I_rep: int,
    I_eff: int,
    llm_client,
    smoke_time_limit_s: float = 30.0,
    spec_module_path: Optional[str] = None,
    n_workers: int = 1,
    hard_kill_slack: float = 1.5,
    use_subprocess: bool = False,
    remote_eval_cfg=None,
) -> tuple[bool, str, np.ndarray, dict]:
    """Evaluate `code` on training instances, repairing via O6/O7 on failure.

    Returns (ok, final_code, perf_vec, stats_dict). `ok=False` means the
    candidate could not be made to pass smoke test even after repairs.

    Even when ok=True, perf_vec may contain PENALTY entries for some instances
    where the candidate timed out or threw — those are kept as PENALTY (the
    MILP / rank will simply not pick this heuristic for those instances).
    """
    stats = {"o6_calls": 0, "o7_calls": 0, "smoke_attempts": 1}

    # 1) smoke test (small instance) -> O6 loop if needed
    smoke_path = instance_paths[0]
    passed, err = smoke_test(code, spec, smoke_path, smoke_time_limit_s,
                              spec_module_path=spec_module_path,
                              use_subprocess=use_subprocess,
                              hard_kill_slack=hard_kill_slack)
    while not passed and stats["o6_calls"] < I_rep:
        stats["o6_calls"] += 1
        try:
            code = o6_error_repair.repair(spec, code, err, llm_client)
        except Exception as e:
            return False, code, np.array([]), {**stats, "failed_at": f"o6_raised: {e}"}
        stats["smoke_attempts"] += 1
        passed, err = smoke_test(code, spec, smoke_path, smoke_time_limit_s,
                              spec_module_path=spec_module_path,
                              use_subprocess=use_subprocess,
                              hard_kill_slack=hard_kill_slack)
    if not passed:
        return False, code, np.array([]), {**stats, "failed_at": f"smoke_failed: {err}"}

    # 2) full evaluation
    perf_vec, infos = evaluate_code_on_instances(
        spec, code, instance_paths, T_max,
        spec_module_path=spec_module_path, n_workers=n_workers,
        hard_kill_slack=hard_kill_slack, use_subprocess=use_subprocess,
        remote_eval_cfg=remote_eval_cfg,
    )

    # 3) if there are timeouts, try O7 repair
    for round_idx in range(I_eff):
        timeouts = [(i, info) for i, info in enumerate(infos) if info["status"] == "timeout"]
        if not timeouts:
            break
        if len(timeouts) < max(1, len(infos) // 10):
            # only a few timeouts — accept as-is, no repair
            break
        worst = max(timeouts, key=lambda t: t[1].get("elapsed", 0.0))
        i_inst, info = worst
        stats["o7_calls"] += 1
        try:
            code = o7_efficiency_repair.repair(
                spec, code,
                instance_info=str(instance_paths[i_inst]),
                elapsed_s=info["elapsed"],
                time_limit_s=T_max,
                llm_client=llm_client,
            )
        except Exception as e:
            stats["o7_last_error"] = str(e)
            break

        # re-smoke after O7
        passed, err = smoke_test(code, spec, smoke_path, smoke_time_limit_s,
                              spec_module_path=spec_module_path,
                              use_subprocess=use_subprocess,
                              hard_kill_slack=hard_kill_slack)
        stats["smoke_attempts"] += 1
        if not passed:
            # O7 broke the code — try O6 once to recover
            stats["o6_calls"] += 1
            try:
                code = o6_error_repair.repair(spec, code, err, llm_client)
            except Exception as e:
                return False, code, np.array([]), {**stats, "failed_at": f"o6_after_o7_raised: {e}"}
            passed, err = smoke_test(code, spec, smoke_path, smoke_time_limit_s,
                              spec_module_path=spec_module_path,
                              use_subprocess=use_subprocess,
                              hard_kill_slack=hard_kill_slack)
            stats["smoke_attempts"] += 1
            if not passed:
                return False, code, np.array([]), {**stats, "failed_at": f"smoke_after_o7: {err}"}

        # re-evaluate
        perf_vec, infos = evaluate_code_on_instances(
            spec, code, instance_paths, T_max,
            spec_module_path=spec_module_path, n_workers=n_workers,
            hard_kill_slack=hard_kill_slack,
            remote_eval_cfg=remote_eval_cfg,
        )

    return True, code, perf_vec, stats


# ---------- iteration logging ----------

def _log_iteration(output_dir: Optional[Path], iteration: int, payload: dict):
    if output_dir is None:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    fname = output_dir / f"iter_{iteration:04d}.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


# ---------- main evolution loop ----------

def _generate_candidates_parallel(
    spec, portfolio, F, R, llm_client, op_choices: list[int], llm_concurrency: int,
):
    """Run len(op_choices) generation operators concurrently (LLM is I/O bound).
    Returns a list aligned with op_choices: each entry is
    (code_or_None, meta_dict, op_idx, error_msg_or_None).
    """
    def _gen(op_idx: int):
        op = GENERATION_OPERATORS[op_idx]
        try:
            code, meta = op.generate(spec, portfolio, F, R, llm_client)
            return (code, meta, op_idx, None)
        except Exception as e:
            return (None, {}, op_idx, f"{type(e).__name__}: {e}")

    if llm_concurrency <= 1 or len(op_choices) <= 1:
        return [_gen(op) for op in op_choices]

    with ThreadPoolExecutor(max_workers=llm_concurrency) as ex:
        futs = [ex.submit(_gen, op) for op in op_choices]
        return [f.result() for f in futs]


def evolve(
    spec,
    portfolio: list[str],
    F: np.ndarray,
    training_instances: list[str],
    N: int,
    I_iter: int,
    T_max: float,
    llm_client,
    I_rep: int = 3,
    I_eff: int = 3,
    output_dir: Optional[str] = None,
    rng_seed: Optional[int] = None,
    milp_time_limit_s: float = 60.0,
    spec_module_path: Optional[str] = None,
    n_workers: int = 1,
    hard_kill_slack: float = 1.5,
    llm_concurrency: int = 1,
    use_subprocess: bool = False,
    remote_eval_cfg=None,
) -> tuple[list[str], np.ndarray, list[dict]]:
    """Run Stage Two evolution.

    Args:
        portfolio: initial Π of N solve-code strings.
        F:         pre-computed cost matrix of `portfolio` on `training_instances`.

    Returns:
        (final_portfolio, final_F, history)
        history is a list of per-iteration dicts (also written to output_dir).
    """
    output_path = Path(output_dir) if output_dir else None
    rng = np.random.default_rng(rng_seed)
    history: list[dict] = []

    assert len(portfolio) == N, f"portfolio size {len(portfolio)} != N {N}"
    assert F.shape == (len(training_instances), N), f"F shape mismatch: {F.shape}"

    cpi0 = compute_cpi(F)
    logger.info("Iter 0 (initial) CPI = %.4f", cpi0)
    _log_iteration(output_path, 0, {
        "iteration": 0,
        "cpi": cpi0,
        "portfolio_size": N,
        "note": "initial portfolio",
    })

    for iteration in range(1, I_iter + 1):
        t_iter_start = time.time()
        R = compute_rank_matrix(F)
        op_usage = {f"O{k+1}": 0 for k in range(5)}
        op_success = {f"O{k+1}": 0 for k in range(5)}
        repair_stats = {"o6_calls": 0, "o7_calls": 0}
        discarded = 0

        new_codes: list[str] = []
        new_F_cols: list[np.ndarray] = []

        while len(new_codes) < N:
            # Pre-sample a batch of operators, generate their candidates in
            # parallel (LLM is I/O bound), then process the resulting code
            # strings sequentially (smoke + eval-with-repair).
            needed = N - len(new_codes)
            batch_size = max(1, min(N, needed))
            op_choices = [int(rng.integers(0, 5)) for _ in range(batch_size)]
            for c in op_choices:
                op_usage[f"O{c+1}"] += 1

            batch = _generate_candidates_parallel(
                spec, portfolio, F, R, llm_client, op_choices, llm_concurrency,
            )

            for code, meta, op_idx, gen_err in batch:
                if len(new_codes) >= N:
                    break
                op_name = f"O{op_idx+1}"
                if code is None:
                    logger.warning("Iter %d: %s.generate raised: %s", iteration, op_name, gen_err)
                    discarded += 1
                    continue

                ok, final_code, perf_vec, stats = run_with_repair(
                    spec, code, training_instances, T_max, I_rep, I_eff, llm_client,
                    spec_module_path=spec_module_path, n_workers=n_workers,
                    hard_kill_slack=hard_kill_slack, use_subprocess=use_subprocess,
                    remote_eval_cfg=remote_eval_cfg,
                )
                repair_stats["o6_calls"] += stats.get("o6_calls", 0)
                repair_stats["o7_calls"] += stats.get("o7_calls", 0)

                if not ok:
                    logger.info(
                        "Iter %d: %s candidate discarded (%s)",
                        iteration, op_name, stats.get("failed_at", "?"),
                    )
                    discarded += 1
                    continue

                op_success[op_name] += 1
                new_codes.append(final_code)
                new_F_cols.append(perf_vec)
                logger.info(
                    "Iter %d: %s candidate accepted (%d/%d) -- perf min %.3f mean %.3f",
                    iteration, op_name, len(new_codes), N,
                    float(perf_vec[perf_vec < PENALTY].min()) if (perf_vec < PENALTY).any() else float("nan"),
                    float(perf_vec[perf_vec < PENALTY].mean()) if (perf_vec < PENALTY).any() else float("nan"),
                )

        # build pool and MILP-select
        pool_codes = portfolio + new_codes
        pool_F = np.hstack([F, np.column_stack(new_F_cols)])
        try:
            selected = milp_select(pool_F, n_select=N, time_limit_s=milp_time_limit_s)
        except Exception as e:
            logger.error("Iter %d: MILP raised %s -- keeping previous portfolio", iteration, e)
            selected = list(range(N))

        portfolio = [pool_codes[i] for i in selected]
        F = pool_F[:, selected]
        cpi = compute_cpi(F)

        iter_payload = {
            "iteration": iteration,
            "elapsed_s": time.time() - t_iter_start,
            "cpi": cpi,
            "op_usage": op_usage,
            "op_success": op_success,
            "repair_stats": repair_stats,
            "discarded": discarded,
            "selected_indices_in_pool": selected,
            "new_count_from_pool": sum(1 for i in selected if i >= N),
        }
        history.append(iter_payload)
        _log_iteration(output_path, iteration, iter_payload)

        # Persist the live portfolio code + train F after EVERY iter so a
        # later-iter crash doesn't lose the evolved code. Writes TWO files:
        # `current_portfolio.json` (always overwrites, easiest to find), and
        # `portfolio_iter_{N}.json` (per-iter snapshot, preserves history so
        # any iteration's portfolio is recoverable independently).
        if output_path:
            try:
                save_portfolio(portfolio, str(Path(output_path) / "current_portfolio.json"))
                save_portfolio(portfolio, str(Path(output_path) / f"portfolio_iter_{iteration:04d}.json"))
                np.save(Path(output_path) / "current_train_F.npy", F)
                np.save(Path(output_path) / f"train_F_iter_{iteration:04d}.npy", F)
            except Exception as _e:
                logger.warning("per-iter portfolio dump failed: %s", _e)

        logger.info(
            "Iter %d done: CPI %.4f  selected %d from new  ops %s  elapsed %.1fs",
            iteration, cpi, iter_payload["new_count_from_pool"], op_usage,
            iter_payload["elapsed_s"],
        )

    return portfolio, F, history


# ---------- final portfolio / write helpers ----------

def save_portfolio(portfolio: list[str], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(portfolio, f, indent=2, ensure_ascii=False)


def evaluate_on_test(
    spec,
    portfolio: list[str],
    test_instances: list[str],
    T_max: float,
    spec_module_path: Optional[str] = None,
    n_workers: int = 1,
    hard_kill_slack: float = 1.5,
    use_subprocess: bool = False,
    remote_eval_cfg=None,
) -> dict:
    F_test, infos = evaluate_portfolio(
        spec, portfolio, test_instances, T_max,
        spec_module_path=spec_module_path, n_workers=n_workers,
        hard_kill_slack=hard_kill_slack, use_subprocess=use_subprocess,
        remote_eval_cfg=remote_eval_cfg,
    )
    return {
        "F_test": F_test.tolist(),
        "infos": infos,
        "cpi_test": compute_cpi(F_test),
        "per_instance_best": F_test.min(axis=1).tolist(),
        "instance_paths": test_instances,
    }
