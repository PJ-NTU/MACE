"""Client-side helper: dispatch a batch of (code, instance) evaluations to a
remote SSH host running _eval_worker_batch.

Usage:

    from mace.evolution.remote_eval import RemoteEvalConfig, remote_evaluate_code_on_instances

    cfg = RemoteEvalConfig(
        host="root@153.0.134.134", port=53023, key_path="~/.ssh/cn_id",
        remote_python="cd ~/task && source ~/venv/bin/activate && python",
        n_workers=25,
    )
    perf, infos = remote_evaluate_code_on_instances(cfg, spec_module_path, code, instance_paths, T_max)

Returns same shape as evolve.evaluate_code_on_instances:
    perf:  np.ndarray of cost per instance
    infos: list of dict per instance
"""
from __future__ import annotations
from dataclasses import dataclass, field
import json
import subprocess
from typing import Optional, List, Tuple
import numpy as np

PENALTY = 1e10


@dataclass
class RemoteEvalConfig:
    host: str                # e.g. "root@153.0.134.134"
    port: int = 22
    key_path: Optional[str] = None        # ~/.ssh/somekey
    remote_python: str = "python3"        # full shell prefix that runs python
    n_workers: int = 25                   # parallel processes on remote
    hard_kill_slack: float = 2.0
    extra_timeout_s: float = 60.0         # added on top of compute estimate
    control_path: Optional[str] = None    # SSH ControlPath for multiplexing
    # Path rewrite: instance_paths starting with local_task_root get rewritten
    # to remote_task_root before being sent.  Both sides must have same data
    # layout under their respective task roots.
    local_task_root: Optional[str] = None
    remote_task_root: str = "/root/task"

    def rewrite_path(self, p: str) -> str:
        if self.local_task_root and p.startswith(self.local_task_root):
            return self.remote_task_root + p[len(self.local_task_root):]
        return p

    def ssh_cmd_prefix(self) -> List[str]:
        cmd = ["ssh", "-o", "StrictHostKeyChecking=no",
                "-o", "BatchMode=yes",
                "-o", "ServerAliveInterval=30",
                "-p", str(self.port)]
        if self.key_path:
            cmd += ["-i", self.key_path]
        if self.control_path:
            cmd += ["-o", f"ControlPath={self.control_path}",
                     "-o", "ControlMaster=auto",
                     "-o", "ControlPersist=10m"]
        cmd.append(self.host)
        return cmd


def remote_evaluate_code_on_instances(
    cfg: RemoteEvalConfig,
    spec_module_path: str,
    code: str,
    instance_paths: List[str],
    T_max: float,
) -> Tuple[np.ndarray, List[dict]]:
    """Send a (code, instance_paths[]) batch to the remote host. Block until done."""
    M = len(instance_paths)
    if M == 0:
        return np.zeros(0, dtype=np.float64), []
    payload = json.dumps({
        "spec_module_path": spec_module_path,
        "T_max": T_max,
        "n_workers": cfg.n_workers,
        "hard_kill_slack": cfg.hard_kill_slack,
        "code": code,
        "cells": [{"instance_path": cfg.rewrite_path(p), "tag": str(i)}
                    for i, p in enumerate(instance_paths)],
    })
    # Estimate timeout: ceil(M / n_workers) * T_max * slack + overhead
    import math
    batches = max(1, math.ceil(M / max(1, cfg.n_workers)))
    timeout = batches * T_max * cfg.hard_kill_slack + cfg.extra_timeout_s

    ssh_cmd = cfg.ssh_cmd_prefix() + [
        f"{cfg.remote_python} -m mace.evolution._eval_worker_batch"
    ]
    try:
        result = subprocess.run(
            ssh_cmd, input=payload, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        perf = np.full(M, PENALTY, dtype=np.float64)
        infos = [{"status": "remote_batch_hard_timeout",
                   "msg": f">{timeout:.0f}s"} for _ in range(M)]
        return perf, infos
    except Exception as e:
        perf = np.full(M, PENALTY, dtype=np.float64)
        infos = [{"status": "remote_batch_launch_error",
                   "msg": f"{type(e).__name__}: {e}"} for _ in range(M)]
        return perf, infos

    if result.returncode != 0:
        perf = np.full(M, PENALTY, dtype=np.float64)
        infos = [{"status": "remote_batch_exit",
                   "code": result.returncode,
                   "stderr": (result.stderr or "")[:500]} for _ in range(M)]
        return perf, infos

    # Last non-empty line of stdout should be a single JSON array
    lines = [ln for ln in (result.stdout or "").splitlines() if ln.strip()]
    if not lines:
        perf = np.full(M, PENALTY, dtype=np.float64)
        infos = [{"status": "remote_batch_empty_stdout",
                   "stderr": (result.stderr or "")[:500]} for _ in range(M)]
        return perf, infos
    try:
        data = json.loads(lines[-1])
    except Exception as e:
        perf = np.full(M, PENALTY, dtype=np.float64)
        infos = [{"status": "remote_batch_parse_error",
                   "msg": f"{type(e).__name__}: {e}",
                   "stdout": (result.stdout or "")[:500]} for _ in range(M)]
        return perf, infos
    if not isinstance(data, list) or len(data) != M:
        perf = np.full(M, PENALTY, dtype=np.float64)
        infos = [{"status": "remote_batch_shape_error",
                   "got_len": len(data) if isinstance(data, list) else None,
                   "expected": M} for _ in range(M)]
        return perf, infos

    perf = np.full(M, PENALTY, dtype=np.float64)
    infos: List[dict] = [{} for _ in range(M)]
    for cell in data:
        try:
            i = int(cell.get("tag"))
        except Exception:
            continue
        if 0 <= i < M:
            perf[i] = float(cell["cost"])
            infos[i] = cell["info"]
    return perf, infos
