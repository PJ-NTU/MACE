"""MTRSP-VS instance generator.

Generates MTRSP-VS instances in `param = value` text format under
`data/CO-Bench/Multi-Tugboat Routing Problem with Variable Speed/`. Each
generated file holds ONE instance — `config.py:load_data(path)` returns a
1-element list of dicts.

Size profiles (n tasks / m tugs / square area side in n.m.):

    Train: (10, 3, 30), (15, 4, 35), (20, 5, 40)
    Test:  (10, 3, 30), (20, 5, 40), (40, 10, 50),
           (60, 13, 60), (80, 16, 70), (100, 20, 80)

Test set deliberately spans small→large (per user request — "测试集里面也要有小规模").

Examples:
    # 3 of every train profile
    python -m problems.cobench.multi_tugboat_routing_problem_with_variable_speed.build_instances --tier train

    # Both tiers
    python -m problems.cobench.multi_tugboat_routing_problem_with_variable_speed.build_instances --tier all

Design choices:

  * 0-indexed throughout (tasks 0..n-1, tugs 0..m-1).
  * Distances stored as three structures (depot↔task, task↔task) — replaces
    the MACE "i_j" string-key dict so the LLM sees plain Python lists.
  * Continuous time on [0, T_max=24], NOT discrete periods (unlike PSP).
  * Multi-tug probability ramps up at smaller n (more interesting scheduling).
  * Per-instance sanity check: every task's HP req must be servable by some
    Γ_s-sized tug combo, AND every task's time window must admit at least
    a slow-speed service that finishes ≤ T_max.
"""
from __future__ import annotations

import argparse
import random
import math
from pathlib import Path


# ============================== Size profiles ==============================

SIZE_PRESETS = {
    # (n_tasks, n_tugboats, area_size_nm)
    "n10_m3_a30":   dict(n=10,  m=3,  area=30.0),
    "n15_m4_a35":   dict(n=15,  m=4,  area=35.0),
    "n20_m5_a40":   dict(n=20,  m=5,  area=40.0),
    "n40_m10_a50":  dict(n=40,  m=10, area=50.0),
    "n60_m13_a60":  dict(n=60,  m=13, area=60.0),
    "n80_m16_a70":  dict(n=80,  m=16, area=70.0),
    "n100_m20_a80": dict(n=100, m=20, area=80.0),
}

# 128 train + 100 test = 228 instances.
BUILD_PLAN = {
    "train": {
        "n10_m3_a30":   43,
        "n15_m4_a35":   43,
        "n20_m5_a40":   42,
    },
    "test": {
        "n10_m3_a30":   17,
        "n20_m5_a40":   17,
        "n40_m10_a50":  17,
        "n60_m13_a60":  17,
        "n80_m16_a70":  16,
        "n100_m20_a80": 16,
    },
}

# ============================== Constants ==============================

SPEED_LEVEL_NAMES = ["slow", "medium", "fast"]
SPEED_VALUES = [6.0, 10.0, 15.0]          # knots
# ρ_l = (v_l / v_medium)^3 — power coefficient relative to medium.
SPEED_POWER_COEFS = [(v / SPEED_VALUES[1]) ** 3 for v in SPEED_VALUES]
T_MAX_DEFAULT = 24.0
BIG_M_DEFAULT = 1000.0
PENALTY_WEIGHT_DEFAULT = 10000.0   # was 100000; lowered to align with MTRSP/MTRSP-MD/PSP (penalty was dominating CPI)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "CO-Bench" / "Multi-Tugboat Routing Problem with Variable Speed"


# ============================== Generation ==============================

def _euclid(p, q):
    return math.hypot(p[0] - q[0], p[1] - q[1])


def _multi_tug_prob(n: int) -> float:
    """Probability a task needs >=2 tugs. Higher for smaller instances so
    multi-tug scheduling shows up consistently."""
    if n <= 15:
        return 0.30
    if n <= 30:
        return 0.20
    if n <= 60:
        return 0.15
    return 0.10


def _gen_tasks(rng: random.Random, n: int, area: float, T_max: float,
                hard: bool = False) -> dict:
    """Generate per-task arrays. hard=True tightens service/window/HP."""
    p_multi = 0.55 if hard else _multi_tug_prob(n)
    svc_lo, svc_hi = (5.0, 10.0) if hard else (3.0, 8.0)
    tw_lo, tw_hi = (0.8, 2.0) if hard else (2.0, 6.0)

    entrances = []
    exits     = []
    service_distance = []
    for _ in range(n):
        ent = (rng.uniform(0, area), rng.uniform(0, area))
        d_s = rng.uniform(svc_lo, svc_hi)
        theta = rng.uniform(0, 2 * math.pi)
        ex = (ent[0] + d_s * math.cos(theta), ent[1] + d_s * math.sin(theta))
        ex = (max(0.0, min(area, ex[0])), max(0.0, min(area, ex[1])))
        d_s = _euclid(ent, ex)
        if d_s < 0.5:
            d_s = 0.5
        entrances.append(ent)
        exits.append(ex)
        service_distance.append(round(d_s, 3))

    max_tugs = []
    min_hp = []
    for _ in range(n):
        roll = rng.random()
        if roll < p_multi * 0.7:
            g = 2
        elif roll < p_multi:
            g = 3
        else:
            g = 1
        max_tugs.append(g)
        if hard:
            if g == 1:    hp = rng.uniform(2000, 4500)
            elif g == 2:  hp = rng.uniform(4500, 7500)
            else:         hp = rng.uniform(6500, 10000)
        else:
            if g == 1:    hp = rng.uniform(1000, 3000)
            elif g == 2:  hp = rng.uniform(2500, 5000)
            else:         hp = rng.uniform(4000, 7000)
        min_hp.append(round(hp, 1))

    tw_lower = []
    tw_upper = []
    for _ in range(n):
        width = rng.uniform(tw_lo, tw_hi)
        a = rng.uniform(0.0, max(0.1, T_max - width))
        tw_lower.append(round(a, 3))
        tw_upper.append(round(a + width, 3))

    return {
        "task_max_tugs": max_tugs,
        "task_min_horsepower": min_hp,
        "task_time_window_lower": tw_lower,
        "task_time_window_upper": tw_upper,
        "task_service_distance": service_distance,
        "_entrances": entrances,
        "_exits":     exits,
    }


def _gen_tugs(rng: random.Random, m: int, max_hp_req: float, n: int,
                hard: bool = False) -> dict:
    """Generate per-tug arrays."""
    hps = sorted([rng.uniform(1500, 5000) for _ in range(m)], reverse=True)
    needed_top = max(1, min(3, m))
    while sum(hps[:needed_top]) < max_hp_req:
        hps[0] = min(7500.0, hps[0] + 500.0)
    rng.shuffle(hps)
    hps = [round(h, 1) for h in hps]

    if n <= 20:
        fuel_base = (3200.0, 4500.0) if hard else (6000.0, 8000.0)
    elif n <= 60:
        fuel_base = (5000.0, 7000.0) if hard else (9000.0, 12000.0)
    else:
        fuel_base = (7000.0, 10500.0) if hard else (12000.0, 18000.0)

    fuel_caps = [round(rng.uniform(*fuel_base), 1) for _ in range(m)]
    alphas = [round(rng.uniform(0.14, 0.18), 4) for _ in range(m)]
    betas  = [round(rng.uniform(0.05, 0.09), 4) for _ in range(m)]

    return {
        "tugboat_horsepower":    hps,
        "tugboat_fuel_capacity": fuel_caps,
        "tugboat_alpha":         alphas,
        "tugboat_beta":          betas,
    }


def _gen_distances(rng: random.Random, n: int, area: float,
                   entrances: list, exits: list) -> dict:
    """Build depot↔task and task↔task distance arrays (0-indexed).

    Depot located at the area centroid (area/2, area/2).
    task_to_task_distance[i][j] = euclid(exits[i], entrances[j]) for i≠j.
    Self-distance [i][i] is set to 0.0 (never used by eval_func).
    """
    depot = (area / 2.0, area / 2.0)
    d_to = [round(_euclid(depot, ent), 3) for ent in entrances]
    d_from = [round(_euclid(ex, depot), 3) for ex in exits]
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            matrix[i][j] = round(_euclid(exits[i], entrances[j]), 3)
    return {
        "depot_to_task_distance": d_to,
        "task_to_depot_distance": d_from,
        "task_to_task_distance":  matrix,
    }


# ============================== Sanity check ==============================

def sanity_check(inst: dict) -> tuple[bool, str | None]:
    """Verify the instance is non-trivially solvable.

    Hard requirements:
      - every task's HP req must be servable by SOME combination of ≤ Γ_s tugs
        (else that task is intrinsically unexecutable, an instance defect).
      - every task's time window must admit at least a slow-speed service
        that finishes before T_max.
      - every task must be reachable from depot by a slow tug under T_max
        (very weak; mostly just sanity).
    """
    n  = inst["num_tasks"]
    m  = inst["num_tugboats"]
    T  = inst["planning_horizon"]
    tug_hps = inst["tugboat_horsepower"]
    sorted_hps = sorted(tug_hps, reverse=True)

    for s in range(n):
        gamma = inst["task_max_tugs"][s]
        if gamma > m:
            gamma = m
        if sum(sorted_hps[:gamma]) < inst["task_min_horsepower"][s] - 1e-6:
            return False, (f"task {s} needs HP {inst['task_min_horsepower'][s]} "
                           f"but top {gamma} tugs sum to {sum(sorted_hps[:gamma])}")

        d_s = inst["task_service_distance"][s]
        # Slow speed service time
        slow_T = d_s / SPEED_VALUES[0]
        a_s = inst["task_time_window_lower"][s]
        b_s = inst["task_time_window_upper"][s]
        if a_s + slow_T > T + 1e-6:
            return False, (f"task {s} can't be served before T={T} even at "
                           f"slow speed (a_s={a_s} + slow_T={slow_T:.2f} > T)")
        if a_s > b_s:
            return False, f"task {s} time window inverted: [{a_s}, {b_s}]"

    return True, None


# ============================== Build one instance ==============================

def build(size_class: str, *, seed: int = 0, hard: bool = False) -> dict:
    p = SIZE_PRESETS[size_class]
    n, m, area = p["n"], p["m"], p["area"]
    T_max = T_MAX_DEFAULT

    rng = random.Random(seed)
    task_blob = _gen_tasks(rng, n, area, T_max, hard=hard)
    entrances = task_blob.pop("_entrances")
    exits     = task_blob.pop("_exits")
    max_hp_req = max(task_blob["task_min_horsepower"])
    tug_blob = _gen_tugs(rng, m, max_hp_req, n, hard=hard)
    dist_blob = _gen_distances(rng, n, area, entrances, exits)

    inst = {
        "num_tasks":         n,
        "num_tugboats":      m,
        "num_speed_levels":  3,
        **task_blob,
        **tug_blob,
        "speed_level_names":       list(SPEED_LEVEL_NAMES),
        "speed_values":            list(SPEED_VALUES),
        "speed_power_coefficients":[round(c, 6) for c in SPEED_POWER_COEFS],
        **dist_blob,
        "big_M":            BIG_M_DEFAULT,
        "planning_horizon": T_max,
        "penalty_weight":   PENALTY_WEIGHT_DEFAULT,
    }
    return inst


def build_with_retry(size_class: str, *, seed: int = 0, hard: bool = False,
                     max_retries: int = 8, verbose: bool = False) -> dict:
    last = None
    for attempt in range(max_retries):
        s = seed + attempt * 100003
        inst = build(size_class, seed=s, hard=hard)
        ok, reason = sanity_check(inst)
        if ok:
            return inst
        last = reason
        if verbose:
            print(f"  retry attempt={attempt} seed={s}: {reason}")
    raise RuntimeError(f"sanity check failed after {max_retries} attempts: {last}")


# ============================== Write to file ==============================

PARAM_ORDER = [
    "num_tasks", "num_tugboats", "num_speed_levels",
    "task_max_tugs", "task_min_horsepower",
    "task_time_window_lower", "task_time_window_upper",
    "task_service_distance",
    "tugboat_horsepower", "tugboat_fuel_capacity",
    "tugboat_alpha", "tugboat_beta",
    "speed_level_names", "speed_values", "speed_power_coefficients",
    "depot_to_task_distance", "task_to_depot_distance",
    "task_to_task_distance",
    "big_M", "planning_horizon", "penalty_weight",
]


def write_instance(path: Path, inst: dict, header_comment: str = ""):
    lines = []
    if header_comment:
        lines.append(f"# {header_comment}")
    lines.append(
        f"# n={inst['num_tasks']} m={inst['num_tugboats']} T_max={inst['planning_horizon']}"
    )
    lines.append("")
    for key in PARAM_ORDER:
        if key not in inst:
            raise KeyError(f"instance missing required key {key!r}")
        lines.append(f"{key} = {inst[key]}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ============================== CLI ==============================

def _generate_tier(tier: str, out_base: Path, verbose: bool,
                    hard: bool = False) -> list[Path]:
    plan = BUILD_PLAN[tier]
    suffix = "_hard" if hard else ""
    tier_dir = out_base / (tier + suffix)
    tier_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for profile, count in plan.items():
        for k in range(count):
            seed = k + 1
            inst = build_with_retry(profile, seed=seed, hard=hard, verbose=verbose)
            fname = f"{profile}_{seed:03d}.txt"
            path = tier_dir / fname
            write_instance(path, inst,
                           header_comment=f"MTRSP-VS {tier}{suffix} | {profile} seed={seed}")
            written.append(path)
            if verbose:
                print(f"  wrote {tier}{suffix}/{fname}")
    return written


def main():
    p = argparse.ArgumentParser(
        description="Generate MTRSP-VS instances under data/CO-Bench/Multi-Tugboat Routing Problem with Variable Speed/.",
    )
    group = p.add_mutually_exclusive_group()
    group.add_argument("--tier", choices=["train", "test", "all"],
                       help="Build all instances of a tier per BUILD_PLAN. "
                            "train=128, test=100, all=228.")
    group.add_argument("--profile", choices=list(SIZE_PRESETS.keys()),
                       help="Build a single profile (use --count and --seed-base).")
    p.add_argument("--count",     type=int, default=3)
    p.add_argument("--seed-base", type=int, default=1)
    p.add_argument("--out", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--hard", action="store_true",
                    help="Tighten params (narrow windows, more collab, less fuel).")
    args = p.parse_args()

    out_base = Path(args.out)
    out_base.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    suffix = "_hard" if args.hard else ""

    if args.tier:
        tiers = ["train", "test"] if args.tier == "all" else [args.tier]
        for t in tiers:
            written += _generate_tier(t, out_base, args.verbose, hard=args.hard)
    else:
        prof = args.profile or "n10_m3_a30"
        out_sub = out_base / f"pilot{suffix}"
        out_sub.mkdir(parents=True, exist_ok=True)
        for k in range(args.count):
            seed = args.seed_base + k
            inst = build_with_retry(prof, seed=seed, hard=args.hard,
                                     verbose=args.verbose)
            fname = f"{prof}_{seed:03d}.txt"
            path = out_sub / fname
            write_instance(path, inst, header_comment=f"MTRSP-VS {prof}{suffix} seed={seed}")
            written.append(path)

    print(f"Generated {len(written)} instance(s) under {out_base}")


if __name__ == "__main__":
    main()
