"""Multi-Base Tugboat Routing and Scheduling Problem (MTRSP-MB) instance
generator.

Generates MTRSP-MB instances in `param = value` text format under
`data/CO-Bench/Multi-Base Tugboat Routing and Scheduling Problem/`. Each
generated file holds ONE instance — `config.py:load_data(path)` returns a
1-element list of dicts.

Size classes (n tasks / m tugs / p bases; T_max = 24 hours fixed):

    train profiles (fast LLM iteration):
        5_3_2     :  n=5   m=3   p=2
        8_5_2     :  n=8   m=5   p=2
        12_6_3    :  n=12  m=6   p=3

    test profiles (paper-scale benchmark):
        15_8_3    :  n=15  m=8   p=3
        20_10_3   :  n=20  m=10  p=3
        25_12_4   :  n=25  m=12  p=4
        30_15_4   :  n=30  m=15  p=4
        40_18_5   :  n=40  m=18  p=5
        50_20_5   :  n=50  m=20  p=5

Examples:
    python -m problems.cobench.multi_base_tugboat_routing_and_scheduling_problem.build_instances --tier all
    python -m problems.cobench.multi_base_tugboat_routing_and_scheduling_problem.build_instances --tier train
    python -m problems.cobench.multi_base_tugboat_routing_and_scheduling_problem.build_instances --profile 8_5_2 --count 3 --seed-base 1

Design choices:

  * Coordinate-based: bases + tasks placed in a 10×10 nautical-mile port area.
    v = 10 knots, so travel times stay within 0–1.4 hours.
  * Time windows widen with horizon; service durations 0.5–2.0 h; HP demands
    chosen so most tasks can be served by 1 tug, ~30% need 2-tug collab.
  * Pre-write sanity check: at least one base has tugs whose top-Γ_max HP
    sum ≥ the largest task's H_s^min, and at least 70% of tasks have a
    feasible time window that admits service before T_max with a typical
    travel time. On failure, re-seed up to 5 times.
"""
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path


# ============================== Constants ==============================

SIZE_PRESETS = {
    # Legacy train/test (tiny scales) - kept for compatibility
    "5_3_2":   dict(n=5,  m=3,  p=2, T_max=24.0),
    "8_5_2":   dict(n=8,  m=5,  p=2, T_max=24.0),
    "12_6_3":  dict(n=12, m=6,  p=3, T_max=24.0),
    "15_8_3":  dict(n=15, m=8,  p=3, T_max=24.0),
    "20_10_3": dict(n=20, m=10, p=3, T_max=24.0),
    "25_12_4": dict(n=25, m=12, p=4, T_max=24.0),
    "30_15_4": dict(n=30, m=15, p=4, T_max=24.0),
    "40_18_5": dict(n=40, m=18, p=5, T_max=24.0),
    "50_20_5": dict(n=50, m=20, p=5, T_max=24.0),
    # ─── New large-scale test profiles (literature benchmark, n=10..290) ───
    # m = n/5, p = max(2, ceil(m/4)) capped at 10
    "10_2_2":   dict(n=10,  m=2,  p=2,  T_max=24.0),
    "30_6_2":   dict(n=30,  m=6,  p=2,  T_max=24.0),
    "50_10_3":  dict(n=50,  m=10, p=3,  T_max=24.0),
    "70_14_4":  dict(n=70,  m=14, p=4,  T_max=24.0),
    "90_18_5":  dict(n=90,  m=18, p=5,  T_max=24.0),
    "110_22_6": dict(n=110, m=22, p=6,  T_max=24.0),
    "130_26_7": dict(n=130, m=26, p=7,  T_max=24.0),
    "150_30_8": dict(n=150, m=30, p=8,  T_max=24.0),
    "170_34_9": dict(n=170, m=34, p=9,  T_max=24.0),
    "190_38_10":dict(n=190, m=38, p=10, T_max=24.0),
    "210_42_10":dict(n=210, m=42, p=10, T_max=24.0),
    "230_46_10":dict(n=230, m=46, p=10, T_max=24.0),
    "250_50_10":dict(n=250, m=50, p=10, T_max=24.0),
    "270_54_10":dict(n=270, m=54, p=10, T_max=24.0),
    "290_58_10":dict(n=290, m=58, p=10, T_max=24.0),
}

BUILD_PLAN = {
    "train": {
        "5_3_2":   43,
        "8_5_2":   43,
        "12_6_3":  42,
    },
    "test": {
        "15_8_3":  17,
        "20_10_3": 17,
        "25_12_4": 17,
        "30_15_4": 17,
        "40_18_5": 16,
        "50_20_5": 16,
    },
}

# Hard mode: large-scale literature-benchmark profiles for test (15 profiles, 100 instances)
BUILD_PLAN_HARD_TEST = {
    "10_2_2":   7, "30_6_2":   7, "50_10_3":  7, "70_14_4":  7, "90_18_5":  7,
    "110_22_6": 7, "130_26_7": 7, "150_30_8": 7, "170_34_9": 7, "190_38_10":7,
    "210_42_10":6, "230_46_10":6, "250_50_10":6, "270_54_10":6, "290_58_10":6,
}  # = 100

# Port-area side length (nautical miles). Travel time = distance / v.
PORT_SIDE_NM = 10.0
TUG_SPEED_KNOTS = 10.0  # 1 nm = 0.1 hour at 10 kn

# Big-M, penalty-weight (per task description)
BIG_M = 1000.0
PENALTY_WEIGHT = 10000.0

DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[3]
    / "data" / "CO-Bench" / "Multi-Base Tugboat Routing and Scheduling Problem"
)


# ============================== Generation helpers ==============================

def _random_point(rng: random.Random) -> tuple[float, float]:
    return (
        round(rng.uniform(0.0, PORT_SIDE_NM), 2),
        round(rng.uniform(0.0, PORT_SIDE_NM), 2),
    )


def _euclid(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _gen_bases(rng: random.Random, p: int, m: int) -> tuple[list, list[int], list[int]]:
    """Return (base_locs, base_caps, tug_base_assignment).

    base_locs[i] is the (x,y) of base -(i+1).
    base_caps[i] is δ for base -(i+1).
    tug_base_assignment[k] is the negative base id for tug k (k=0..m-1).
    """
    base_locs = [_random_point(rng) for _ in range(p)]

    # Distribute m tugs over p bases roughly evenly, plus randomness.
    base_of_tug = []
    base_count = [0] * p
    for k in range(m):
        b = k % p
        base_of_tug.append(-(b + 1))
        base_count[b] += 1
    # Shuffle assignment so tug-0 isn't always at base -1.
    rng.shuffle(base_of_tug)

    # Recompute counts after shuffle
    base_count = [0] * p
    for k in range(m):
        base_count[-base_of_tug[k] - 1] += 1

    # Base capacity δ_b = ceil(1.0 * base_count_b) — i.e., a base can dispatch
    # all of its own tugs.
    base_caps = [max(1, base_count[i]) for i in range(p)]
    return base_locs, base_caps, base_of_tug


def _gen_tugs(rng: random.Random, m: int, hard: bool = False) -> tuple[list[float], list[float], list[float], list[float]]:
    """Tiered HP. hard=True bumps tug HP up so small-profile single-base
    collaborative HP capacity can still cover bumped HP demand."""
    high = max(1, m // 3)
    mid_ = max(1, m // 3)
    low = max(0, m - high - mid_)
    if hard:
        hi_r = (4500, 6500); md_r = (2500, 4500); lo_r = (1500, 2800)
    else:
        hi_r = (3000, 5000); md_r = (1500, 3500); lo_r = (800, 1800)
    hps: list[float] = []
    hps += [round(rng.uniform(*hi_r), 0) for _ in range(high)]
    hps += [round(rng.uniform(*md_r), 0) for _ in range(mid_)]
    hps += [round(rng.uniform(*lo_r), 0) for _ in range(low)]
    rng.shuffle(hps)

    alphas: list[float] = []
    betas: list[float] = []
    fuels: list[float] = []
    for hp in hps:
        alphas.append(round(rng.uniform(0.04, 0.07), 4))
        betas.append(round(rng.uniform(0.06, 0.10), 4))
        if hard:
            # Fuel sized for ~1.5 tasks worth of work -> binds at optimum
            typical_burn = (
                alphas[-1] * hp * 2.5 * 1.5   # service: 2.5h x 1.5 tasks
                + betas[-1] * hp * 0.6 * 2.5  # transit: 0.6h x 2.5 edges
            )
            fuels.append(round(typical_burn * rng.uniform(0.9, 1.15) / 50) * 50)
        else:
            typical_burn = (
                alphas[-1] * hp * 1.2 * 4
                + betas[-1] * hp * 0.6 * 5
            )
            fuels.append(round(typical_burn * rng.uniform(1.0, 1.4) / 50) * 50)
    return hps, fuels, alphas, betas


def _gen_tasks(rng: random.Random, n: int, T_max: float,
               max_tug_hp: float, hard: bool = False) -> list[dict]:
    """Generate n tasks. hard=True tightens service times, windows, HP demand."""
    tasks = []
    for s in range(1, n + 1):
        start_loc = _random_point(rng)
        end_loc = _random_point(rng)
        if hard:
            T_s = round(rng.uniform(1.5, 3.5), 2)
            window_w = rng.uniform(0.4, 1.2)
        else:
            T_s = round(rng.uniform(0.5, 2.0), 2)
            window_w = rng.uniform(1.0, 4.0)
        a_s = round(rng.uniform(0.0, max(0.5, T_max - T_s - 3.0)), 2)
        w = round(window_w, 2)
        b_s = min(T_max - T_s, a_s + w)
        if b_s < a_s:
            b_s = a_s
        # Γ_s distribution: hard makes more multi-tug tasks
        r = rng.random()
        if hard:
            gamma_s = 1 if r < 0.35 else (2 if r < 0.80 else 3)
        else:
            gamma_s = 1 if r < 0.60 else (2 if r < 0.90 else 3)
        # H_s^min: hard pushes upward so single tug often can't cover
        if hard:
            if gamma_s == 1:
                hp_req = round(rng.uniform(1500, 3000), 0)
            elif gamma_s == 2:
                hp_req = round(rng.uniform(3500, 6000), 0)
            else:
                hp_req = round(rng.uniform(5500, 9000), 0)
        else:
            if gamma_s == 1:
                hp_req = round(rng.uniform(700, 2200), 0)
            elif gamma_s == 2:
                hp_req = round(rng.uniform(2200, 4500), 0)
            else:
                hp_req = round(rng.uniform(3500, 6500), 0)
        hp_req = min(hp_req, max_tug_hp * gamma_s * 0.95)
        tasks.append({
            "id": s,
            "start": start_loc,
            "end": end_loc,
            "a_s": a_s,
            "b_s": round(b_s, 2),
            "T_s": T_s,
            "gamma_s": gamma_s,
            "hp_req": float(hp_req),
        })
    return tasks


def _build_time_matrix(bases: list, tasks: list[dict],
                       n: int, p: int) -> dict[str, float]:
    """Construct sparse time_matrix dict[str, float] with keys:
       'b_s'                base origin b (negative)  → task s entrance
       'i_j'                task i exit → task j entrance (i ≠ j)
       's_(n+|b|)'          task s exit → base b destination (positive)
    """
    tm: dict[str, float] = {}
    v = TUG_SPEED_KNOTS
    # Base origin → task start
    for b_idx in range(p):
        b = -(b_idx + 1)
        bx_loc = bases[b_idx]
        for task in tasks:
            t = round(_euclid(bx_loc, task["start"]) / v, 4)
            tm[f"{b}_{task['id']}"] = t
    # Task exit → task start (i ≠ j)
    for i_task in tasks:
        for j_task in tasks:
            if i_task["id"] == j_task["id"]:
                continue
            t = round(_euclid(i_task["end"], j_task["start"]) / v, 4)
            tm[f"{i_task['id']}_{j_task['id']}"] = t
    # Task exit → base destination
    for s_task in tasks:
        for b_idx in range(p):
            b = -(b_idx + 1)
            bx_loc = bases[b_idx]
            t = round(_euclid(s_task["end"], bx_loc) / v, 4)
            # Destination node id = n - b = n + (b_idx + 1)
            dest_node = n - b
            tm[f"{s_task['id']}_{dest_node}"] = t
    return tm


# ============================== Sanity check ==============================

def sanity_check(inst: dict, min_servable_frac: float = 0.7) -> tuple[bool, str | None]:
    """Verify the instance is non-trivially solvable.

    Hard requirements:
      - At least one base can muster top-Γ_max tugs with HP sum ≥ max H_s^min
        (so the largest task has SOME feasible serving team in some base).
      - At least min_servable_frac of tasks have a feasible time window such
        that some tug from some base can travel there and complete the
        service before T_max.
    """
    n = inst["num_tasks"]
    m = inst["num_tugboats"]
    p = inst["num_bases"]
    hps = inst["tugboat_horsepower"]
    base_of_tug = inst["tugboat_base_assignment"]
    hp_reqs = inst["task_min_horsepower"]
    gammas = inst["task_max_tugs"]
    a_s = inst["task_time_window_lower"]
    b_s = inst["task_time_window_upper"]
    T_s = inst["task_service_time"]
    T_max = inst["planning_horizon"]
    tm = inst["time_matrix"]

    # Group tugs by base
    by_base: dict[int, list[float]] = {}
    for k in range(m):
        by_base.setdefault(base_of_tug[k], []).append(hps[k])

    max_req = max(hp_reqs)
    max_gamma = max(gammas)
    can_serve_largest = False
    for b, base_hps in by_base.items():
        top = sorted(base_hps, reverse=True)[:max_gamma]
        if sum(top) >= max_req:
            can_serve_largest = True
            break
    if not can_serve_largest:
        return False, (f"no base has top-{max_gamma} tugs with HP ≥ max H_s^min={max_req}")

    # Feasibility of time window: from the closest base, the cheapest tug can
    # reach the task and complete service before T_max.
    servable = 0
    for s in range(1, n + 1):
        # Cheapest travel from any base to task s entrance
        t_b_s_min = min(tm[f"{b}_{s}"] for b in by_base.keys())
        earliest_arrival = t_b_s_min
        # task can start at max(a_s, earliest_arrival), must end by T_max
        start = max(a_s[s - 1], earliest_arrival)
        if start <= b_s[s - 1] + 1e-9 and start + T_s[s - 1] <= T_max + 1e-9:
            servable += 1
    need = max(1, int(n * min_servable_frac))
    if servable < need:
        return False, f"only {servable}/{n} tasks have a feasible window (need ≥ {need})"
    return True, None


# ============================== Build one instance ==============================

def build(size_class: str = None, *, n: int = None, m: int = None, p: int = None,
          T_max: float = None, seed: int = 0, hard: bool = False) -> dict:
    """Build one MTRSP-MB instance dict. hard=True activates tightened params."""
    if size_class is not None:
        c = SIZE_PRESETS[size_class]
        n, m, p, T_max = c["n"], c["m"], c["p"], c["T_max"]
    elif None in (n, m, p, T_max):
        raise ValueError("either size_class or full (n, m, p, T_max) required")

    rng = random.Random(seed)
    bases, base_caps, base_of_tug = _gen_bases(rng, p, m)
    hps, fuel_caps, alphas, betas = _gen_tugs(rng, m, hard=hard)
    max_hp = max(hps)
    tasks = _gen_tasks(rng, n, T_max, max_hp, hard=hard)
    tm = _build_time_matrix(bases, tasks, n, p)

    inst = {
        "num_tasks":    n,
        "num_tugboats": m,
        "num_bases":    p,
        "task_max_tugs":           [t["gamma_s"] for t in tasks],
        "task_min_horsepower":     [t["hp_req"]  for t in tasks],
        "task_time_window_lower":  [t["a_s"]     for t in tasks],
        "task_time_window_upper":  [t["b_s"]     for t in tasks],
        "task_service_time":       [t["T_s"]     for t in tasks],
        "tugboat_horsepower":      hps,
        "tugboat_fuel_capacity":   fuel_caps,
        "tugboat_alpha":           alphas,
        "tugboat_beta":            betas,
        "tugboat_base_assignment": base_of_tug,
        "base_capacity":           base_caps,
        "time_matrix":             tm,
        "big_M":            BIG_M,
        "planning_horizon": T_max,
        "penalty_weight":   PENALTY_WEIGHT,
    }
    return inst


def build_with_retry(size_class: str = None, *, seed: int = 0,
                     max_retries: int = 8, hard: bool = False,
                     verbose: bool = False, **kwargs) -> dict:
    """Sanity-checked build. Re-seed up to max_retries on failure."""
    last_reason = None
    # In hard mode allow a much lower min-servable threshold — we WANT many
    # tasks to be unservable so the heuristic has to make real choices.
    sanity_thresh = 0.4 if hard else 0.7
    for attempt in range(max_retries):
        s = seed + attempt * 100003
        inst = build(size_class, seed=s, hard=hard, **kwargs)
        ok, reason = sanity_check(inst, min_servable_frac=sanity_thresh)
        if ok:
            return inst
        last_reason = reason
        if verbose:
            print(f"  retry attempt={attempt} seed={s}: {reason}")
    raise RuntimeError(f"sanity check failed after {max_retries} attempts: {last_reason}")


# ============================== Write to file ==============================

PARAM_ORDER = [
    "num_tasks", "num_tugboats", "num_bases",
    "task_max_tugs", "task_min_horsepower",
    "task_time_window_lower", "task_time_window_upper", "task_service_time",
    "tugboat_horsepower", "tugboat_fuel_capacity",
    "tugboat_alpha", "tugboat_beta", "tugboat_base_assignment",
    "base_capacity",
    "time_matrix",
    "big_M", "planning_horizon", "penalty_weight",
]


def write_instance(path: Path, inst: dict, header_comment: str = ""):
    lines = []
    if header_comment:
        lines.append(f"# {header_comment}")
    lines.append(
        f"# n={inst['num_tasks']} m={inst['num_tugboats']} p={inst['num_bases']} "
        f"T_max={inst['planning_horizon']}"
    )
    lines.append("")
    for key in PARAM_ORDER:
        if key not in inst:
            raise KeyError(f"instance missing required key {key!r}")
        lines.append(f"{key} = {inst[key]!r}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ============================== CLI ==============================

def _generate_tier(tier: str, out_base: Path, verbose: bool,
                     hard: bool = False) -> list[Path]:
    """Generate all instances for one tier per BUILD_PLAN."""
    # Hard-mode test uses 15 large-scale literature-benchmark profiles.
    # We generate them with LOOSE params (hard params infeasible for tiny m).
    if hard and tier == "test":
        plan = BUILD_PLAN_HARD_TEST
        use_hard_params = False
    else:
        plan = BUILD_PLAN[tier]
        use_hard_params = hard
    suffix = "_hard" if hard else ""
    tier_dir = out_base / (tier + suffix)
    tier_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for profile, count in plan.items():
        for k in range(count):
            seed = k + 1
            inst = build_with_retry(size_class=profile, seed=seed,
                                     hard=use_hard_params, verbose=verbose)
            fname = f"{profile}_{seed:03d}.txt"
            path = tier_dir / fname
            write_instance(path, inst,
                           header_comment=f"MTRSP-MB {tier}{suffix} | {profile} seed={seed}")
            written.append(path)
            if verbose:
                print(f"  wrote {tier}{suffix}/{fname}")
    return written


def main():
    p = argparse.ArgumentParser(
        description=("Generate MTRSP-MB instances under data/CO-Bench/"
                     "Multi-Base Tugboat Routing and Scheduling Problem/."),
    )
    group = p.add_mutually_exclusive_group()
    group.add_argument("--tier", choices=["train", "test", "all"],
                       help="Build all instances of a tier per BUILD_PLAN.")
    group.add_argument("--profile", choices=list(SIZE_PRESETS.keys()),
                       help="Build a single profile (use --count and --seed-base).")
    p.add_argument("--count", type=int, default=3)
    p.add_argument("--seed-base", type=int, default=1)
    p.add_argument("--out", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--hard", action="store_true",
                     help="Use tightened parameters (binds fuel, narrow windows).")
    args = p.parse_args()

    out_base = Path(args.out)
    out_base.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    if args.tier:
        tiers = ["train", "test"] if args.tier == "all" else [args.tier]
        for t in tiers:
            written += _generate_tier(t, out_base, args.verbose, hard=args.hard)
    else:
        prof = args.profile or "5_3_2"
        suffix = "_hard" if args.hard else ""
        for k in range(args.count):
            seed = args.seed_base + k
            inst = build_with_retry(size_class=prof, seed=seed, hard=args.hard,
                                     verbose=args.verbose)
            fname = f"{prof}_{seed:03d}.txt"
            out_sub = out_base / f"pilot{suffix}"
            out_sub.mkdir(parents=True, exist_ok=True)
            path = out_sub / fname
            write_instance(path, inst,
                            header_comment=f"MTRSP-MB {prof}{suffix} seed={seed}")
            written.append(path)

    print(f"Generated {len(written)} instance(s) under {out_base}")


if __name__ == "__main__":
    main()
