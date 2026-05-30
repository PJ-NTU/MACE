"""Multi-Tugboat Routing and Scheduling Problem (MTRSP) instance generator.

Generates MTRSP instances in `param = value` text format under
`data/CO-Bench/Multi-Tugboat Routing and Scheduling Problem/`. Each generated
file holds ONE instance — `config.py:load_data(path)` returns a 1-element list.

Size profiles (all T=24 hr, single-difficulty "medium" — see ipynb source).
Each profile is (n_tasks, n_tugs, area_size):

  Train (T=24, fast LLM iteration):
    n10_K5,  n15_K6,  n20_K8,  n25_K10        (4 × 32 = 128)

  Test (T=24, paper-scale, full small→large ladder):
    n20_K8 / n30_K10 / n50_K15 / n80_K22 /
    n100_K28 / n150_K40 / n200_K55             (10+14+14+14+16+16+16 = 100)

Examples:
    # generate all 128 train + 100 test = 228 instances
    python -m problems.cobench.multi_tugboat_routing_and_scheduling_problem.build_instances --tier all

    # 3 of one profile (ad-hoc smoke)
    python -m problems.cobench.multi_tugboat_routing_and_scheduling_problem.build_instances --profile n10_K5_T24 --count 3

    # Ad-hoc custom (n K area T)
    python -m problems.cobench.multi_tugboat_routing_and_scheduling_problem.build_instances --custom 12 6 35 24

Design choices (in contrast to the upstream MACE ipynb generator):

  * Single difficulty preset (medium) — multi-difficulty would obscure
    Stage-2 evolution signals.
  * `key = value` text format (ast.literal_eval) to match other CO-Bench
    problems (PSP, etc.), not JSON.
  * Pre-write sanity check: max H_s^min ≤ best top-Γₘₐₓ tug combo, ≥50% of
    tasks have time windows that admit completion before T_max. Reseed on
    failure; abort after 5 attempts.
"""
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path


# ============================== Constants ==============================

# (n, K, area_size) per profile. T=24 hr everywhere.
SIZE_PRESETS = {
    # Train (small + fast)
    "n10_K5_T24":   dict(n=10,  K=5,  area=30.0),
    "n15_K6_T24":   dict(n=15,  K=6,  area=35.0),
    "n20_K8_T24":   dict(n=20,  K=8,  area=40.0),
    "n25_K10_T24":  dict(n=25,  K=10, area=45.0),
    # Test (paper-scale ladder, includes small for cross-checking)
    "n30_K10_T24":  dict(n=30,  K=10, area=45.0),
    "n50_K15_T24":  dict(n=50,  K=15, area=55.0),
    "n80_K22_T24":  dict(n=80,  K=22, area=70.0),
    "n100_K28_T24": dict(n=100, K=28, area=80.0),
    "n150_K40_T24": dict(n=150, K=40, area=100.0),
    "n200_K55_T24": dict(n=200, K=55, area=130.0),
}

BUILD_PLAN = {
    "train": {
        "n10_K5_T24":   32,
        "n15_K6_T24":   32,
        "n20_K8_T24":   32,
        "n25_K10_T24":  32,
    },
    # Counts sum to 100. Includes n20_K8_T24 (same profile as train, but seeds
    # are distinct between tiers so instances differ.)
    "test": {
        "n20_K8_T24":   10,
        "n30_K10_T24":  14,
        "n50_K15_T24":  14,
        "n80_K22_T24":  14,
        "n100_K28_T24": 16,
        "n150_K40_T24": 16,
        "n200_K55_T24": 16,
    },
}

# Medium-difficulty parameter ranges from the ipynb generator (validated).
T_MAX = 24.0          # planning horizon (h)
V_ECO = 10.0          # economic speed (knots)
W_PENALTY = 10000.0   # per-unexecuted-task penalty
BIG_M = 1000.0        # large constant from MILP linearization

SERVICE_TIME_RANGE = (0.8, 1.6)     # T_s ~ U(0.8, 1.6) hr  (loose)
SERVICE_TIME_RANGE_HARD = (2.0, 3.5) # hard mode
TIME_WINDOW_WIDTH = 2.5             # b_s - a_s (hr)
TIME_WINDOW_WIDTH_HARD = 0.8        # narrow window forces conflicts
MULTI_TUG_PROB_BY_N = {            # P(task needs ≥2 tugs), depends on n
    "small":  0.25,
    "medium": 0.20,
    "large":  0.18,
}
MULTI_TUG_PROB_HARD = 0.55          # >50% of tasks need collaboration in hard mode
# HP requirements by collaborative count (kW)
HP_REQ_BY_NTUGS = {
    1: (3000, 4500),
    2: (6000, 9000),
    3: (10000, 13000),
}
HP_REQ_BY_NTUGS_HARD = {
    1: (4200, 5500),      # so single tug barely covers / often needs 2
    2: (8500, 11000),
    3: (13000, 16000),
}
TUG_HP_RANGE   = (4500, 6001)       # HP_k (kW) — uniform int
TUG_ALPHA_RANGE = (0.15, 0.18)
TUG_BETA_RANGE  = (0.06, 0.08)
FUEL_MULTIPLIER = 1.8               # loose
FUEL_MULTIPLIER_HARD = 0.85         # binds at optimum

# Default output directory
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "CO-Bench" / "Multi-Tugboat Routing and Scheduling Problem"


# ============================== Generation ==============================

def _multi_tug_prob(n: int) -> float:
    if n <= 50:
        return MULTI_TUG_PROB_BY_N["small"]
    if n <= 150:
        return MULTI_TUG_PROB_BY_N["medium"]
    return MULTI_TUG_PROB_BY_N["large"]


def _base_fuel(n: int) -> int:
    """Base fuel capacity (kg) before fuel_multiplier scaling."""
    if n <= 50:
        return 12000
    if n <= 100:
        return 18000
    if n <= 200:
        return 25000
    return 35000


def _gen_tasks(rng: random.Random, n: int, area: float, hard: bool = False) -> list[dict]:
    """Generate task parameters. hard=True tightens service/window/HP."""
    tasks = []
    multi_tug_p = MULTI_TUG_PROB_HARD if hard else _multi_tug_prob(n)
    svc_range = SERVICE_TIME_RANGE_HARD if hard else SERVICE_TIME_RANGE
    tw_width = TIME_WINDOW_WIDTH_HARD if hard else TIME_WINDOW_WIDTH
    hp_table = HP_REQ_BY_NTUGS_HARD if hard else HP_REQ_BY_NTUGS
    total_time_span = T_MAX - 2.0
    time_slot = total_time_span / n

    for s in range(1, n + 1):
        sx = rng.uniform(-area / 2, area / 2)
        sy = rng.uniform(-area / 2, area / 2)
        angle = rng.uniform(0, 2 * math.pi)
        dist  = rng.uniform(3.0, 8.0)
        ex = sx + dist * math.cos(angle)
        ey = sy + dist * math.sin(angle)

        T_s = rng.uniform(*svc_range)

        earliest = 0.5 + (s - 1) * time_slot
        max_latest = T_MAX - T_s - 0.5
        latest = min(earliest + tw_width, max_latest)
        earliest = max(0.5, min(earliest, max_latest - 0.5))
        if latest <= earliest:
            latest = max_latest
            earliest = max(0.5, latest - tw_width)

        if rng.random() < multi_tug_p:
            n_tugs = rng.choices([2, 3], weights=[0.8, 0.2], k=1)[0]
        else:
            n_tugs = 1

        hp_lo, hp_hi = hp_table[n_tugs]
        hp_req = rng.randint(hp_lo, hp_hi)

        tasks.append({
            "id":           s,
            "start_loc":    (sx, sy),
            "end_loc":      (ex, ey),
            "service_time": round(T_s, 2),
            "tw_lower":     round(earliest, 2),
            "tw_upper":     round(latest, 2),
            "max_tugs":     n_tugs,
            "min_hp":       hp_req,
        })
    return tasks


def _gen_tugs(rng: random.Random, K: int, n: int, hard: bool = False) -> list[dict]:
    """Generate tugboat parameters."""
    base = _base_fuel(n)
    mult = FUEL_MULTIPLIER_HARD if hard else FUEL_MULTIPLIER
    capacity = int(base * mult)
    return [
        {
            "hp":       rng.randint(*TUG_HP_RANGE),
            "fuel_cap": capacity,
            "alpha":    round(rng.uniform(*TUG_ALPHA_RANGE), 3),
            "beta":     round(rng.uniform(*TUG_BETA_RANGE),  3),
        }
        for _ in range(K)
    ]


def _gen_time_matrix(tasks: list[dict], base_loc: tuple = (0.0, 0.0)) -> dict:
    """Sparse dict: keys 'i_j' (str), values travel-time (hours).

    Includes:
      - '0_j'   depot → task j entrance        j ∈ {1..n}
      - 'i_j'   task i exit → task j entrance  i, j ∈ {1..n}, i ≠ j
      - 'i_n+1' task i exit → depot            i ∈ {1..n}
    """
    n = len(tasks)
    bx, by = base_loc
    tm = {}

    def d(p, q):
        return math.hypot(p[0] - q[0], p[1] - q[1])

    for j in range(1, n + 1):
        sj = tasks[j - 1]["start_loc"]
        tm[f"0_{j}"] = round(d((bx, by), sj) / V_ECO, 4)
    for i in range(1, n + 1):
        ei = tasks[i - 1]["end_loc"]
        for j in range(1, n + 1):
            if i == j:
                continue
            sj = tasks[j - 1]["start_loc"]
            tm[f"{i}_{j}"] = round(d(ei, sj) / V_ECO, 4)
    end_depot = n + 1
    for i in range(1, n + 1):
        ei = tasks[i - 1]["end_loc"]
        tm[f"{i}_{end_depot}"] = round(d(ei, (bx, by)) / V_ECO, 4)
    return tm


# ============================== Sanity check ==============================

def sanity_check(inst: dict, min_servable_frac: float = 0.5) -> tuple[bool, str | None]:
    """Verify the instance is non-trivially solvable.

    Hard requirements:
      - max H_s^min ≤ sum of top-Γₘₐₓ tug HPs (else no task can be served at all)
      - ≥ min_servable_frac of tasks have time windows + service time that fit
        inside T_max with a feasible depot-roundtrip (so the instance isn't
        degenerate where every task must be skipped).

    Tasks whose time window is too tight for any tug to reach + service are
    LEGAL (they just go unexecuted, paying W per task). We tolerate up to
    (1 - frac) of these per instance.
    """
    n = inst["num_tasks"]
    K = inst["num_tugboats"]
    hp_reqs = inst["task_min_horsepower"]
    max_tugs = inst["task_max_tugs"]
    tug_hps = inst["tugboat_horsepower"]
    tm = inst["time_matrix"]
    aS = inst["task_time_window_lower"]
    bS = inst["task_time_window_upper"]
    TS = inst["task_service_time"]
    T_max = inst["planning_horizon"]

    # 1) Every task must in principle be servable by SOME tug combo.
    sorted_hps = sorted(tug_hps, reverse=True)
    for s in range(n):
        gamma = max_tugs[s]
        if gamma > K:
            return False, f"task {s+1} max_tugs={gamma} > num_tugboats={K}"
        top_combo = sum(sorted_hps[:gamma])
        if hp_reqs[s] > top_combo:
            return False, (
                f"task {s+1} HP req {hp_reqs[s]} > best top-{gamma} tug combo "
                f"{top_combo}")

    # 2) Time-window servability count.
    end_depot = n + 1
    servable = 0
    for s in range(n):
        t_depot_to = tm[f"0_{s+1}"]
        t_back     = tm[f"{s+1}_{end_depot}"]
        latest_finish = bS[s] + TS[s]
        if (t_depot_to <= bS[s] + 1e-6
                and latest_finish + t_back <= T_max + 1e-6
                and aS[s] >= t_depot_to - 1e-6):
            servable += 1
        elif aS[s] < t_depot_to <= bS[s] and latest_finish <= T_max:
            # The window allows starting at some τ ≥ t_depot_to.
            servable += 1
    if servable < max(1, int(n * min_servable_frac)):
        return False, (
            f"only {servable}/{n} tasks have feasible time windows; "
            f"need ≥ {int(n * min_servable_frac)}")
    return True, None


# ============================== Build one instance ==============================

def build(size_class: str = None, *, n: int = None, K: int = None,
          area: float = None, seed: int = 0, hard: bool = False) -> dict:
    """Build one MTRSP instance dict. hard=True activates tightened params."""
    if size_class is not None:
        p = SIZE_PRESETS[size_class]
        n, K, area = p["n"], p["K"], p["area"]
    elif None in (n, K, area):
        raise ValueError("either size_class or full (n, K, area) required")

    rng = random.Random(seed)
    tasks = _gen_tasks(rng, n, area, hard=hard)
    tugs  = _gen_tugs(rng, K, n, hard=hard)
    tm    = _gen_time_matrix(tasks)

    inst = {
        "num_tasks":              n,
        "num_tugboats":           K,
        "task_max_tugs":          [t["max_tugs"]     for t in tasks],
        "task_min_horsepower":    [t["min_hp"]       for t in tasks],
        "task_time_window_lower": [t["tw_lower"]     for t in tasks],
        "task_time_window_upper": [t["tw_upper"]     for t in tasks],
        "task_service_time":      [t["service_time"] for t in tasks],
        "tugboat_horsepower":     [g["hp"]        for g in tugs],
        "tugboat_fuel_capacity":  [g["fuel_cap"]  for g in tugs],
        "tugboat_alpha":          [g["alpha"]     for g in tugs],
        "tugboat_beta":           [g["beta"]      for g in tugs],
        "time_matrix":            tm,
        "big_M":                  BIG_M,
        "planning_horizon":       T_MAX,
        "penalty_weight":         W_PENALTY,
    }
    return inst


def build_with_retry(size_class: str = None, *, seed: int = 0,
                     max_retries: int = 8, hard: bool = False,
                     verbose: bool = False, **kwargs) -> dict:
    """Sanity-checked build. Re-seed up to max_retries on failure."""
    last_reason = None
    sanity_thresh = 0.35 if hard else 0.5
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
    "num_tasks", "num_tugboats",
    "task_max_tugs", "task_min_horsepower",
    "task_time_window_lower", "task_time_window_upper", "task_service_time",
    "tugboat_horsepower", "tugboat_fuel_capacity",
    "tugboat_alpha", "tugboat_beta",
    "time_matrix",
    "big_M", "planning_horizon", "penalty_weight",
]


def write_instance(path: Path, inst: dict, header_comment: str = ""):
    lines = []
    if header_comment:
        lines.append(f"# {header_comment}")
    lines.append(
        f"# n={inst['num_tasks']} K={inst['num_tugboats']} "
        f"T_max={inst['planning_horizon']} W={inst['penalty_weight']}"
    )
    lines.append("")
    for key in PARAM_ORDER:
        if key not in inst:
            raise KeyError(f"instance missing required key {key!r}")
        val = inst[key]
        lines.append(f"{key} = {val!r}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ============================== CLI ==============================

def _generate_tier(tier: str, out_base: Path, verbose: bool,
                     hard: bool = False) -> list[Path]:
    """Generate all instances for one tier per BUILD_PLAN."""
    plan = BUILD_PLAN[tier]
    suffix = "_hard" if hard else ""
    tier_dir = out_base / (tier + suffix)
    tier_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    seed_base = 1 if tier == "train" else 10001
    for profile, count in plan.items():
        for k in range(count):
            seed = seed_base + k
            inst = build_with_retry(size_class=profile, seed=seed, hard=hard,
                                     verbose=verbose)
            fname = f"{profile}_{seed:05d}.txt"
            path = tier_dir / fname
            write_instance(path, inst,
                           header_comment=f"MTRSP {tier}{suffix} | {profile} seed={seed}")
            written.append(path)
            if verbose:
                print(f"  wrote {tier}{suffix}/{fname}")
    return written


def main():
    p = argparse.ArgumentParser(
        description=("Generate MTRSP instances under "
                     "data/CO-Bench/Multi-Tugboat Routing and Scheduling Problem/."),
    )
    group = p.add_mutually_exclusive_group()
    group.add_argument("--tier", choices=["train", "test", "all"],
                       help="Build all instances of a tier per BUILD_PLAN. "
                            "train=128 instances, test=100 instances, all=228.")
    group.add_argument("--profile", choices=list(SIZE_PRESETS.keys()),
                       help="Build a single profile (use --count and --seed-base).")
    group.add_argument("--custom", nargs=3, type=int, metavar=("N", "K", "AREA"),
                       help="Custom dimensions; T_max=24 hr fixed.")
    p.add_argument("--count",     type=int, default=3,
                   help="Instances for --profile mode (default 3).")
    p.add_argument("--seed-base", type=int, default=1,
                   help="First seed for --profile (default 1).")
    p.add_argument("--seed",      type=int, default=42,
                   help="Seed for --custom (default 42).")
    p.add_argument("--out", default=str(DEFAULT_OUTPUT_DIR),
                   help=f"Output base directory (default: {DEFAULT_OUTPUT_DIR}).")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--hard", action="store_true",
                    help="Tighten params (binds fuel, narrow windows, more collab).")
    args = p.parse_args()

    out_base = Path(args.out)
    out_base.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    suffix = "_hard" if args.hard else ""

    if args.tier:
        tiers = ["train", "test"] if args.tier == "all" else [args.tier]
        for t in tiers:
            written += _generate_tier(t, out_base, args.verbose, hard=args.hard)
    elif args.custom:
        n, K, area = args.custom
        inst = build_with_retry(n=n, K=K, area=float(area), hard=args.hard,
                                seed=args.seed, verbose=args.verbose)
        fname = f"custom_n{n}_K{K}_area{area}_seed{args.seed:03d}.txt"
        out_sub = out_base / f"pilot{suffix}"
        out_sub.mkdir(parents=True, exist_ok=True)
        path = out_sub / fname
        write_instance(path, inst, header_comment=f"MTRSP custom{suffix} seed={args.seed}")
        written.append(path)
    else:
        prof = args.profile or "n10_K5_T24"
        out_sub = out_base / f"pilot{suffix}"
        out_sub.mkdir(parents=True, exist_ok=True)
        for k in range(args.count):
            seed = args.seed_base + k
            inst = build_with_retry(size_class=prof, seed=seed, hard=args.hard,
                                     verbose=args.verbose)
            fname = f"{prof}_{seed:05d}.txt"
            path = out_sub / fname
            write_instance(path, inst, header_comment=f"MTRSP {prof}{suffix} seed={seed}")
            written.append(path)

    print(f"Generated {len(written)} instance(s) under {out_base}")


if __name__ == "__main__":
    main()
