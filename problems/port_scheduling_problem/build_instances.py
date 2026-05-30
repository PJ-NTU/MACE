"""Port Scheduling Problem (PSP) instance generator.

Generates PSP instances in `param = value` text format under
`data/CO-Bench/Port Scheduling Problem/`. Each generated file holds ONE
instance — `config.py:load_data(path)` returns a 1-element list of dicts.

Three preset size classes (n vessels / J berths / K tugs / T periods / H_max):

    small  :  n=5    J=2   K=3    T=24   H_max=2
    medium :  n=20   J=5   K=8    T=48   H_max=2
    large  :  n=200  J=30  K=60   T=96   H_max=3

Examples:
    # 3 small instances, seeds 42,43,44, written to default data folder
    python -m problems.cobench.psp.build_instances --sizes small --count 3 --seed-base 42

    # 3 of each size class
    python -m problems.cobench.psp.build_instances --sizes small medium large --count 3

    # Ad-hoc custom (n J K T)
    python -m problems.cobench.psp.build_instances --custom 8 3 5 24 --seed 7

Design choices (in contrast to the old ipynb generator):

  * No duplicate parameter writes (drop the Greek-symbol aliases).
  * Transparent λ / M derivation — every constant has a one-line rationale.
  * Pre-write sanity check: max-vessel-size ≤ max-berth-cap, max-HP ≤ top-Hmax
    tug-combo, every vessel time window fits a complete (in,berth,out) triple
    before T. On failure, retry with a different seed; abort after 5 tries.
  * Per-period parameters (durations, service times, eps) scale smoothly with T
    instead of via if/elif T-ladders.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np


# ============================== Constants ==============================

SIZE_PRESETS = {
    # Tiny profiles (legacy)
    "12_2_3_T12":     dict(n=12,  J=2,  K=3,   T=12, H_max=2),
    "20_3_4_T18":     dict(n=20,  J=3,  K=4,   T=18, H_max=2),
    # ───── Small tier (literature benchmark) ─────
    "20_2_3_T12":     dict(n=20,  J=2,  K=3,   T=12, H_max=2),
    "40_3_4_T12":     dict(n=40,  J=3,  K=4,   T=12, H_max=2),
    "60_5_8_T12":     dict(n=60,  J=5,  K=8,   T=12, H_max=2),
    "80_10_15_T48":   dict(n=80,  J=10, K=15,  T=48, H_max=2),
    "100_15_30_T24":  dict(n=100, J=15, K=30,  T=24, H_max=2),
    "100_15_30_T48":  dict(n=100, J=15, K=30,  T=48, H_max=2),
    # ───── Medium tier ─────
    "150_20_40_T24":  dict(n=150, J=20, K=40,  T=24, H_max=2),
    "150_20_40_T48":  dict(n=150, J=20, K=40,  T=48, H_max=2),
    "200_20_40_T24":  dict(n=200, J=20, K=40,  T=24, H_max=2),
    "200_20_40_T48":  dict(n=200, J=20, K=40,  T=48, H_max=2),
    "250_20_50_T24":  dict(n=250, J=20, K=50,  T=24, H_max=2),
    "250_20_50_T48":  dict(n=250, J=20, K=50,  T=48, H_max=2),
    # ───── Large tier ─────
    "300_25_60_T24":  dict(n=300, J=25, K=60,  T=24, H_max=3),
    "300_25_60_T48":  dict(n=300, J=25, K=60,  T=48, H_max=3),
    "400_30_80_T24":  dict(n=400, J=30, K=80,  T=24, H_max=3),
    "400_30_80_T48":  dict(n=400, J=30, K=80,  T=48, H_max=3),
    "500_40_100_T24": dict(n=500, J=40, K=100, T=24, H_max=3),
    "500_40_100_T48": dict(n=500, J=40, K=100, T=48, H_max=3),
}

# Build targets per tier — counts sum to 128 (train) + 100 (test) = 228 total.
BUILD_PLAN = {
    "train": {
        "20_2_3_T12":     43,
        "40_3_4_T12":     43,
        "60_5_8_T12":     42,
    },
    "test": {
        "80_10_15_T48":   17,
        "100_15_30_T48":  17,
        "150_20_40_T48":  17,
        "250_20_50_T48":  17,
        "300_25_60_T48":  16,
        "500_40_100_T48": 16,
    },
}

VESSEL_TYPE_CONFIG = {
    "container": {"ratio": 0.35, "size_range": (3, 5), "priority_range": (2.2, 3.0)},
    "bulk":      {"ratio": 0.30, "size_range": (3, 5), "priority_range": (1.8, 2.5)},
    "tanker":    {"ratio": 0.25, "size_range": (2, 4), "priority_range": (2.0, 2.8)},
    "general":   {"ratio": 0.10, "size_range": (1, 3), "priority_range": (1.2, 2.0)},
}

# Required HP centers by vessel size — bigger vessel → more horsepower
HP_CENTER_BY_SIZE = {1: 1000, 2: 1700, 3: 2700, 4: 4000, 5: 5200}
# Hard mode: raise HP centers ~1.5×, forces more frequent collaboration
HP_CENTER_BY_SIZE_HARD = {1: 1500, 2: 2600, 3: 4000, 4: 6000, 5: 7800}

# Default output directory (CO-Bench data folder for this problem)
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "CO-Bench" / "Port Scheduling Problem"


# ============================== Time profile ==============================

def time_period_profile(T: int, hard: bool = False) -> dict:
    """Per-period time parameter ranges that scale smoothly with T.

    hard=True tightens windows + lengthens durations to make the schedule pack
    much closer to capacity.
    """
    if hard:
        # Longer berth duration, longer service, tighter window
        duration_range = (max(2, T // 8), max(3, T // 4))
        service_range  = (max(1, T // 12), max(2, T // 6))
        window_range   = (0, max(1, T // 24))   # ETA ±0 or ±1 only
        prep_time      = max(1, T // 24)
        eps_time       = max(1, T // 24)        # tighter slack
    else:
        duration_range = (max(1, T // 24), max(2, T // 8))
        service_range  = (max(1, T // 24), max(1, T // 12))
        window_range   = (max(1, T // 24), max(2, T // 8))
        prep_time      = max(1, T // 24)
        eps_time       = max(2, T // 12)
    return {
        "duration_range": duration_range,
        "service_range":  service_range,
        "window_range":   window_range,
        "prep_time":      prep_time,
        "eps_time":       eps_time,
    }


# ============================== Generation ==============================

def _split_counts(n: int) -> list[tuple[str, int]]:
    """Split n vessels across types respecting ratios; last type absorbs remainder."""
    out = []
    remaining = n
    types = list(VESSEL_TYPE_CONFIG.items())
    for i, (vt, cfg) in enumerate(types):
        if i == len(types) - 1:
            count = remaining
        else:
            count = int(round(n * cfg["ratio"]))
            count = min(count, remaining)
        out.append((vt, count))
        remaining -= count
    return out


def _gen_etas(rng: random.Random, n: int, T: int) -> list[int]:
    """Uniform-random ETAs in [0, T-1], sorted. Handles n > T (collisions OK)."""
    etas = sorted(rng.randint(0, T - 1) for _ in range(n))
    return etas


def _gen_vessels(rng: random.Random, n: int, T: int, prof: dict,
                  hard: bool = False) -> list[dict]:
    """Generate n vessel dicts with size / priority / time params / HP req."""
    vessels = []
    for vt, count in _split_counts(n):
        cfg = VESSEL_TYPE_CONFIG[vt]
        for _ in range(count):
            vessels.append({
                "type":     vt,
                "size":     rng.randint(*cfg["size_range"]),
                "priority": round(rng.uniform(*cfg["priority_range"]), 2),
            })
    rng.shuffle(vessels)

    hp_table = HP_CENTER_BY_SIZE_HARD if hard else HP_CENTER_BY_SIZE
    etas = _gen_etas(rng, n, T)
    for i, v in enumerate(vessels):
        v["eta"]     = etas[i]
        v["D"]       = rng.randint(*prof["duration_range"])
        v["tau_in"]  = rng.randint(*prof["service_range"])
        v["tau_out"] = rng.randint(*prof["service_range"])
        v["early"]   = rng.randint(*prof["window_range"])
        v["late"]    = rng.randint(*prof["window_range"])
        hp_center    = hp_table.get(v["size"], 2500)
        v["hp_req"]  = rng.randint(int(hp_center * 0.85), int(hp_center * 1.15))
    return vessels


def _gen_berths(rng: random.Random, J: int, max_vessel_size: int) -> list[int]:
    """Tiered capacity. Ensure ≥1 berth can host the largest vessel."""
    high = max(1, J // 3)
    mid  = max(1, J // 3)
    low  = max(0, J - high - mid)
    caps = []
    caps += [max(max_vessel_size, 5)] * high
    caps += [rng.choice([3, 4]) for _ in range(mid)]
    caps += [rng.choice([2, 3]) for _ in range(low)]
    rng.shuffle(caps)
    return caps


def _gen_tugs(rng: random.Random, K: int,
                hard: bool = False) -> tuple[list[int], list[float]]:
    """Tiered HP. hard=True bumps HP ranges proportionally to vessel HP."""
    high = max(1, K // 3)
    mid  = max(1, K // 3)
    low  = max(0, K - high - mid)
    if hard:
        hi_r = (4500, 7000); md_r = (2500, 4500); lo_r = (1500, 2800)
    else:
        hi_r = (3000, 5000); md_r = (1500, 3500); lo_r = (800,  1800)
    hps = []
    hps += [rng.randint(*hi_r) for _ in range(high)]
    hps += [rng.randint(*md_r) for _ in range(mid)]
    hps += [rng.randint(*lo_r) for _ in range(low)]
    rng.shuffle(hps)

    costs = []
    for hp in hps:
        if   hp >= 4000: costs.append(round(rng.uniform(80, 130), 1))
        elif hp >= 3000: costs.append(round(rng.uniform(60, 100), 1))
        elif hp >= 2000: costs.append(round(rng.uniform(40,  75), 1))
        elif hp >= 1500: costs.append(round(rng.uniform(30,  55), 1))
        else:            costs.append(round(rng.uniform(20,  40), 1))
    return hps, costs


def _derive_costs(rng: random.Random, vessels: list[dict],
                  tug_costs: list[float], H_max: int,
                  hard: bool = False) -> tuple[list[float], float]:
    """Compute β/γ per vessel, then λ weights and M penalty, all transparently.

    Returns: (objective_weights = [λ₁, λ₂, λ₃, λ₄], M).
    Side effect: writes v["beta"] and v["gamma"] into each vessel dict.

    Recipe (NO magic constants):
      1. β per vessel — tiered by size: small=22, medium=35, large=50 (±20%).
      2. γ per vessel — 30%-50% of β.
      3. Estimate typical Z_k per assigned vessel from averages of (α, β, γ,
         port_time, tug_cost). port_time ≈ τ_in + D + τ_out;
         eta_dev ≈ 2 (half avg window); tugs/service ≈ 2.
      4. M = 5 × typical_service_cost, rounded to 100.  Rationale: skipping a
         vessel must be ~5× as bad as serving it normally.
      5. λ_k = 1 / total_z_k, normalized to sum to 1.  Each Z component
         contributes equally to objective scale at the typical operating point.
    """
    # Step 1+2: per-vessel β and γ
    for v in vessels:
        if   v["size"] >= 4: base = 50
        elif v["size"] >= 3: base = 35
        else:                base = 22
        v["beta"]  = round(base * rng.uniform(0.8, 1.2), 1)
        v["gamma"] = round(v["beta"] * rng.uniform(0.3, 0.5), 1)

    n = len(vessels)
    avg_alpha = float(np.mean([v["priority"] for v in vessels]))
    avg_beta  = float(np.mean([v["beta"]     for v in vessels]))
    avg_gamma = float(np.mean([v["gamma"]    for v in vessels]))
    avg_D     = float(np.mean([v["D"]        for v in vessels]))
    avg_tin   = float(np.mean([v["tau_in"]   for v in vessels]))
    avg_tout  = float(np.mean([v["tau_out"]  for v in vessels]))
    avg_tugc  = float(np.mean(tug_costs))

    typical_port_time = avg_tin + avg_D + avg_tout
    typical_eta_dev   = 2.0
    typical_tugs_per_service = min(H_max, 2)

    z2_per = avg_alpha * avg_beta  * typical_port_time
    z3_per = avg_alpha * avg_gamma * typical_eta_dev
    z4_per = avg_tugc  * (avg_tin + avg_tout) * typical_tugs_per_service

    typical_cost = z2_per + z3_per + z4_per

    # Step 4: M = 5× typical_cost (LOOSE) or 2× (HARD) — hard mode lowers
    # the drop-vs-serve threshold so optimum genuinely drops some vessels.
    M_factor = 2 if hard else 5
    M = round(typical_cost * M_factor / 100) * 100
    if M <= 0:
        M = 100

    # Step 5: λ weights so each Z_k contributes equally
    typical_unserved_ratio = 0.25 if hard else 0.05  # expect ~25% / 5% unserved
    totals = [
        M * avg_alpha * n * typical_unserved_ratio,
        z2_per * n * (1 - typical_unserved_ratio),
        z3_per * n * (1 - typical_unserved_ratio),
        z4_per * n * (1 - typical_unserved_ratio),
    ]
    raw = [1.0 / max(t, 1e-6) for t in totals]
    s = sum(raw)
    lambdas = [round(x / s, 4) for x in raw]
    return lambdas, float(M)


# ============================== Sanity check ==============================

def sanity_check(inst: dict, min_servable_frac: float = 0.5) -> tuple[bool, str | None]:
    """Verify the instance is non-trivially solvable.

    Hard requirements:
      - max vessel size ≤ max berth capacity (else no vessel can dock)
      - max HP req ≤ sum of top-H_max tug HPs (else no vessel can be towed)
      - at least `min_servable_frac` of vessels have a time window that admits
        a feasible (in,berth,out) triple before T (else instance is degenerate)

    Vessels whose ETA is too close to T to fit a full triple are LEGAL — they
    just must be left unassigned (paying Z₁). We tolerate up to (1 - frac) of
    these per instance; the heuristic's job is to decide who to drop.
    """
    sizes      = inst["vessel_sizes"]
    berth_caps = inst["berth_capacities"]
    hp_reqs    = inst["vessel_horsepower_requirements"]
    tug_hps    = inst["tugboat_horsepower"]
    H_max      = inst["max_tugboats_per_service"]
    T          = inst["time_periods"]
    n          = inst["vessel_num"]

    if max(sizes) > max(berth_caps):
        return False, f"max vessel size {max(sizes)} > max berth capacity {max(berth_caps)}"

    top_combo = sum(sorted(tug_hps, reverse=True)[:H_max])
    if max(hp_reqs) > top_combo:
        return False, (f"max HP req {max(hp_reqs)} > best top-{H_max} tug combo "
                       f"{top_combo}")

    servable = 0
    for i in range(n):
        eta     = inst["vessel_etas"][i]
        early   = inst["vessel_early_limits"][i]
        late    = inst["vessel_late_limits"][i]
        tau_in  = inst["vessel_inbound_service_times"][i]
        D       = inst["vessel_durations"][i]
        tau_out = inst["vessel_outbound_service_times"][i]

        earliest_in = max(0, eta - early)
        latest_in   = min(T - tau_in - D - tau_out, eta + late)
        if earliest_in <= latest_in:
            servable += 1

    if servable < max(1, int(n * min_servable_frac)):
        return False, (f"only {servable}/{n} vessels have feasible time windows; "
                       f"need ≥ {int(n * min_servable_frac)}")
    return True, None


# ============================== Build one instance ==============================

def build(size_class: str = None, *, n: int = None, J: int = None, K: int = None,
          T: int = None, H_max: int = None, seed: int = 0,
          hard: bool = False) -> dict:
    """Build one instance dict. Either supply `size_class` or all of n/J/K/T/H_max.
    hard=True tightens time windows, raises HP, lowers penalty M, longer durations."""
    if size_class is not None:
        p = SIZE_PRESETS[size_class]
        n, J, K, T, H_max = p["n"], p["J"], p["K"], p["T"], p["H_max"]
    elif None in (n, J, K, T, H_max):
        raise ValueError("either size_class or full (n,J,K,T,H_max) required")

    rng = random.Random(seed)
    prof = time_period_profile(T, hard=hard)

    vessels = _gen_vessels(rng, n, T, prof, hard=hard)
    berths  = _gen_berths(rng, J, max(v["size"] for v in vessels))
    tug_hps, tug_costs = _gen_tugs(rng, K, hard=hard)
    lambdas, M = _derive_costs(rng, vessels, tug_costs, H_max, hard=hard)

    inst = {
        "vessel_num":  n,
        "berth_num":   J,
        "tugboat_num": K,
        "time_periods": T,
        "vessel_sizes":                  [v["size"]     for v in vessels],
        "vessel_etas":                   [v["eta"]      for v in vessels],
        "vessel_durations":              [v["D"]        for v in vessels],
        "vessel_inbound_service_times":  [v["tau_in"]   for v in vessels],
        "vessel_outbound_service_times": [v["tau_out"]  for v in vessels],
        "vessel_priority_weights":       [v["priority"] for v in vessels],
        "vessel_waiting_costs":          [v["beta"]     for v in vessels],
        "vessel_jit_costs":              [v["gamma"]    for v in vessels],
        "vessel_horsepower_requirements":[v["hp_req"]   for v in vessels],
        "vessel_early_limits":           [v["early"]    for v in vessels],
        "vessel_late_limits":            [v["late"]     for v in vessels],
        "berth_capacities":   berths,
        "tugboat_horsepower": tug_hps,
        "tugboat_costs":      tug_costs,
        "inbound_preparation_time":  prof["prep_time"],
        "outbound_preparation_time": prof["prep_time"],
        "max_tugboats_per_service":  H_max,
        "time_constraint_tolerance": prof["eps_time"],
        "penalty_parameter":         M,
        "objective_weights":         lambdas,
    }
    return inst


def build_with_retry(size_class: str = None, *, seed: int = 0,
                     max_retries: int = 8, hard: bool = False,
                     verbose: bool = False, **kwargs) -> dict:
    """Sanity-checked build. Re-seed up to max_retries on failure."""
    last_reason = None
    thresh = 0.4 if hard else 0.5
    for attempt in range(max_retries):
        s = seed + attempt * 100003
        inst = build(size_class, seed=s, hard=hard, **kwargs)
        ok, reason = sanity_check(inst, min_servable_frac=thresh)
        if ok:
            return inst
        last_reason = reason
        if verbose:
            print(f"  retry attempt={attempt} seed={s}: {reason}")
    raise RuntimeError(f"sanity check failed after {max_retries} attempts: {last_reason}")


# ============================== Write to file ==============================

PARAM_ORDER = [
    "vessel_num", "berth_num", "tugboat_num", "time_periods",
    "vessel_sizes", "vessel_etas", "vessel_durations",
    "vessel_inbound_service_times", "vessel_outbound_service_times",
    "vessel_priority_weights", "vessel_waiting_costs", "vessel_jit_costs",
    "vessel_horsepower_requirements",
    "vessel_early_limits", "vessel_late_limits",
    "berth_capacities",
    "tugboat_horsepower", "tugboat_costs",
    "inbound_preparation_time", "outbound_preparation_time",
    "max_tugboats_per_service", "time_constraint_tolerance",
    "penalty_parameter", "objective_weights",
]


def write_instance(path: Path, inst: dict, header_comment: str = ""):
    lines = []
    if header_comment:
        lines.append(f"# {header_comment}")
    lines.append(
        f"# n={inst['vessel_num']} J={inst['berth_num']} K={inst['tugboat_num']} "
        f"T={inst['time_periods']} H_max={inst['max_tugboats_per_service']}"
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
    """Generate all instances for one tier per BUILD_PLAN."""
    # In hard mode we use the literature "Small" tier (Gurobi can solve some to
    # optimality within 30 min, MILP CBC at 60s can verify the small ones).
    # Counts match the standard 128 train / 100 test target.
    if hard:
        if tier == "train":
            # Train stays Small tier (heur is trained at moderate scales)
            plan = {
                "20_2_3_T12": 22, "40_3_4_T12": 22, "60_5_8_T12": 22,
                "80_10_15_T48": 21, "100_15_30_T24": 21, "100_15_30_T48": 20,
            }  # = 128
        else:
            # Test spans all 18 literature-benchmark profiles (Small/Medium/Large)
            plan = {
                # Small
                "20_2_3_T12": 5,    "40_3_4_T12": 5,    "60_5_8_T12": 5,
                "80_10_15_T48": 5,  "100_15_30_T24": 5, "100_15_30_T48": 6,
                # Medium
                "150_20_40_T24": 6, "150_20_40_T48": 6,
                "200_20_40_T24": 6, "200_20_40_T48": 6,
                "250_20_50_T24": 6, "250_20_50_T48": 6,
                # Large
                "300_25_60_T24": 6, "300_25_60_T48": 6,
                "400_30_80_T24": 5, "400_30_80_T48": 5,
                "500_40_100_T24": 5,"500_40_100_T48": 6,
            }  # = 100
    else:
        plan = BUILD_PLAN[tier]
    suffix = "_hard" if hard else ""
    tier_dir = out_base / (tier + suffix)
    tier_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for profile, count in plan.items():
        for k in range(count):
            seed = k + 1
            inst = build_with_retry(size_class=profile, seed=seed, hard=hard,
                                     verbose=verbose)
            fname = f"{profile}_{seed:03d}.txt"
            path = tier_dir / fname
            write_instance(path, inst,
                           header_comment=f"PSP {tier}{suffix} | {profile} seed={seed}")
            written.append(path)
            if verbose:
                print(f"  wrote {tier}{suffix}/{fname}")
    return written


def main():
    p = argparse.ArgumentParser(
        description="Generate PSP instances under data/CO-Bench/Port Scheduling Problem/.",
    )
    group = p.add_mutually_exclusive_group()
    group.add_argument("--tier", choices=["train", "test", "all"],
                       help="Build all instances of a tier per BUILD_PLAN. "
                            "train=128 instances, test=100 instances, all=228.")
    group.add_argument("--profile", choices=list(SIZE_PRESETS.keys()),
                       help="Build a single profile (use --count and --seed-base).")
    group.add_argument("--custom", nargs=4, type=int, metavar=("N", "J", "K", "T"),
                       help="Custom dimensions (--seed for the single seed).")
    p.add_argument("--count",     type=int, default=3,
                   help="Instances for --profile mode (default 3).")
    p.add_argument("--seed-base", type=int, default=1,
                   help="First seed for --profile (default 1).")
    p.add_argument("--seed",      type=int, default=42,
                   help="Seed for --custom (default 42).")
    p.add_argument("--h-max",     type=int, default=2,
                   help="H_max for --custom only (default 2).")
    p.add_argument("--out", default=str(DEFAULT_OUTPUT_DIR),
                   help=f"Output base directory (default: {DEFAULT_OUTPUT_DIR}).")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--hard", action="store_true",
                    help="Tighten params (windows, HP, M, durations); switches to "
                         "smaller-profile BUILD_PLAN for MILP verification.")
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
        n, J, K, T = args.custom
        inst = build_with_retry(n=n, J=J, K=K, T=T, H_max=args.h_max, hard=args.hard,
                                seed=args.seed, verbose=args.verbose)
        fname = f"custom_n{n}_J{J}_K{K}_T{T}_seed{args.seed:03d}.txt"
        out_sub = out_base / f"pilot{suffix}"
        out_sub.mkdir(parents=True, exist_ok=True)
        path = out_sub / fname
        write_instance(path, inst, header_comment=f"PSP custom{suffix} seed={args.seed}")
        written.append(path)
    else:
        prof = args.profile or "20_2_3_T12"
        out_sub = out_base / f"pilot{suffix}"
        out_sub.mkdir(parents=True, exist_ok=True)
        for k in range(args.count):
            seed = args.seed_base + k
            inst = build_with_retry(size_class=prof, seed=seed, hard=args.hard,
                                     verbose=args.verbose)
            fname = f"{prof}_{seed:03d}.txt"
            path = out_sub / fname
            write_instance(path, inst, header_comment=f"PSP {prof}{suffix} seed={seed}")
            written.append(path)

    print(f"Generated {len(written)} instance(s) under {out_base}")


if __name__ == "__main__":
    main()
