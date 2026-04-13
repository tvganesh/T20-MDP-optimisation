#!/usr/bin/env python3
"""
Bowling Order Optimisation — Defending Team (Gujarat Titans)
GT vs Punjab Kings, 31 March 2026  |  Intervention: P Simran Singh dismissed
(ball 2nd.9.3, caught off Rashid Khan)

Match context:
  GT first innings: 162
  PBKS target: 163
  At intervention: PBKS 83/2, needing 80 more runs

Rashid Khan is mid-over at the wicket fall (3 balls remaining in over 9).
He is committed to finishing that over.  The optimisation covers the
10 complete overs that follow (overs 10–19, 0-indexed).

State at optimisation start (after over 9 completes):
  PBKS need  : 80 runs to win
  Balls left : 60  (overs 10–19)
  PBKS wickets left : 8  (Simran just dismissed, 2 down total)
  RRR        : 8.00 per over

GT bowling resources (overs 10–19):
  Bowler               Overs used  Quota left
  Ashok Sharma         1           3
  K Rabada             2           2
  Mohammed Siraj       2           2
  Rashid Khan          3 *         1   (* finishes over 9 regardless)
  Washington Sundar    2           2
  M Prasidh Krishna    0           4
  Total available      —          14  (need 10)
"""

import glob
import random
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────
DATA_DIR        = "data/matches"
EXCLUDE_DATE    = "2026-03-31"
MATCH_TITLE     = "GT vs PBKS  |  31 Mar 2026  |  Intervention: P Simran Singh wkt (PBKS 83/2)"

RUNS_TO_DEFEND  = 80     # PBKS needs 80 to win; GT defends if PBKS scores < 80
BALLS_REMAINING = 60     # 10 full overs (10–19) after Rashid finishes over 9
OVERS_REMAINING = 10
FIRST_OVER_IDX  = 10     # first remaining over (0-indexed)
WICKETS_REMAINING = 8    # PBKS wickets in hand at optimisation start

N_SIMS_FAST     = 5_000
N_SIMS_REFINE   = 30_000
N_SA_STEPS      = 8_000

MAX_OVERS_PER_BOWLER = 4

# GT bowlers: overs bowled AFTER over 9 completes (Rashid charged with over 9)
BOWLERS_USED = {
    "Ashok Sharma":      1,   # over 4
    "K Rabada":          2,   # overs 1, 3
    "Mohammed Siraj":    2,   # overs 0, 2
    "Rashid Khan":       3,   # overs 5, 7, 9
    "Washington Sundar": 2,   # overs 6, 8
    "M Prasidh Krishna": 0,
}
QUOTA   = {b: MAX_OVERS_PER_BOWLER - used for b, used in BOWLERS_USED.items()}
BOWLERS = list(BOWLERS_USED.keys())

# Actual bowling plan used in the match (overs 10–19, 0-indexed)
ACTUAL_PLAN = [
    "Ashok Sharma",      # over 10
    "Rashid Khan",       # over 11
    "M Prasidh Krishna", # over 12
    "Washington Sundar", # over 13
    "M Prasidh Krishna", # over 14
    "K Rabada",          # over 15
    "M Prasidh Krishna", # over 16
    "Ashok Sharma",      # over 17
    "M Prasidh Krishna", # over 18
    "Washington Sundar", # over 19
]

# Outcome indices and run values
# 0=Wicket  1=Dot  2=1run  3=2runs  4=3runs  5=4runs  6=6runs
OUTCOME_RUNS = np.array([0, 0, 1, 2, 3, 4, 6], dtype=np.int32)
N_OUTCOMES   = 7
PHASES       = ["PP", "MI", "DE"]


# ══════════════════════════════════════════════════════════════════════
# 1.  BOWLER PROFILE ENGINE
# ══════════════════════════════════════════════════════════════════════

def _phase_of(over: int) -> str:
    if over <= 5:    return "PP"
    elif over <= 14: return "MI"
    else:            return "DE"


def load_ball_data(data_dir: str, exclude_date: str) -> pd.DataFrame:
    files = [f for f in glob.glob(f"{data_dir}/*.csv") if exclude_date not in f]
    print(f"  Loading {len(files)} historical match files ...")
    chunks = []
    for f in files:
        try:
            df = pd.read_csv(
                f,
                usecols=["ball", "bowler", "batsman", "runs",
                          "wides", "wicketPlayerOut"],
                dtype={"runs": "int32", "wides": "int32"},
            )
            chunks.append(df)
        except Exception:
            pass
    all_balls = pd.concat(chunks, ignore_index=True)

    # Parse over from ball string (format: "1st.9.3" or "2nd.9.3")
    all_balls["over"]  = all_balls["ball"].map(
        lambda s: int(str(s).split(".")[1])
    )
    all_balls["phase"] = all_balls["over"].map(_phase_of)
    all_balls["runs"]  = all_balls["runs"].clip(0, 6)

    is_runout = all_balls["wicketPlayerOut"].map(
        lambda x: any(r in str(x).lower() for r in ["run out", "runout"])
    )
    all_balls["is_wicket"] = (
        (all_balls["wicketPlayerOut"] != "nobody") & (~is_runout)
    ).astype(np.int8)

    print(f"  {len(all_balls):,} total ball records loaded")
    return all_balls


def _count_outcomes(grp: pd.DataFrame) -> np.ndarray:
    legal = grp[grp["wides"] == 0]
    r = legal["runs"].values
    w = legal["is_wicket"].values
    return np.array([
        w.sum(),
        ((r == 0) & (w == 0)).sum(),
        (r == 1).sum(), (r == 2).sum(),
        (r == 3).sum(), (r == 4).sum(), (r == 6).sum(),
    ], dtype=np.float64)


def build_bowler_profiles(
    all_balls: pd.DataFrame,
    min_balls: int = 60,
    alpha: float   = 1.0,
) -> tuple[dict, dict]:
    pop = {}
    for phase in PHASES:
        grp    = all_balls[all_balls["phase"] == phase]
        counts = _count_outcomes(grp) + alpha
        pop[phase] = counts / counts.sum()

    profiles = {}
    for (bowler, phase), grp in all_balls.groupby(["bowler", "phase"]):
        legal  = grp[grp["wides"] == 0]
        n      = len(legal)
        raw    = _count_outcomes(grp)
        sm     = (raw + alpha) / (raw + alpha).sum()
        w_ind  = n / (n + min_balls)
        profiles[(bowler, phase)] = (
            w_ind * sm + (1 - w_ind) * pop[phase] if n < min_balls else sm
        )
    return profiles, pop


def get_profile(bowler, phase, profiles, pop):
    return profiles.get((bowler, phase), pop[phase])


# ══════════════════════════════════════════════════════════════════════
# 2.  PROFILE DISPLAY
# ══════════════════════════════════════════════════════════════════════

def rpb(bowler, phase, profiles, pop):
    return float(np.dot(get_profile(bowler, phase, profiles, pop), OUTCOME_RUNS))

def pw(bowler, phase, profiles, pop):
    return float(get_profile(bowler, phase, profiles, pop)[0])

def economy(bowler, phase, profiles, pop):
    return rpb(bowler, phase, profiles, pop) * 6


def print_bowler_profiles(bowlers, profiles, pop):
    phases_to_show = ["MI", "DE"]
    print(f"\n  {'Bowler':<22} {'Phase':<6} {'Economy':>8} {'P(W)/ball':>10} "
          f"{'Dot%':>6} {'4s%':>5} {'6s%':>5}  {'Quota':>5}")
    print("  " + "-" * 76)
    for bowler in bowlers:
        for phase in phases_to_show:
            p   = get_profile(bowler, phase, profiles, pop)
            eco = economy(bowler, phase, profiles, pop)
            print(f"  {bowler:<22} {phase:<6} {eco:>8.2f} {p[0]:>10.3f} "
                  f"{p[1]*100:>6.1f} {p[5]*100:>5.1f} {p[6]*100:>5.1f}  "
                  f"{QUOTA.get(bowler, 0):>5}")
        print()


# ══════════════════════════════════════════════════════════════════════
# 3.  VALIDATE A BOWLING PLAN
# ══════════════════════════════════════════════════════════════════════

def is_valid_plan(plan: list) -> bool:
    used = defaultdict(int)
    for i, bowler in enumerate(plan):
        used[bowler] += 1
        if used[bowler] > QUOTA.get(bowler, 0):
            return False
        if i > 0 and plan[i] == plan[i - 1]:
            return False
    return True


# ══════════════════════════════════════════════════════════════════════
# 4.  VECTORISED MONTE CARLO SIMULATION
# ══════════════════════════════════════════════════════════════════════

def phase_of_over(over_idx: int) -> str:
    if over_idx <= 5:    return "PP"
    elif over_idx <= 14: return "MI"
    else:                return "DE"


def simulate_bowling_plan(
    plan: list,
    profiles: dict,
    pop: dict,
    n_sims: int             = N_SIMS_FAST,
    runs_to_defend: int     = RUNS_TO_DEFEND,
    wickets_remaining: int  = WICKETS_REMAINING,
    rng: np.random.Generator = None,
) -> float:
    """
    Simulate n_sims innings of PBKS batting against the given GT bowling plan.
    Returns P(defend) = fraction where PBKS scores < runs_to_defend.
    """
    if rng is None:
        rng = np.random.default_rng()

    over_probs = np.zeros((OVERS_REMAINING, N_OUTCOMES))
    for i, bowler in enumerate(plan):
        over_abs      = FIRST_OVER_IDX + i
        phase         = phase_of_over(over_abs)
        over_probs[i] = get_profile(bowler, phase, profiles, pop)

    runs_scored = np.zeros(n_sims, dtype=np.int32)
    wickets     = np.zeros(n_sims, dtype=np.int32)
    done        = np.zeros(n_sims, dtype=bool)

    for over_i in range(OVERS_REMAINING):
        probs = over_probs[over_i]
        for _ in range(6):
            active    = ~done
            outcomes  = rng.choice(N_OUTCOMES, size=n_sims, p=probs)
            runs_ball = OUTCOME_RUNS[outcomes]
            is_wkt    = outcomes == 0

            runs_scored = np.where(active, runs_scored + runs_ball, runs_scored)
            wickets     = np.where(active & is_wkt, wickets + 1, wickets)

            won  = active & (runs_scored >= runs_to_defend)
            out  = active & (wickets >= wickets_remaining)
            done = done | won | out

        if done.all():
            break

    return float((runs_scored < runs_to_defend).mean())


# ══════════════════════════════════════════════════════════════════════
# 5.  PLAN SEARCH — SIMULATED ANNEALING
# ══════════════════════════════════════════════════════════════════════

def random_valid_plan(rng_py: random.Random) -> list:
    remaining_quota = dict(QUOTA)
    plan = []
    for slot in range(OVERS_REMAINING):
        last    = plan[-1] if plan else None
        choices = [b for b, q in remaining_quota.items() if q > 0 and b != last]
        if not choices:
            return None
        bowler = rng_py.choice(choices)
        plan.append(bowler)
        remaining_quota[bowler] -= 1
    return plan


def neighbour_plan(plan: list, rng_py: random.Random) -> list:
    new_plan = plan[:]
    used = defaultdict(int)
    for b in new_plan:
        used[b] += 1

    attempts = 0
    while attempts < 100:
        slot       = rng_py.randrange(OVERS_REMAINING)
        old_bowler = new_plan[slot]
        prev_bowler = new_plan[slot - 1] if slot > 0 else None
        next_bowler = new_plan[slot + 1] if slot < OVERS_REMAINING - 1 else None
        candidates  = [
            b for b in BOWLERS
            if b != old_bowler
            and used[b] - (1 if b == old_bowler else 0) < QUOTA[b]
            and b != prev_bowler
            and b != next_bowler
        ]
        if not candidates:
            attempts += 1
            continue
        new_bowler = rng_py.choice(candidates)
        new_plan[slot] = new_bowler
        used[old_bowler] -= 1
        used[new_bowler] += 1
        return new_plan
    return plan


def simulated_annealing_search(
    profiles: dict,
    pop:      dict,
    n_steps:  int = N_SA_STEPS,
    seed:     int = 42,
) -> list[tuple]:
    rng_py  = random.Random(seed)
    rng_np  = np.random.default_rng(seed)

    current_plan = random_valid_plan(rng_py)
    current_val  = simulate_bowling_plan(current_plan, profiles, pop,
                                          n_sims=N_SIMS_FAST, rng=rng_np)
    best_plan = current_plan[:]
    best_val  = current_val

    seen = {}
    seen[tuple(current_plan)] = current_val

    T0 = 0.05
    for step in range(n_steps):
        T = T0 * (1.0 - step / n_steps) + 1e-6

        neighbour = neighbour_plan(current_plan, rng_py)
        if not is_valid_plan(neighbour):
            continue

        key = tuple(neighbour)
        if key not in seen:
            val = simulate_bowling_plan(neighbour, profiles, pop,
                                         n_sims=N_SIMS_FAST, rng=rng_np)
            seen[key] = val
        else:
            val = seen[key]

        delta = val - current_val
        if delta > 0 or rng_py.random() < np.exp(delta / T):
            current_plan = neighbour[:]
            current_val  = val
            if val > best_val:
                best_val  = val
                best_plan = neighbour[:]

        if (step + 1) % 2000 == 0:
            print(f"    SA step {step+1}/{n_steps}  "
                  f"best defend% = {best_val*100:.1f}%  "
                  f"unique plans evaluated = {len(seen)}")

    all_plans = sorted(seen.items(), key=lambda x: -x[1])
    return [(list(plan), val) for plan, val in all_plans]


# ══════════════════════════════════════════════════════════════════════
# 6.  MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 72)
    print("Bowling Order Optimisation  —  GT vs PBKS, 31 March 2026")
    print("Intervention: P Simran Singh dismissed  (ball 2nd.9.3, caught Rashid Khan)")
    print("=" * 72)
    print(f"""
Match state at intervention (GT's perspective):
  PBKS score at dismissal : 83 / 2
  GT target to defend     : 162  (PBKS need 163 to win)
  Runs still to defend    : {RUNS_TO_DEFEND}  (PBKS need exactly {RUNS_TO_DEFEND} more)
  Balls left (after ov 9) : {BALLS_REMAINING}  (overs 10–19)
  PBKS wickets in hand    : {WICKETS_REMAINING}
  Required RR             : {RUNS_TO_DEFEND / (BALLS_REMAINING / 6):.2f} per over

  Note: Rashid Khan finishes over 9 regardless (3 balls already bowled).
        Optimisation covers overs 10–19 only.

GT bowling resources (overs remaining after over 9):
""")
    for b in BOWLERS:
        print(f"  {b:<22}  bowled: {BOWLERS_USED[b]}  quota left: {QUOTA[b]}")
    print(f"\n  Total quota available: {sum(QUOTA.values())} overs  "
          f"(need to fill {OVERS_REMAINING} slots)\n")

    # ── Build profiles ─────────────────────────────────────────────────
    print("Step 1: Building bowler profiles from historical IPL data ...")
    all_balls = load_ball_data(DATA_DIR, EXCLUDE_DATE)
    profiles, pop = build_bowler_profiles(all_balls)

    print("\nBowler profiles — Middle (overs 6–14) & Death (overs 15–19):")
    print_bowler_profiles(BOWLERS, profiles, pop)

    # Check ball counts for each bowler (to understand shrinkage)
    print("  Ball counts per bowler in historical data:")
    for bowler in BOWLERS:
        for phase in ["MI", "DE"]:
            grp   = all_balls[(all_balls["bowler"] == bowler) &
                               (all_balls["phase"] == phase) &
                               (all_balls["wides"] == 0)]
            print(f"    {bowler:<22} {phase}: {len(grp)} legal balls")
        print()

    # ── Actual plan ────────────────────────────────────────────────────
    print("Step 2: Evaluating ACTUAL bowling plan used in the match ...")
    ov_labels = [f"Ov{FIRST_OVER_IDX+i}:{b}" for i, b in enumerate(ACTUAL_PLAN)]
    print(f"  Actual plan: {' | '.join(ov_labels)}")

    if not is_valid_plan(ACTUAL_PLAN):
        print("  WARNING: actual plan violates constraints!")
    else:
        print("  Actual plan is valid (no consecutive overs, quotas respected).")

    rng_actual = np.random.default_rng(7)
    actual_dp  = simulate_bowling_plan(
        ACTUAL_PLAN, profiles, pop, n_sims=N_SIMS_REFINE, rng=rng_actual
    )
    print(f"  Defend probability (actual plan): {actual_dp*100:.1f}%\n")

    # ── SA search ──────────────────────────────────────────────────────
    print("Step 3: Searching for optimal bowling plan "
          f"(simulated annealing, {N_SA_STEPS} steps) ...")
    all_found = simulated_annealing_search(profiles, pop,
                                            n_steps=N_SA_STEPS, seed=42)
    print(f"  SA complete. {len(all_found)} unique plans evaluated.\n")

    # Refine top-10
    print("Step 4: Refining top 10 plans with "
          f"{N_SIMS_REFINE:,} simulations each ...")
    rng_ref = np.random.default_rng(99)
    top10   = []
    for plan, _ in all_found[:10]:
        dp = simulate_bowling_plan(plan, profiles, pop,
                                    n_sims=N_SIMS_REFINE, rng=rng_ref)
        top10.append((plan, dp))
    top10.sort(key=lambda x: -x[1])

    best_plan, best_dp = top10[0]

    # ── Results ────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("RESULTS")
    print("=" * 72)
    print(f"\nContext: Defend {RUNS_TO_DEFEND} runs in {BALLS_REMAINING} balls  "
          f"(RRR = {RUNS_TO_DEFEND / (BALLS_REMAINING / 6):.2f})")

    print("\nActual bowling plan:")
    _print_plan(ACTUAL_PLAN, actual_dp)

    print(f"\nOptimal bowling plan (model recommendation):")
    _print_plan(best_plan, best_dp)

    gain = best_dp - actual_dp
    print(f"\n  Gain from optimal plan : {gain*100:+.1f} pp  "
          f"({gain/actual_dp*100 if actual_dp>0 else 0:.0f}% relative)")

    print(f"\nTop 10 plans by defend probability:")
    hdr = f"  {'Rank':<5} {'Defend%':>7}   " + "  ".join(
        f"{'Ov'+str(FIRST_OVER_IDX+i):>10}" for i in range(OVERS_REMAINING))
    print(hdr)
    print("  " + "-" * (len(hdr) - 2 + 10))
    for rank, (plan, dp) in enumerate(top10, 1):
        tag  = "  ← optimal" if plan == best_plan else ""
        tag2 = "  ← actual"  if plan == ACTUAL_PLAN else ""
        row  = f"  {rank:<5} {dp*100:>7.1f}%  " + "  ".join(
            f"{b:<10}" for b in plan)
        print(row + tag + tag2)

    # ── Tactical breakdown ─────────────────────────────────────────────
    print("\nTactical breakdown — optimal vs actual:")
    print(f"\n  {'Over':<6} {'Actual':>22} {'Optimal':>22} {'Phase':<6} "
          f"{'Eco(act)':>9} {'Eco(opt)':>9} {'P(W)opt':>8}")
    print("  " + "-" * 90)
    for i in range(OVERS_REMAINING):
        ov    = FIRST_OVER_IDX + i
        phase = phase_of_over(ov)
        ab    = ACTUAL_PLAN[i]
        ob    = best_plan[i]
        ae    = economy(ab, phase, profiles, pop)
        oe    = economy(ob, phase, profiles, pop)
        op_w  = pw(ob, phase, profiles, pop)
        chg   = "  ←" if ab != ob else ""
        print(f"  {ov:<6} {ab:>22} {ob:>22} {phase:<6} "
              f"{ae:>9.2f} {oe:>9.2f} {op_w:>8.3f}{chg}")

    # ── Key observations ───────────────────────────────────────────────
    print("\nKey observations:")
    death_best = min(
        [(b, economy(b, "DE", profiles, pop), pw(b, "DE", profiles, pop))
         for b in BOWLERS if QUOTA[b] > 0],
        key=lambda x: x[1]
    )
    mid_best = min(
        [(b, economy(b, "MI", profiles, pop), pw(b, "MI", profiles, pop))
         for b in BOWLERS if QUOTA[b] > 0],
        key=lambda x: x[1]
    )
    print(f"  • Best Death bowler : {death_best[0]}  "
          f"(Econ {death_best[1]:.2f}, P(W) {death_best[2]:.3f})")
    print(f"  • Best Middle bowler: {mid_best[0]}  "
          f"(Econ {mid_best[1]:.2f}, P(W) {mid_best[2]:.3f})")

    for b in BOWLERS:
        opt_use = best_plan.count(b)
        act_use = ACTUAL_PLAN.count(b)
        if opt_use != act_use:
            diff = opt_use - act_use
            print(f"  • {b}: actual {act_use} over(s) → optimal {opt_use} over(s) "
                  f"({'more' if diff>0 else 'fewer'} by {abs(diff)})")

    # Specifically flag Mohammed Siraj usage
    siraj_actual  = ACTUAL_PLAN.count("Mohammed Siraj")
    siraj_optimal = best_plan.count("Mohammed Siraj")
    if siraj_actual == 0 and siraj_optimal > 0:
        siraj_eco_de = economy("Mohammed Siraj", "DE", profiles, pop)
        print(f"\n  ★ Mohammed Siraj (quota: 2 overs) was NOT used in the actual plan.")
        print(f"    His Death economy is {siraj_eco_de:.2f}. The model recommends "
              f"{siraj_optimal} over(s).")

    print(f"""
  • Phases: overs 10–14 are MIDDLE phase; overs 15–19 are DEATH phase.
    The batting team (PBKS) is accelerating toward a {RUNS_TO_DEFEND}-run target
    in {BALLS_REMAINING} balls — Death-over specialists are critical in the final 5.

  • PBKS ultimately {'won' if True else 'lost'} — the match result shows the value
    of every percentage point of defend probability.
""")
    print("=" * 72)

    # ── Summary dict for dashboard use ────────────────────────────────
    print("\n--- SUMMARY FOR DASHBOARD ---")
    print(f"ACTUAL_DP = {actual_dp*100:.1f}")
    print(f"OPTIMAL_DP = {best_dp*100:.1f}")
    print(f"GAIN_PP = {(best_dp-actual_dp)*100:.1f}")
    print(f"OPTIMAL_PLAN = {best_plan}")
    print(f"TOP10_PLANS = {[(p, round(v*100,1)) for p,v in top10]}")

    # Per-bowler economy summary for dashboard
    print("\n--- BOWLER PROFILES FOR DASHBOARD ---")
    for bowler in BOWLERS:
        mi_eco = economy(bowler, "MI", profiles, pop)
        de_eco = economy(bowler, "DE", profiles, pop)
        mi_pw  = pw(bowler, "MI", profiles, pop)
        de_pw  = pw(bowler, "DE", profiles, pop)
        print(f"  {bowler}: MI_eco={mi_eco:.2f} DE_eco={de_eco:.2f} "
              f"MI_pw={mi_pw:.3f} DE_pw={de_pw:.3f} quota={QUOTA[bowler]}")


def _print_plan(plan: list, dp: float) -> None:
    for i, b in enumerate(plan):
        ov    = FIRST_OVER_IDX + i
        phase = phase_of_over(ov)
        print(f"  Over {ov:2d} ({phase})  →  {b}")
    print(f"  Defend probability : {dp*100:.1f}%")


if __name__ == "__main__":
    main()
