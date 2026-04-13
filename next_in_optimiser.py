#!/usr/bin/env python3
"""
Next-In Decision Optimiser — After Rohit Sharma's Dismissal
============================================================
KKR vs MI, 29 March 2026  |  Ball 2nd.11.6

Match state when Rohit is dismissed (caught off VG Arora):
  • Runs needed      : 73
  • Legal balls left : 44  (~7.3 overs)
  • Wickets in hand  : 9
  • At the crease    : RD Rickelton (non-striker, stays)
  • Required RR      : 9.95 per over

Decision:
  MI must choose ONE of [SA Yadav, Tilak Varma, HH Pandya, Naman Dhir]
  to come in ON STRIKE immediately.  The remaining three then fill
  positions 4, 5, 6 in whatever order is optimal for each scenario.

Approach:
  For each of the 4 candidates as "next in" (position 3):
    → Enumerate all 6 orderings of the remaining 3 (positions 4, 5, 6)
    → Simulate 50,000 innings for each ordering
    → Report best ordering and best win% for that "next in" choice

  Total: 4 × 6 = 24 evaluations.
  Best "next in" = the choice that maximises the best achievable win%.
"""

import glob
import warnings
from itertools import permutations

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────
DATA_DIR     = "data/matches"
EXCLUDE_DATE = "2026-03-29"

RUNS_NEEDED     = 73
BALLS_REMAINING = 44
WICKETS_IN_HAND = 9

NON_STRIKER = "RD Rickelton"          # stays at crease, index 1

# The pool of 4 batsmen MI can choose from
POOL = ["SA Yadav", "Tilak Varma", "HH Pandya", "Naman Dhir"]

# What actually happened: SA Yadav came in next
ACTUAL_NEXT_IN = "SA Yadav"
ACTUAL_REMAINING = ["Tilak Varma", "HH Pandya", "Naman Dhir"]   # actual positions 4,5,6

N_SIMS = 50_000

# Outcome: 0=W  1=dot  2=1  3=2  4=3  5=4  6=6
OUTCOME_RUNS = np.array([0, 0, 1, 2, 3, 4, 6], dtype=np.int32)
N_OUTCOMES   = 7
PHASES       = ["PP", "MI", "DE"]


# ══════════════════════════════════════════════════════════════════════
# 1.  PLAYER PROFILES  (same engine as previous scripts)
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
                f, usecols=["ball", "batsman", "runs", "wides", "wicketPlayerOut"],
                dtype={"runs": "int32", "wides": "int32"},
            )
            chunks.append(df)
        except Exception:
            pass
    all_balls = pd.concat(chunks, ignore_index=True)
    all_balls = all_balls[all_balls["wides"] == 0].copy()
    all_balls["over"]      = all_balls["ball"].map(lambda s: int(s.split(".")[1]))
    all_balls["phase"]     = all_balls["over"].map(_phase_of)
    all_balls["is_wicket"] = (
        (all_balls["wicketPlayerOut"] != "nobody") &
        (all_balls["wicketPlayerOut"] == all_balls["batsman"])
    ).astype(np.int8)
    all_balls["runs"] = all_balls["runs"].clip(0, 6)
    print(f"  {len(all_balls):,} legal ball records loaded")
    return all_balls[["batsman", "phase", "runs", "is_wicket"]]


def _count_outcomes(grp: pd.DataFrame) -> np.ndarray:
    r, w = grp["runs"].values, grp["is_wicket"].values
    return np.array([
        w.sum(),
        ((r == 0) & (w == 0)).sum(),
        (r == 1).sum(), (r == 2).sum(),
        (r == 3).sum(), (r == 4).sum(), (r == 6).sum(),
    ], dtype=np.float64)


def build_profiles(all_balls: pd.DataFrame, min_balls: int = 50, alpha: float = 1.0):
    pop = {}
    for phase in PHASES:
        c = _count_outcomes(all_balls[all_balls["phase"] == phase]) + alpha
        pop[phase] = c / c.sum()

    profiles = {}
    for (batsman, phase), grp in all_balls.groupby(["batsman", "phase"]):
        n   = len(grp)
        raw = _count_outcomes(grp)
        sm  = (raw + alpha) / (raw + alpha).sum()
        w   = n / (n + min_balls)
        profiles[(batsman, phase)] = w * sm + (1 - w) * pop[phase] if n < min_balls else sm
    return profiles, pop


def get_profile(batsman, phase, profiles, pop):
    return profiles.get((batsman, phase), pop[phase])


def rpb(batsman, phase, profiles, pop):
    """Expected runs per ball."""
    return float(np.dot(get_profile(batsman, phase, profiles, pop), OUTCOME_RUNS))


def pw(batsman, phase, profiles, pop):
    """P(dismissal) per ball."""
    return float(get_profile(batsman, phase, profiles, pop)[0])


# ══════════════════════════════════════════════════════════════════════
# 2.  PROFILE DISPLAY TABLE
# ══════════════════════════════════════════════════════════════════════

def print_profiles(batsmen, profiles, pop):
    """
    Print Middle + Death stats for each batsman.
    These are the only phases that matter from over 12 onward.
    """
    print(f"\n  {'Batsman':<18} {'Phase':<6} {'Death SR':>9} {'P(W)':>6} "
          f"{'Dot%':>6} {'4s%':>5} {'6s%':>5}  Comment")
    print("  " + "-" * 72)
    for b in batsmen:
        for phase in ["MI", "DE"]:
            p    = get_profile(b, phase, profiles, pop)
            sr   = rpb(b, phase, profiles, pop) * 100
            p_w  = p[0]
            dot  = p[1] * 100
            p4   = p[5] * 100
            p6   = p[6] * 100
            note = ""
            if phase == "DE":
                if sr >= 190:  note = "★★ Elite finisher"
                elif sr >= 175: note = "★  Strong finisher"
                else:           note = "   Solid"
            print(f"  {b:<18} {phase:<6} {sr:>9.1f} {p_w:>6.3f} "
                  f"{dot:>6.1f} {p4:>5.1f} {p6:>5.1f}  {note}")
        print()


# ══════════════════════════════════════════════════════════════════════
# 3.  VECTORISED SIMULATION (mid-innings)
# ══════════════════════════════════════════════════════════════════════

def phase_idx(ball_num: int) -> int:
    """ball_num is 1-indexed from start of remaining balls."""
    abs_ball = (120 - BALLS_REMAINING) + ball_num
    abs_over = (abs_ball - 1) // 6
    return 0 if abs_over <= 5 else (1 if abs_over <= 14 else 2)


def build_matrix(order, profiles, pop, n_pos=11):
    mat = np.zeros((n_pos, 3, 7))
    for pos in range(n_pos):
        for ph, phase in enumerate(PHASES):
            src = order[pos] if pos < len(order) else None
            mat[pos, ph] = get_profile(src, phase, profiles, pop) if src else pop[phase]
    return mat


def simulate(order, profiles, pop, n_sims=N_SIMS, rng=None):
    """
    order[0] = on-strike,  order[1] = non-striker,  order[2..] = next in.
    Returns win probability.
    """
    if rng is None:
        rng = np.random.default_rng()
    pm = build_matrix(order, profiles, pop)

    runs   = np.full(n_sims, RUNS_NEEDED,     dtype=np.int32)
    wkts   = np.full(n_sims, WICKETS_IN_HAND, dtype=np.int32)
    on_s   = np.zeros(n_sims,                 dtype=np.int32)
    non_s  = np.ones(n_sims,                  dtype=np.int32)
    nxt    = np.full(n_sims, 2,               dtype=np.int32)
    done   = np.zeros(n_sims,                 dtype=bool)

    for ball_num in range(1, BALLS_REMAINING + 1):
        if done.all():
            break
        active = ~done
        ph     = phase_idx(ball_num)
        abs_b  = (120 - BALLS_REMAINING) + ball_num
        last   = ((abs_b - 1) % 6) == 5

        # Sample outcomes grouped by on-strike position index
        out = np.zeros(n_sims, dtype=np.int32)
        for pos in np.unique(on_s[active]):
            mask = active & (on_s == pos)
            out[mask] = rng.choice(N_OUTCOMES, size=int(mask.sum()),
                                   p=pm[min(pos, pm.shape[0]-1), ph])
        scored   = OUTCOME_RUNS[out]
        is_wkt   = out == 0

        runs = np.where(active, runs - scored, runs)

        # Wicket
        wk       = active & is_wkt
        wkts     = np.where(wk, wkts - 1, wkts)
        sv_on, sv_non = on_s.copy(), non_s.copy()
        on_s     = np.where(wk, sv_non, on_s)
        non_s    = np.where(wk, nxt,    non_s)
        nxt      = np.where(wk, nxt + 1, nxt)

        # Strike rotation
        rot      = (active & ~is_wkt) & ((scored % 2 == 1) | last)
        cur_on, cur_non = on_s.copy(), non_s.copy()
        on_s     = np.where(rot, cur_non, cur_on)
        non_s    = np.where(rot, cur_on,  cur_non)

        done = done | (active & (runs <= 0)) | (active & (wkts <= 0))

    return float((runs <= 0).mean())


# ══════════════════════════════════════════════════════════════════════
# 4.  EVALUATE ALL 24 COMBINATIONS
# ══════════════════════════════════════════════════════════════════════

def evaluate_all(profiles, pop):
    """
    For each of the 4 candidates as 'next in' (on-strike position 3),
    find the best ordering of the remaining 3 (positions 4, 5, 6).

    Returns list of:
        (next_in, best_remaining_order, best_wp, all_orderings_for_this_next_in)
    sorted by best_wp descending.
    """
    rng = np.random.default_rng(42)
    summary = []

    for next_in in POOL:
        remaining = [b for b in POOL if b != next_in]
        orderings = []

        for perm in permutations(remaining):
            tail  = list(perm)
            order = [next_in, NON_STRIKER] + tail
            wp    = simulate(order, profiles, pop, rng=rng)
            orderings.append((tail, wp))

        orderings.sort(key=lambda x: x[1], reverse=True)
        best_tail, best_wp = orderings[0]
        summary.append((next_in, best_tail, best_wp, orderings))

    summary.sort(key=lambda x: x[2], reverse=True)
    return summary


# ══════════════════════════════════════════════════════════════════════
# 5.  MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("Next-In Decision  —  KKR vs MI, 29 March 2026")
    print("Point: Rohit Sharma dismissed  (ball 2nd.11.6, over 12)")
    print("=" * 70)
    print(f"""
State at intervention:
  Runs needed      : {RUNS_NEEDED}  (MI at 148 / 221)
  Legal balls left : {BALLS_REMAINING}  (~7.3 overs)
  Wickets in hand  : {WICKETS_IN_HAND}
  Required RR      : {RUNS_NEEDED / (BALLS_REMAINING / 6):.2f} per over
  At crease        : {NON_STRIKER}  (non-striker, stays)
  Decision         : who comes in ON STRIKE next?

  Candidates: {', '.join(POOL)}
  Actual choice  : {ACTUAL_NEXT_IN}  (→ then {' → '.join(ACTUAL_REMAINING)})
""")

    # ── Profiles ────────────────────────────────────────────────────────
    print("Step 1: Building player profiles from historical IPL data ...")
    all_balls = load_ball_data(DATA_DIR, EXCLUDE_DATE)
    profiles, pop = build_profiles(all_balls)

    print("\nBatsman profiles — Middle & Death phases (overs 7-15 and 16-20):")
    print_profiles([NON_STRIKER] + POOL, profiles, pop)

    # ── Evaluate all 24 combos ──────────────────────────────────────────
    print("Step 2: Evaluating all 24 combinations "
          f"(4 choices × 6 orderings × {N_SIMS:,} sims) ...\n")
    results = evaluate_all(profiles, pop)

    # ── Print full detail per candidate ────────────────────────────────
    for next_in, best_tail, best_wp, orderings in results:
        tag = "  ← ACTUAL CHOICE" if next_in == ACTUAL_NEXT_IN else ""
        print(f"  Next in: {next_in}{tag}")
        print(f"  {'Order (pos 4, 5, 6)':<48} {'Win%':>6}")
        print("  " + "-" * 56)
        for tail, wp in orderings:
            marker = " ← best for this next-in" if tail == best_tail else ""
            print(f"  {' → '.join(tail):<48} {wp*100:>5.1f}%{marker}")
        print(f"  Best achievable win% with {next_in} next in: {best_wp*100:.1f}%\n")

    # ── Summary table ───────────────────────────────────────────────────
    print("=" * 70)
    print("SUMMARY — Best achievable win% for each 'next in' choice")
    print("=" * 70)
    print(f"\n  {'Rank':<5} {'Next in (pos 3)':<20} {'Best remaining order':<40} {'Win%':>6}")
    print("  " + "-" * 74)

    actual_wp = None
    for rank, (next_in, best_tail, best_wp, _) in enumerate(results, 1):
        tag = ""
        if next_in == ACTUAL_NEXT_IN:
            actual_wp = best_wp
            tag = "  ← actual"
        print(f"  {rank:<5} {next_in:<20} {' → '.join(best_tail):<40} "
              f"{best_wp*100:>5.1f}%{tag}")

    # ── Key recommendation ──────────────────────────────────────────────
    best_next_in, best_tail_overall, best_wp_overall, _ = results[0]
    actual_rank = next(
        r + 1 for r, (ni, _, _, _) in enumerate(results) if ni == ACTUAL_NEXT_IN
    )

    print(f"""
Recommendation:
  Send in  : {best_next_in}  (next in, on strike)
  Then     : {' → '.join(best_tail_overall)}
  Win prob : {best_wp_overall*100:.1f}%

  Actual choice ({ACTUAL_NEXT_IN}) was #{actual_rank} of 4.
  Gain over actual: {(best_wp_overall - actual_wp)*100:+.1f} percentage points.
""")

    # ── Why: Death-over metric comparison ──────────────────────────────
    print("Why the model recommends this:")
    print(f"  {'Batsman':<18} {'Death SR':>9} {'P(W) Death':>11} "
          f"{'Middle SR':>10} {'P(W) Mid':>9}")
    print("  " + "-" * 62)
    for b in POOL:
        de_sr  = rpb(b, "DE", profiles, pop) * 100
        de_pw  = pw(b, "DE", profiles, pop)
        mi_sr  = rpb(b, "MI", profiles, pop) * 100
        mi_pw  = pw(b, "MI", profiles, pop)
        tag    = "  ← send in next" if b == best_next_in else ""
        print(f"  {b:<18} {de_sr:>9.1f} {de_pw:>11.3f} "
              f"{mi_sr:>10.1f} {mi_pw:>9.3f}{tag}")

    print(f"""
  With {BALLS_REMAINING} balls left, roughly 18 are in the Middle overs (12-15)
  and 26 are Death overs (16-20).  The model prioritises Death SR above
  all else — with 9 wickets in hand, the risk of losing additional
  wickets is low, so the emphasis is on maximising runs per ball.

  The worst choice is sending a batsman with a high Middle SR but lower
  Death SR, because they consume Middle-over balls they could dominate
  but cede Death-over balls to a lesser finisher.
""")
    print("=" * 70)


if __name__ == "__main__":
    main()
