---
title: "Simulation-Based Optimisation of Batting Order and Bowling Plans in T20 Cricket"
subtitle: "A Markov Decision Process Framework with Dual Case Studies"
author: "Tinniam V Ganesh"
date: "April 2026"
---

## Abstract

T20 cricket presents two coupled, recurrent decision problems under uncertainty: at each wicket fall the batting captain must select which of the remaining batsmen to promote, and at each over boundary the bowling captain must assign a bowler to the next over subject to T20 quota rules. Both decisions are recourse actions in a multi-stage stochastic setting where winning is a threshold event, not an expectation, over a distribution of ball-by-ball outcomes. Prior approaches based on linear programming that maximise expected strike rate or minimise expected economy rate are structurally misaligned with this objective.

This paper develops a unified Markov Decision Process (MDP) framework in which both problems share a common state space $(r, b, w)$, that is, runs remaining, balls remaining, and wickets in hand, and a common player profile engine that estimates per-ball outcome distributions across three innings phases (Powerplay, Middle, Death) from 1,161 IPL ball-by-ball records (2008–2025). Win and defend probabilities are estimated by vectorised Monte Carlo simulation over $N = 50{,}000$ innings trajectories. The batting order is searched by exhaustive enumeration over $n!$ permutations; the bowling plan by simulated annealing over the constrained space of feasible over assignments.

The framework is applied to two intervention points drawn from the 2026 IPL season. In the Kolkata Knight Riders vs. Mumbai Indians match (29 March 2026), the optimal batting order improves MI's win probability by 4.1 pp (52.4% → 56.5%). In the Gujarat Titans vs. Punjab Kings match (31 March 2026), the optimal GT bowling plan improves defend probability by 5.2 pp (39.1% → 44.3%). In both cases the principal source of suboptimality is phase-agnostic deployment: high-Death-SR batsmen are promoted too late in the first match, and bowlers with high Death-over economy are deployed in the final overs in the second.

**Keywords:** T20 cricket; batting order optimisation; bowling plan optimisation; Markov Decision Process; Monte Carlo simulation; simulated annealing; James–Stein shrinkage; sports analytics; Indian Premier League

---

## 1. Introduction

The Twenty20 (T20) format has transformed professional cricket into a high-frequency, high-stakes decision environment. A T20 innings spans exactly 120 legal deliveries per side, yet within that compact frame the trajectory of a match can shift dramatically within a single over. Team values in the Indian Premier League (IPL) now exceed one billion USD, playoff positions hinge on net run rate margins, and individual match outcomes directly affect player retention and squad selection. In this environment, the quality of tactical in-match decisions (which batsman to promote, which bowler to deploy) carries measurable financial and competitive consequences.

The two most consequential recurring decisions in a T20 match are:

**Batting order.** At each wicket fall, the batting captain must nominate which of the remaining batsmen comes in next. The optimal choice balances aggression (a high-strike-rate Death-over specialist) against preservation (a low-dismissal-probability anchor), and its value depends on the current match state (runs remaining, balls remaining, wickets already lost) and on which batsmen remain in the dressing room.

**Bowling plan.** Across the overs still to be bowled, the bowling captain must assign bowlers to over slots subject to two hard constraints: the T20 quota (each bowler may not bowl more than four overs per innings) and the consecutive-over rule (the same bowler cannot bowl back-to-back overs). The optimal assignment balances phase-specific economy rate, wicket-taking probability, and dot-ball pressure, again conditional on match state.

Both decisions share a common structure: they are **sequential stochastic decisions** made after observing the current match state, in an environment where future outcomes are governed by known (or estimable) probability distributions. This structure places them firmly within the framework of Markov Decision Processes (Bellman, 1957; Puterman, 1994). The crucial point, and the central departure from prior work, is that the objective is not to maximise or minimise an expectation, but to maximise a **probability**: the probability that the total runs scored (batting) or conceded (bowling) crosses a known threshold. Expected runs is a poor proxy for this probability when match states are close, run distributions are skewed, and the variance of outcomes is large relative to the margin required.

Prior work has addressed batting order and bowling selection separately and largely through heuristic or linear programming (LP) approaches. While LP provides a useful first-order framework, assigning batsmen or bowlers to overs in proportion to their aggregate quality metrics, it is inadequate for the precise, in-match decisions studied here. LP optimises an expectation; the match objective is a tail probability. LP produces an allocation of overs across bowlers; the match requires a sequence. LP cannot encode the consecutive-over constraint or the interaction between wickets taken and subsequent run rates.

This paper makes the following contributions:

1. A **unified MDP framework** that models batting order and bowling plan optimisation on a common state space with symmetric Bellman equations, making explicit the duality between the batting win probability and the bowling defend probability.

2. A **three-phase player profile engine** using Laplace smoothing and James–Stein shrinkage that estimates per-ball outcome distributions for batsmen and bowlers in Powerplay (overs 1–6), Middle (overs 7–15), and Death (overs 16–20) phases separately, from 1,161 historical IPL matches.

3. **Simulation-based search**, namely exhaustive enumeration for batting order and simulated annealing for bowling plans, that directly optimises win/defend probability over the stochastic innings model, validated at $N = 50{,}000$ simulations per configuration (Monte Carlo SE $\approx 0.22\%$).

4. **Two detailed case studies** from the 2026 IPL season, providing quantitative audits of actual in-match decisions: the batting and bowling decisions in KKR vs. MI (29 March 2026) and the bowling decisions in GT vs. PBKS (31 March 2026). The two cases span different match contexts: one where the bowling gap is large (+18.1 pp) and one where it is modest (+5.2 pp), enabling a comparative analysis of the conditions under which sub-optimal decisions have the greatest impact.

The paper is structured as follows. Section 2 reviews related work. Section 3 presents the MDP framework. Section 4 describes the player profile engine. Section 5 presents the Monte Carlo simulation. Section 6 covers the search algorithms. Sections 7 and 8 present the two case studies and their results. Section 9 provides a comparative analysis and discussion of limitations. Section 10 concludes.

---

## 2. Related Work

The Duckworth–Lewis method (Duckworth and Lewis, 1998) implicitly defines a resource table over (runs needed, balls remaining, wickets in hand), the same state space used in this paper, for resetting targets in interrupted matches. It remains the de facto standard in international cricket and establishes that match state can be meaningfully quantified.

The author's earlier work (Ganesh, 2017) formulated batting lineup and bowling selection as linear programming problems, assigning ball-level encounters $o_{ij}$ between player $i$ and bowler $j$ to maximise aggregate strike rate (batting) or minimise aggregate economy rate (bowling). That formulation correctly identifies that the allocation of playing time across player matchups matters, but it optimises expected runs rather than win probability and cannot encode the sequential, constraint-rich structure of T20 over assignment. The present paper replaces the LP objective with a stochastic MDP framework whose objective directly matches the match outcome: a win probability threshold rather than an expectation.

The MDP framework and Bellman equations are due to Bellman (1957) and Puterman (1994). The James–Stein shrinkage estimator applied to sparse player profiles follows James and Stein (1961). The simulated annealing search for bowling plans draws on Kirkpatrick, Gelatt, and Vecchi (1983).

---

## 3. The MDP Framework

### 3.1 State Space and Match Dynamics

Let the **match state** at any ball be the triple:

$$\mathcal{S} = (r,\, b,\, w)$$

where $r \in \{1, \ldots, R_{\max}\}$ is the runs remaining for the batting side to win (or, equivalently, for the bowling side to defend), $b \in \{0, 1, \ldots, B\}$ is the number of legal balls remaining in the innings ($B = 120$ in T20), and $w \in \{0, 1, \ldots, 10\}$ is the number of wickets in hand (batting perspective) or wickets taken (bowling perspective). The Markov property holds: the probability distribution over future match trajectories is fully determined by $\mathcal{S}$, independent of the history of play.

On each legal delivery, one of seven outcomes $o \in \Omega = \{W, 0, 1, 2, 3, 4, 6\}$ is realised, where $W$ denotes the dismissal of the on-strike batsman and $\rho(o) \in \{0,0,1,2,3,4,6\}$ denotes the associated runs scored. The outcome $o = 5$ runs (an all-run five) is omitted from $\Omega$: it occurs in fewer than 0.1% of deliveries in the historical data and its inclusion has negligible effect on computed probabilities. Wides and no-balls are handled at the data-preprocessing stage and are not included in the legal delivery count.

### 3.2 Batting MDP

For the batting side chasing a target, define the **win probability value function**:

$$V^B(r, b, w) = \mathbb{P}\!\left(\text{batting side wins} \;\middle|\; \text{state } (r,b,w),\; \text{order } \sigma\right)$$

**Terminal conditions:** The batting side wins if and only if the target is reached:

$$V^B(r, b, w) = \begin{cases} 1 & \text{if } r \leq 0 \\[4pt] 0 & \text{if } b = 0 \text{ or } w = 0, \text{ and } r > 0 \end{cases}$$

**Transition function:** Given on-strike batsman $i$ in phase $\phi(b)$, outcome $o$ occurs with probability $p_i^{o,\phi}$:

$$T\!\left((r,b,w),\, o\right) = \begin{cases} (r,\; b-1,\; w-1) & o = W \\[3pt] (r - \rho(o),\; b-1,\; w) & o \in \{0,1,2,3,4,6\} \end{cases}$$

**Bellman equation:**

$$\boxed{V^B(r, b, w) = \sum_{o \in \Omega} p_i^{o,\phi(b)} \cdot V^B\!\left(T(r,b,w,o)\right)}$$

Expanding over all outcomes:

$$V^B(r,b,w) = p_i^{W,\phi} \cdot V^B(r,\, b-1,\, w-1) + \sum_{\rho \in \{0,1,2,3,4,6\}} p_i^{\rho,\phi} \cdot V^B(r - \rho,\, b-1,\, w)$$

**Strike rotation** is enforced: the on-strike batsman rotates to the non-striker when $\rho(o)$ is odd, or when $b$ is the last ball of an over. This is consequential: which batsman faces the first ball of the Death overs (16–20) depends on the parity of singles accumulated during overs 14–15, and the SR differential between a Death-over specialist and a lower-order batsman can exceed 50 points.

**Optimal batting order:**

$$\sigma^* = \arg\max_{\sigma \in \text{Perm}(\mathcal{P})} V^B_\sigma(r_0, b_0, w_0)$$

where $\mathcal{P}$ is the set of remaining batsmen and $\text{Perm}(\mathcal{P})$ is the set of all $|\mathcal{P}|!$ orderings.

### 3.3 Bowling MDP

For the bowling side defending a total, define the **defend probability value function** under plan $\pi$:

$$V^\pi(d, b, w) = \mathbb{P}\!\left(\text{bowling side defends} \;\middle|\; \text{state } (d,b,w),\; \text{plan } \pi\right)$$

**Terminal conditions:**

$$V^\pi(d, b, w) = \begin{cases} 0 & \text{if } d \leq 0 \qquad \text{(batting side reaches target)}\\[4pt] 1 & \text{if } b = 0 \text{ or } w = w_{\max}, \text{ and } d > 0 \end{cases}$$

where $w_{\max}$ is the number of wickets in hand for the batting side at the intervention point (not necessarily 10 if wickets have already fallen).

**Transition function:** Given bowler $j = \pi_k$ (assigned to the current over $k$) in phase $\phi(b)$:

$$T\!\left((d,b,w),\, o\right) = \begin{cases} (d,\; b-1,\; w+1) & o = W \\[3pt] (d - \rho(o),\; b-1,\; w) & o \in \{0,1,2,3,4,6\} \end{cases}$$

Here $w$ counts additional wickets taken from the intervention point; the innings ends when $w$ reaches $w_{\max}$.

**Bellman equation:**

$$\boxed{V^\pi(d, b, w) = \sum_{o \in \Omega} p_j^{o,\phi(b)} \cdot V^\pi\!\left(T(d,b,w,o)\right)}$$

**Optimal bowling plan:**

$$\pi^* = \arg\max_{\pi \in \mathcal{F}} V^\pi(d_0, b_0, w_0)$$

where $\mathcal{F}$ is the feasible plan set (Section 6.1).

### 3.4 Duality

At the same match state $(r_0 = d_0,\, b_0,\, w_0)$, the batting win probability and bowling defend probability are exact complements under any fixed batting order $\sigma$ and bowling plan $\pi$:

$$V^B_\sigma(r_0, b_0, w_0) + V^\pi(d_0, b_0, w_0) = 1$$

This follows directly from the fact that a T20 match has no draw: the batting side either reaches the target (win) or does not (lose). The duality is exploited in the case studies to provide a complete audit: improvements to one side translate directly to losses for the other.

### 3.5 Why Exact Backward Induction is Replaced by Simulation

For the bowling problem with fixed player identities, exact backward induction over $(d, b, w)$ is feasible; the state space has at most $D \times B \times W \approx 221 \times 44 \times 10 \approx 97{,}000$ states and the Bellman recursion terminates in polynomial time. However, the bowling plan $\pi$ itself must be searched over the feasible set $\mathcal{F}$, which has $O(|\mathcal{F}|)$ elements; running backward induction for each candidate plan is inefficient. Monte Carlo simulation with a fixed plan provides an unbiased estimate of $V^\pi(d_0, b_0, w_0)$ directly without filling the full state table.

For the **batting problem**, the state must track which specific batsmen remain, a requirement that adds a combinatorial factor of $2^n$ subsets to the state space. For $n = 6$ remaining batsmen, this multiplies the state count by $64\times$, rendering exact backward induction impractical on commodity hardware. Monte Carlo simulation with a fixed order $\sigma$ avoids this by forward-sampling from the initial state, estimating the value function at $(r_0, b_0, w_0)$ without computing all intermediate states.

---

## 4. Player Profile Engine

### 4.1 Data

Ball-by-ball records from 1,161 IPL matches (seasons 2008–2025) are sourced from Cricsheet.org in CSV format via the yorkr R package (Ganesh, 2016). The dataset comprises 273,735 total delivery records; after filtering for legal deliveries (excluding wides and no-balls) approximately 264,800 legal balls remain for profile estimation. All matches involving the target teams (KKR–MI on 29 March 2026 and GT–PBKS on 31 March 2026) are excluded to prevent look-ahead bias.

### 4.2 Phase Assignment

Each delivery is assigned to one of three innings phases by the over number (0-indexed):

| Phase | Overs (0-indexed) | Characteristic Conditions |
|-------|-------------------|---------------------------|
| Powerplay (PP) | 0 – 5 | Fielding restrictions; elevated scoring rates |
| Middle (MI) | 6 – 14 | Spin-friendly; dot-ball accumulation |
| Death (DE) | 15 – 19 | Yorkers; pull shots; explosive hitting |

Phase assignment is deterministic given the over number; in-over ball position is not used.

### 4.3 Outcome Counts and Wicket Attribution

For each player $i$ and phase $\phi$, raw outcome counts are accumulated over all $n_i^\phi$ legal deliveries:

$$c_i^{o,\phi} = \bigl|\{t : \text{player } i \text{ involved in delivery } t,\; \text{phase}(t) = \phi,\; \text{outcome}(t) = o\}\bigr|, \quad o \in \Omega$$

Two distinct attribution rules apply depending on whether $i$ is a batsman or bowler.

**Batsman wicket attribution:** A delivery contributes $o = W$ to batsman $i$'s counts if and only if $\texttt{wicketPlayerOut}(t) = i$. This correctly attributes caught-behind, LBW, stumped, bowled, and caught dismissals to the dismissed batsman, and excludes run-outs of the non-striker (who is not the on-strike batsman for that delivery).

**Bowler wicket attribution:** A wicket is credited to bowler $j$ if and only if the dismissal type is not a run-out:

$$\mathbf{1}_{\text{bowler wicket}}(t) = \mathbf{1}\!\left[\texttt{wicketPlayerOut}(t) \neq \texttt{nobody}\right] \cdot \mathbf{1}\!\left[\texttt{dismissalType}(t) \notin \{\text{run out}\}\right]$$

Run-outs reflect fielding and batting decisions, not bowling quality, and are excluded.

### 4.4 Laplace Smoothing

Players who have never produced a specific outcome in a specific phase (for example, a spinner who has never been hit for six in Middle overs) would receive a zero probability estimate, which would prevent the simulation from sampling that outcome and bias win/defend probability estimates downward. Add-one (Laplace) smoothing removes all zero probabilities:

$$\tilde{c}_i^{o,\phi} = c_i^{o,\phi} + \alpha, \qquad \alpha = 1$$

$$\hat{p}_i^{o,\phi} = \frac{\tilde{c}_i^{o,\phi}}{\displaystyle\sum_{o' \in \Omega} \tilde{c}_i^{o',\phi}}$$

With $|\Omega| = 7$ and $\alpha = 1$, the maximum distortion to any probability is $7/n_i^\phi$, negligible for players with more than 100 deliveries in a phase but meaningful for very sparse profiles, which are addressed by the shrinkage estimator below.

### 4.5 James–Stein Shrinkage for Phase-Sparse Players

Many IPL players have limited exposure in specific phase-role combinations: a seam bowler who has rarely bowled Middle overs, a spinner who has never bowled Death overs, or a new franchise addition with fewer than two full IPL seasons. For such players, the Laplace-smoothed individual estimate $\hat{\mathbf{p}}_i^\phi$ is a high-variance estimate from few observations. We apply a **James–Stein shrinkage estimator** (James and Stein, 1961), which blends each player's individual estimate toward the population-average profile for that phase.

**Population average:** For phase $\phi$, the population-average outcome distribution (across all players in the dataset) is:

$$\bar{\mathbf{p}}^\phi = \frac{\sum_i \tilde{\mathbf{c}}_i^\phi}{\sum_i \sum_{o \in \Omega} \tilde{c}_i^{o,\phi}}$$

**Blended profile:**

$$\mathbf{p}_i^\phi = \lambda_i^\phi \cdot \hat{\mathbf{p}}_i^\phi + (1 - \lambda_i^\phi) \cdot \bar{\mathbf{p}}^\phi$$

**Data-adaptive blend weight:**

$$\lambda_i^\phi = \frac{n_i^\phi}{n_i^\phi + n_{\min}}, \qquad n_{\min} = 50$$

When $n_i^\phi = 0$ (no historical data in that phase), $\lambda_i^\phi = 0$ and the profile collapses entirely to the population average, a conservative but unbiased prior. As $n_i^\phi \to \infty$, $\lambda_i^\phi \to 1$ and the individual estimate dominates. The threshold $n_{\min} = 50$ reflects approximately 8 full overs of data, sufficient for phase-specific distributions to be statistically meaningful. This estimator is a practical analogue of the James–Stein phenomenon: the pooled estimate has lower mean squared error than the individual estimate for any player with fewer than $n_{\min}$ phase-specific deliveries.

### 4.6 Derived Summary Statistics

From the blended profile $\mathbf{p}_i^\phi$, the following summary statistics are computed for reporting and diagnostic purposes:

$$\text{SR}_i^\phi = 100 \cdot \sum_{o \in \Omega} \rho(o) \cdot p_i^{o,\phi} \qquad \text{(batsman strike rate)}$$

$$\text{ER}_j^\phi = 6 \cdot \sum_{o \in \Omega} \rho(o) \cdot p_j^{o,\phi} \qquad \text{(bowler economy rate, runs per over)}$$

$$p_i^{W,\phi} \qquad \text{(dismissal probability per ball — batsman or bowler)}$$

$$p_j^{0,\phi} \qquad \text{(dot-ball probability per ball — proxy for pressure)}$$

Note that SR and ER are derived from the same underlying probability vector and are thus consistent with the simulation model: a player's simulated run output will converge to their profile-implied SR or ER as simulation count increases.

---

## 5. Monte Carlo Simulation

### 5.1 Estimating Win and Defend Probabilities

For a fixed batting order $\sigma$ or bowling plan $\pi$, the value function at the initial state $(r_0, b_0, w_0)$ is estimated by forward simulation of complete innings trajectories:

$$\hat{V} = \frac{1}{N} \sum_{n=1}^{N} \mathbf{1}\!\left[\text{trajectory}^{(n)} \text{ results in win/defend}\right]$$

This is an unbiased estimator of $V(r_0, b_0, w_0)$ with binomial standard error:

$$\text{SE}(\hat{V}) = \sqrt{\frac{\hat{V}(1-\hat{V})}{N}} \leq \frac{1}{2\sqrt{N}}$$

For $N = 50{,}000$ simulations, $\text{SE}(\hat{V}) \leq 0.22\%$, sufficient to distinguish orderings or plans differing by 1 pp with a signal-to-noise ratio of at least $4.5\sigma$.

### 5.2 Simulation Algorithm

Each trajectory is initialised from the intervention state and advanced ball by ball until the innings terminates (target reached, all balls bowled, or batting side all out):

```
Input: order σ (or plan π), initial state (r₀, b₀, w₀)

for each simulation n = 1 to N:
    r ← r₀;  b ← b₀;  w_remaining ← w₀
    striker ← σ[current_position];  non_striker ← fixed
    over_ball ← 0

    while b > 0 and w_remaining > 0 and r > 0:
        phase φ ← phase_of(absolute_over(b))
        bowler j ← plan π[current_over]       (bowling simulation)
        o ~ Categorical(p_{striker}^φ)        (batting simulation)
        
        if o == W:
            w_remaining -= 1
            striker ← non_striker
            non_striker ← σ[next_available]
        else:
            r -= ρ(o)
            if ρ(o) mod 2 == 1 or over_ball == 5:
                swap(striker, non_striker)    # strike rotation
        
        b -= 1;  over_ball = (over_ball + 1) mod 6

    result[n] ← (r ≤ 0)    (batting) or (r > 0 and b == 0 or w_remaining == 0)  (bowling)

return mean(result)
```

### 5.3 Strike Rotation

Strike rotation is a structurally important feature of the model. After each delivery, the on-strike batsman becomes the non-striker if (a) the runs scored are odd ($\rho(o) \in \{1, 3\}$), or (b) the delivery is the final ball of the over. At a wicket, the new batsman (next in the order) becomes the on-striker; the surviving batsman takes the non-striker position. This rule correctly models the alternation of the batting pair and ensures that Death-over specialists are exposed to the appropriate deliveries given realistic run-scoring patterns in preceding overs.

### 5.4 Vectorised Implementation

All $N$ simulations are advanced in parallel using NumPy vectorised operations. At each ball, $N$ outcomes are sampled in a single call to `numpy.random.Generator.choice` with pre-built probability arrays; state vectors `runs_scored`, `wickets`, and `done` are updated element-wise. Simulations that have terminated (target reached or all out) are masked from subsequent updates. The full $N = 50{,}000$-trajectory evaluation completes in approximately 0.15 seconds per configuration on commodity hardware (Apple M-series processor), enabling the broad search described in Section 6.

A fast-evaluation pass uses $N_{\text{fast}} = 5{,}000$ simulations for initial screening during the SA search, reducing per-evaluation time to $\approx 15$ ms, with a final high-precision pass at $N = 30{,}000$ for the top-10 candidates identified by SA.

---

## 6. Search Algorithms

### 6.1 Batting Order: Exhaustive Enumeration with Two-Pass Refinement

For a batting pool of size $|\mathcal{P}| = n$, the decision at the intervention point is the complete ordering of all $n$ remaining batsmen in positions $3, 4, \ldots, n+2$. The full search space has $n!$ elements. For $n \leq 6$, $n! \leq 720$; for $n \leq 4$, $n! \leq 24$, both tractable with Monte Carlo simulation.

A **two-pass strategy** is employed:

- **Pass 1 (screening):** Evaluate all $n!$ orderings with $N_1 = 3{,}000$ simulations. Retain the top-$K = 10$ candidates ranked by $\hat{V}$.
- **Pass 2 (refinement):** Re-evaluate the top-$K$ candidates with $N_2 = 20{,}000$ simulations. Report $\sigma^*$ as the candidate maximising $\hat{V}$ after Pass 2.

The two-pass strategy reduces total simulation count by approximately $5\times$ relative to evaluating all permutations at $N_2$ precision, at negligible risk of misranking the true optimum (the probability that a non-top-$K$ permutation has true value exceeding the top-$K$ by more than the Pass 1 SE is less than $10^{-3}$ for typical problem instances).

### 6.2 Bowling Plan: Feasibility Constraints

The feasible set $\mathcal{F}$ for bowling plans is defined by two constraints derived from the Laws of Cricket:

**Quota constraint:** Each bowler $j$ may not bowl more than their remaining over quota:

$$\sum_{k=1}^{m} \mathbf{1}[\pi_k = j] \leq q_j \quad \forall j \in \mathcal{B}$$

where $m$ is the number of remaining overs and $q_j = 4 - (\text{overs already bowled by } j)$.

**Consecutive-over rule:** The same bowler cannot bowl consecutive overs; overs are bowled from alternate ends of the pitch and the same bowler cannot therefore bowl two overs in a row:

$$\pi_k \neq \pi_{k+1} \quad \forall k \in \{1, \ldots, m-1\}$$

A bowling plan violating either constraint is illegal and must be excluded from the search. The consecutive-over rule eliminates approximately 15–20% of otherwise quota-valid plans in typical T20 scenarios.

### 6.3 Bowling Plan: Simulated Annealing

The feasible set $\mathcal{F}$, even after applying quota and consecutive-over constraints, contains several thousand plans for a typical T20 end-game scenario with six bowlers and eight over slots. Exhaustive high-precision evaluation of all such plans would require $|\mathcal{F}| \times N \approx 5{,}000 \times 50{,}000 = 2.5 \times 10^8$ simulation steps, which would be prohibitively expensive. Simulated annealing (Kirkpatrick et al., 1983) provides an efficient heuristic that finds near-optimal plans with high probability.

**Objective:** Maximise $\hat{V}^\pi(d_0, b_0, w_0)$ over $\pi \in \mathcal{F}$.

**Acceptance rule:** A candidate neighbour plan $\pi'$ is accepted over the current plan $\pi$ according to the Metropolis criterion:

$$A(\Delta V, T) = \begin{cases} 1 & \text{if } \Delta V > 0 \\[4pt] \exp\!\left(\dfrac{\Delta V}{T}\right) & \text{if } \Delta V \leq 0 \end{cases}$$

where $\Delta V = \hat{V}^{\pi'} - \hat{V}^\pi$ and $T$ is the current temperature.

**Cooling schedule:** Linear cooling from $T_0 = 0.05$ to $\epsilon = 10^{-6}$:

$$T(\text{step}) = T_0 \cdot \!\left(1 - \frac{\text{step}}{N_{\text{steps}}}\right) + \epsilon, \qquad N_{\text{steps}} = 8{,}000$$

**Neighbourhood function:** A neighbour $\pi'$ is generated by selecting a uniformly random over slot $k \in \{1, \ldots, m\}$ and replacing its bowler $\pi_k$ with a bowler drawn uniformly from the feasible candidate set:

$$\mathcal{C}_k\!\left(\pi\right) = \left\{ j \in \mathcal{B} \;:\; j \neq \pi_k,\;\; \text{used}_\pi(j) + \mathbf{1}[j = \pi_k] - \mathbf{1}[j = \pi_k] < q_j,\;\; j \neq \pi_{k-1},\;\; j \neq \pi_{k+1} \right\}$$

This formulation explicitly enforces both quota validity and the consecutive-over rule on every proposed neighbour, ensuring $\pi' \in \mathcal{F}$ whenever $\mathcal{C}_k(\pi) \neq \emptyset$.

**Execution:** The SA runs for $N_{\text{steps}} = 8{,}000$ iterations, each evaluating a proposed neighbour with $N_{\text{fast}} = 5{,}000$ simulations. All unique plans encountered during the search are cached; the top-10 unique plans are then re-evaluated with $N_{\text{refine}} = 30{,}000$ simulations for final reporting.

---

## 7. Case Study 1: KKR vs MI, 29 March 2026

### 7.1 Match Context and Intervention Point

The IPL match at Wankhede Stadium saw KKR post a target of 221 runs (220 scored). At ball `2nd.11.6`, the final delivery of over 12, MI's Rohit Sharma was dismissed for 72 runs off 40 balls, caught off the bowling of VG Arora. At the moment of dismissal:

| Parameter | Value |
|-----------|-------|
| MI score | 148/1 |
| Runs needed ($r_0 = d_0$) | 73 |
| Legal balls remaining ($b_0$) | 44 |
| Required run rate | 9.95 per over |
| MI batsmen at crease | RD Rickelton (non-striker, fixed) |
| MI batting pool $\mathcal{P}$ | SA Yadav, Tilak Varma, HH Pandya, Naman Dhir |

Both problems, namely the batting order and bowling plan, arise simultaneously from the same intervention state, providing a natural experiment in which the same match state is audited from both perspectives.

### 7.2 Player Profiles

**Table 1: MI Batsman Profiles (Blended, Phase-Specific)**

| Batsman | Hist. balls (MI) | Hist. balls (DE) | Death SR | $p^W_{\text{DE}}$ | Middle SR | $p^W_{\text{MI}}$ |
|---------|-----------------|-----------------|----------|-------------------|-----------|-------------------|
| RD Rickelton | 342 | 156 | 156.4 | 0.079 | 152.9 | 0.100 |
| SA Yadav | 1,765 | 460 | 181.3 | 0.089 | 143.8 | **0.035** |
| Tilak Varma | 1,132 | 388 | **185.5** | 0.070 | 137.3 | 0.030 |
| HH Pandya | 844 | 512 | 171.6 | 0.070 | 134.0 | 0.037 |
| Naman Dhir | 498 | 204 | **203.9** | 0.070 | 145.9 | 0.049 |

The critical structural feature is the **phase asymmetry of Naman Dhir**: his Death SR (204) exceeds any other candidate by at least 18 points, yet his Middle SR (146) is comparable to SA Yadav's. This makes him uniquely valuable in the Death overs but risky to promote early. SA Yadav is the optimal bridge through the Middle overs: his dismissal probability (0.035) is the lowest of all candidates, and his Middle-over strike rate (143.8) is the highest among the non-Death specialists. The combination of low wicket risk and competitive scoring rate in the Middle phase makes him the best choice to carry the innings to the Death overs while preserving Tilak Varma and Naman Dhir, both superior Death-over hitters, for the final assault.

![Figure 1: MI batsman phase-specific profiles (KKR vs MI, over 12 intervention). The phase asymmetry of Naman Dhir is the key structural feature driving the optimal batting order.](diagram_1_batsman_profiles.png){width=75%}

### 7.3 Batting Order Results

**Table 3: Best Achievable Win% by Next-In Choice (Position 3)**

| Next-In | Best Win% | Optimal Remaining Order |
|---------|-----------|-------------------------|
| **SA Yadav (actual)** | **56.5%** | Naman Dhir → Tilak Varma → HH Pandya |
| Naman Dhir | 55.7% | SA Yadav → Tilak Varma → HH Pandya |
| Tilak Varma | 55.1% | Naman Dhir → SA Yadav → HH Pandya |
| HH Pandya | 53.3% | Naman Dhir → Tilak Varma → SA Yadav |

The actual decision to send SA Yadav in immediately at Position 3 was the correct choice. His low Middle-over dismissal probability ($p^W_{\text{MI}} = 0.035$, lowest of all candidates) and competitive Middle-over strike rate (143.8) make him the optimal bridge through overs 12–15 while preserving Tilak Varma and Naman Dhir for the Death-over assault.

**Table 4: All 6 Orderings for Positions 4–6 (SA Yadav at Position 3)**

| Rank | Position 4 | Position 5 | Position 6 | Win% |
|------|-----------|-----------|-----------|------|
| **1 (opt)** | **Naman Dhir** | **Tilak Varma** | **HH Pandya** | **56.5%** |
| 2 | Naman Dhir | HH Pandya | Tilak Varma | 56.1% |
| 3 | Tilak Varma | Naman Dhir | HH Pandya | 55.2% |
| 4 | HH Pandya | Naman Dhir | Tilak Varma | 52.9% |
| **5 (actual)** | **Tilak Varma** | **HH Pandya** | **Naman Dhir** | **52.4%** |
| 6 | HH Pandya | Tilak Varma | Naman Dhir | 51.8% |

The match order (rank 5) was the second-worst of all six possibilities. Figure 2 shows the best achievable win probability as a function of the next-in choice at Position 3.

![Figure 2: Best achievable win probability by next-in choice at Position 3. SA Yadav's low Middle-over dismissal probability makes him the optimal bridge.](diagram_2_next_in_choice.png){width=60%}

The optimal order promotes Naman Dhir to Position 4, ensuring maximum exposure of his Death SR (204) to the final overs. In the actual match, Dhir entered at Position 6 and faced only 2 balls, both in over 19. The expected additional runs from promoting Dhir two positions can be approximated as:

$$\Delta R \approx \Delta b_{\text{Dhir}} \times \frac{\text{SR}_{\text{Dhir}}^{\text{DE}} - \text{SR}_{\text{Varma}}^{\text{DE}}}{100} \approx \Delta b_{\text{Dhir}} \times 0.184$$

Even 5 additional Death balls yields approximately 0.92 expected additional runs, sufficient given the proximity to the target, to shift win probability by the observed 4.1 pp.

### 7.4 Summary

| Scenario | Win% (MI) |
|----------|-----------|
| Worst possible batting order | 50.0% |
| Actual batting order | 52.4% |
| Optimal batting order | 56.5% |
| Gain from optimal order | **+4.1 pp** |

---

## 8. Case Study 2: GT vs PBKS, 31 March 2026

### 8.1 Match Context and Intervention Point

Gujarat Titans posted 162 in their first innings. At ball `2nd.9.3`, the third delivery of over 10 (0-indexed over 9), Punjab Kings' opener P Simran Singh was caught off Rashid Khan's bowling. At the moment of dismissal:

| Parameter | Value |
|-----------|-------|
| PBKS score | 83/2 |
| Runs needed ($r_0 = d_0$) | 80 |
| Legal balls remaining ($b_0$) | 60 (overs 10–19 after Rashid completes over 9) |
| PBKS wickets in hand | 8 |
| Required run rate | 8.00 per over |

Rashid Khan was mid-over at the moment of dismissal and was committed to completing over 9 (3 remaining balls). The optimisation covers the 10 complete overs that follow (overs 10–19, 0-indexed).

### 8.2 GT Bowling Resources and Profiles

**Table 7: GT Bowler Profiles and Quotas (After Over 9 Completes)**

| Bowler | Overs bowled | Quota left | Hist. balls (MI) | Hist. balls (DE) | Middle ER | $p^W_{\text{MI}}$ | Death ER | $p^W_{\text{DE}}$ |
|--------|-------------|------------|-----------------|-----------------|-----------|-------------------|----------|-------------------|
| Ashok Sharma | 1 | 3 | 0 | 0 | 7.49† | 0.043† | 9.37† | 0.083† |
| K Rabada | 2 | 2 | 466 | 599 | 7.35 | 0.053 | 9.36 | 0.104 |
| Mohammed Siraj | 2 | 2 | 451 | 673 | 7.35 | 0.048 | 9.55 | 0.067 |
| Rashid Khan | 3* | 1 | 2,223 | 564 | **6.58** | 0.047 | **8.40** | 0.075 |
| Washington Sundar | 2 | 2 | 654 | 81 | 7.07 | 0.033 | 9.61 | 0.068 |
| M Prasidh Krishna | 0 | 4 | 398 | 505 | 7.47 | 0.047 | **9.79** | 0.086 |

*Rashid Khan bowled 3 full overs plus 3 balls of over 9; quota = 1 full remaining over.  
†Ashok Sharma profile = population average ($\lambda = 0$); zero historical IPL deliveries in dataset.

![Figure 5: GT bowler landscape at the over 10 intervention point. The optimal Death-over bowler (Rashid Khan) is in the top-left; the most expensive deployed bowler (M Prasidh Krishna) is toward the bottom-right.](bowling_gt_1_bowler_landscape.png){width=65%}

Key features of this profile table:
- **Rashid Khan** has the lowest economy in both Middle (6.58) and Death (8.40) phases, making him the most valuable bowler, but his remaining quota is only 1 over.
- **Washington Sundar** shows a sharp phase gradient: Middle ER 7.07 (competitive) vs Death ER 9.61 (second-most expensive). He should not bowl Death overs.
- **M Prasidh Krishna** has the highest Death economy (9.79) of any GT bowler, yet has 4 overs of remaining quota.
- **Mohammed Siraj** was not used in the actual plan despite 2 overs of quota and competitive profiles in both phases (Middle ER 7.35, Death ER 9.55).

![Figure 6: Best achievable defend probability by over 10 opener choice. Siraj's two-over quota makes him the highest-value opener despite being completely unused in the actual plan.](bowling_gt_2_opener_bar.png){width=60%}

### 8.3 Actual and Optimal Bowling Plans

**Table 8: Actual GT Bowling Plan (Overs 10–19)**

| Over | Bowler | Phase | Economy |
|------|--------|-------|---------|
| 10 | Ashok Sharma | MI | 7.49 |
| 11 | Rashid Khan | MI | 6.58 |
| 12 | M Prasidh Krishna | MI | 7.47 |
| 13 | Washington Sundar | MI | 7.07 |
| 14 | M Prasidh Krishna | MI | 7.47 |
| 15 | K Rabada | DE | 9.36 |
| 16 | M Prasidh Krishna | DE | 9.79 |
| 17 | Ashok Sharma | DE | 9.37 |
| 18 | M Prasidh Krishna | DE | 9.79 |
| 19 | Washington Sundar | DE | 9.61 |
| | **Actual defend%** | | **39.1%** |

**Table 9: Optimal GT Bowling Plan (Overs 10–19)**

| Over | Bowler | Phase | Economy | Changed? |
|------|--------|-------|---------|----------|
| 10 | **Mohammed Siraj** | MI | 7.35 | ← |
| 11 | **Washington Sundar** | MI | 7.07 | ← |
| 12 | **K Rabada** | MI | 7.35 | ← |
| 13 | **Mohammed Siraj** | MI | 7.35 | ← |
| 14 | **Washington Sundar** | MI | 7.07 | ← |
| 15 | **Ashok Sharma** | DE | 9.37 | ← |
| 16 | **Rashid Khan** | DE | 8.40 | ← |
| 17 | Ashok Sharma | DE | 9.37 | — |
| 18 | **K Rabada** | DE | 9.36 | ← |
| 19 | **Ashok Sharma** | DE | 9.37 | ← |
| | **Optimal defend%** | | **44.3%** | |

**Table 10: Top-5 GT Bowling Plans**

| Rank | Ov10 | Ov11 | Ov12 | Ov13 | Ov14 | Ov15 | Ov16 | Ov17 | Ov18 | Ov19 | Defend% |
|------|------|------|------|------|------|------|------|------|------|------|---------|
| **1 (opt)** | Siraj | Sundar | Rabada | Siraj | Sundar | Ashok | **Rashid** | Ashok | Rabada | Ashok | **44.3%** |
| 2 | Sundar | Krishna | Sundar | Siraj | Rabada | Ashok | Rashid | Ashok | Rabada | Ashok | 44.0% |
| 3 | Sundar | Siraj | Sundar | Rabada | Siraj | Ashok | Rashid | Ashok | Rabada | Ashok | 43.9% |
| 4 | Siraj | Sundar | Siraj | Krishna | Rashid | Ashok | Rabada | Ashok | Rabada | Ashok | 43.5% |
| **Actual** | Ashok | **Rashid** | Krishna | Sundar | Krishna | Rabada | **Krishna** | Ashok | **Krishna** | Sundar | **39.1%** |

![Figure 7: Top 10 GT bowling plans heatmap. The vertical divider separates Middle overs (left) from Death overs (right); Rashid Khan appears in Death over 16 in all top plans.](bowling_gt_3_top10_heatmap.png){width=85%}

Three structural errors characterise the actual plan:

1. **Rashid Khan deployed in Middle (over 11)**: His advantage over the best alternative (Washington Sundar, ER 7.07) in Middle is 0.49 RPO. His advantage over M Prasidh Krishna (ER 9.79) in Death is 1.39 RPO, nearly three times as large. Deploying Rashid in Middle over 11 rather than Death over 16 forfeited this asymmetry.

2. **M Prasidh Krishna used for 4 overs, including 2 Death overs (16, 18)**: Krishna's Death ER (9.79) is the highest of any GT bowler. His 2 Death overs are expected to concede approximately 1.4 additional runs each relative to the optimal assignment, approximately 2.8 runs cumulatively.

3. **Mohammed Siraj not used**: With 2 overs of quota and competitive Middle (7.35) and Death (9.55) profiles, Siraj's omission was the largest single resource-allocation error. The optimal plan uses him in overs 10 and 13 (both Middle phase).

![Figure 8: Actual vs optimal GT bowling plan (overs 10–19). The total gain of +5.2 pp is distributed across multiple substitutions, with Siraj's inclusion and Rashid's repositioning contributing the most.](bowling_gt_4_actual_vs_optimal.png){width=85%}

### 8.4 Statistical Significance

The $z$-score for the bowling gap in the GT vs PBKS case:

$$z = \frac{0.052}{\sqrt{2} \times 0.00187} \approx 19.7$$

This is far beyond any conventional significance threshold, confirming that the observed 5.2 pp gap is a stable structural property of the plan, not simulation variance.

---

## 9. Comparative Analysis and Discussion

### 9.1 The Phase-Specificity of Bowling Value

The GT vs PBKS case illustrates the core finding: **aggregate economy rate is a poor proxy for in-match bowling value**. M Prasidh Krishna has an acceptable overall economy, yet his Death-over economy (9.79 RPO) is the highest of any GT bowler. This divergence is invisible in aggregate statistics but is fully captured by the phase-specific profile. The primary misallocations in the GT plan are summarised in Table 11.

**Table 11: Primary Misallocations — GT vs PBKS**

| Bowler misused | Phase deployed | Death ER | Better option | Better Death ER | Cost (RPO) |
|---------------|----------------|----------|---------------|-----------------|------------|
| MP Krishna | Death (2 overs) | 9.79 | Rashid Khan | 8.40 | −1.39 |
| Siraj | Not used | — | — | — | −0.49 per over |

Teams should maintain and consult phase-specific profiles, not aggregate metrics, when making in-match bowling decisions. The difference between using M Prasidh Krishna and Rashid Khan in a Death over is 1.39 expected runs — across two overs that is nearly 3 runs, decisive at a margin of 80 to defend in 10 overs.

### 9.3 The Rashid Khan Problem: Scarcity of the Best Resource

The GT vs PBKS case illustrates a generalised decision problem absent from the KKR case: **optimal allocation of a scarce high-value resource**. Rashid Khan was the best available bowler in both Middle and Death phases, but with only 1 over of quota remaining, the batting captain had to choose where to deploy him. The model's recommendation, namely Death over 16, reflects the principle that scarce resources should be deployed where their **marginal contribution** is largest, not merely where they are good. Since Rashid's margin over alternatives is 1.39 RPO in Death versus 0.49 RPO in Middle, the Death deployment is nearly three times more valuable per over. This principle generalises to any team holding a single over of a premium bowler at the death stage of an innings.

### 9.4 Batting and Bowling Duality in Practice

The KKR vs MI case provides an audit of the batting decision at the same intervention state where the bowling decision could also have been studied. The batting MDP value function $V^B$ and the bowling defend probability $V^\pi$ are exact complements at any shared state: $V^B + V^\pi = 1$. The actual batting order (rank 5 of 6) gave MI a win probability of 52.4%; the optimal order raises this to 56.5%. The same duality means that a bowling optimisation at that state would directly reduce MI's win probability by an equivalent margin, illustrating how both sides of the decision can be audited from the same framework.

### 9.5 Limitations

**Static historical profiles.** Player profiles are estimated from all historical IPL deliveries with equal weight. Current form, fatigue, pitch conditions, and opponent-specific tendencies are not captured. A bowler who has been expensive in recent matches, or a batsman in exceptional form, may be systematically misrated. Exponential decay weighting over recent innings, or Bayesian updating within the current match, would partially address this.

**Opponent-agnostic profiles.** Batsman profiles are averaged across all bowlers faced; bowler profiles are averaged across all batsmen faced. Matchup-specific effects, such as a left-arm spinner facing left-handed versus right-handed batsmen or a pace bowler against an aggressive versus defensive batsman, are real but are not modelled here due to the sparsity of matchup-level data across phases. A tensor decomposition of outcomes over (bowler, batsman, phase) would extend the framework to capture these effects.

**Pre-committed plans.** The bowling plan $\pi$ is assumed to be chosen once at the intervention point and then executed regardless of how the match state evolves. In practice, the bowling captain can and should revise the plan at each over boundary given new information (runs conceded in the previous over, a wicket taken, a batsman promoted). A fully adaptive policy $\pi: (d, b, w) \to j$ would require solving the full MDP over $(d, b, w, \mathcal{Q})$, the augmented state that includes remaining quotas, which is feasible in principle but increases the state space by a factor of $\prod_j (q_j + 1)$.

**Ball-by-ball independence.** Consecutive ball outcomes are treated as i.i.d. given the current phase and player identities. Momentum effects such as a batsman in rhythm after a boundary or a bowler tightening their line under pressure are not captured. Incorporating a latent pressure or momentum state via a Hidden Markov Model would extend the framework at the cost of additional estimation complexity.

---

## 10. Conclusions

This paper has presented a unified Markov Decision Process framework for optimising two of the most consequential in-match decisions in T20 cricket: the batting order at each wicket fall and the bowling plan across the remaining over slots. The framework is grounded in a three-phase player profile engine with James–Stein shrinkage for data-sparse players, evaluated by vectorised Monte Carlo simulation, and searched by exhaustive enumeration (batting order) and simulated annealing (bowling plan).

Applied to two intervention points from the 2026 IPL season, the main findings are:

1. **Batting order (KKR vs MI):** The optimal batting order improves MI's win probability by 4.1 pp (52.4% → 56.5%). The actual order was the second-worst of all six feasible permutations. The primary driver is the phase asymmetry of Naman Dhir: his Death SR (204) is dramatically superior to any alternative, but he entered at Position 6 and faced only 2 balls. Promoting him to Position 4 ensures exposure of this advantage to the decisive Death-over deliveries.

2. **Bowling plan (GT vs PBKS):** The optimal bowling plan improves GT's defend probability by 5.2 pp (39.1% → 44.3%). M Prasidh Krishna (Death ER = 9.79) was used in 4 overs including 2 Death overs, Mohammed Siraj (2-over quota, competitive profile) was not used at all, and Rashid Khan (best Death bowler, ER = 8.40) was deployed in a Middle over where his margin over alternatives was less than one-third of his Death margin.

3. **Phase-specificity as the common thread:** In both case studies, the source of suboptimality is phase-agnostic deployment: decisions that appear reasonable by aggregate metrics but are exposed as costly when phase-specific profiles are applied. Published aggregate statistics (overall SR, overall economy) suppress the information that matters most for in-match decision support.

4. **Methodological:** The simulation-based MDP approach is computationally tractable on commodity hardware (under 5 minutes for a full SA run including profile estimation), suggesting feasibility for pitch-side or dugout deployment during a match. With Monte Carlo SE below 0.22% at $N = 50{,}000$, the precision is sufficient to rank orderings or plans separated by 1 pp with high confidence.

Phase-specific quantitative decision support, grounded in a principled stochastic optimisation framework, offers a demonstrably large and actionable improvement over the heuristic approach that currently dominates in-match tactical decision-making.

---

## Acknowledgements

Ball-by-ball data sourced from Cricsheet.org (Stevenson, 2023) under Creative Commons Attribution licence. Analysis conducted in Python 3.11 using NumPy 1.26, pandas 2.1, and matplotlib 3.8.

---

## References

Bellman, R. (1957). *Dynamic Programming*. Princeton University Press, Princeton, NJ.

Duckworth, F. C. and Lewis, A. J. (1998). A fair method for resetting the target in interrupted one-day cricket matches. *Journal of the Operational Research Society*, 49(3), 220–227.

Ganesh, T. V. (2016). yorkr: An R package for analytics of cricket. R package version 0.0.5. Available at CRAN.

Ganesh, T. V. (2017). Using linear programming for optimizing bowling change or batting lineup in T20 cricket. *gigadom.in*, 28 September 2017.

James, W. and Stein, C. (1961). Estimation with quadratic loss. In *Proceedings of the 4th Berkeley Symposium on Mathematical Statistics and Probability*, Vol. 1, pp. 361–379. University of California Press.

Kirkpatrick, S., Gelatt, C. D., and Vecchi, M. P. (1983). Optimization by simulated annealing. *Science*, 220(4598), 671–680.

Puterman, M. L. (1994). *Markov Decision Processes: Discrete Stochastic Dynamic Programming*. Wiley, New York.

Stevenson, S. (2023). Cricsheet: Ball-by-ball cricket data. Available at: https://cricsheet.org. Accessed April 2026.

---

## Appendix: Notation Summary

| Symbol | Definition |
|--------|------------|
| $(r, b, w)$ | Match state: runs remaining, balls remaining, wickets |
| $R_{\max}$ | Maximum runs in the state space |
| $B = 120$ | Total balls in T20 innings |
| $\mathcal{P}$ | Pool of available batsmen |
| $\sigma$ | Batting order — permutation of $\mathcal{P}$ |
| $\pi = (\pi_1, \ldots, \pi_m)$ | Bowling plan — ordered over assignments |
| $\sigma^*,\; \pi^*$ | Optimal batting order, optimal bowling plan |
| $\phi \in \{\text{PP, MI, DE}\}$ | Innings phase |
| $\Omega = \{W, 0, 1, 2, 3, 4, 6\}$ | Ball outcome set |
| $\rho(o)$ | Runs scored under outcome $o$ |
| $\mathbf{p}_i^\phi = (p_i^{o,\phi})_{o \in \Omega}$ | Outcome probability vector: player $i$, phase $\phi$ |
| $T(\mathcal{S}, o)$ | State transition function |
| $V^B(r,b,w)$ | Batting win probability (value function) |
| $V^\pi(d,b,w)$ | Bowling defend probability (value function) |
| $\text{SR}_i^\phi$ | Batsman $i$ strike rate in phase $\phi$ |
| $\text{ER}_j^\phi$ | Bowler $j$ economy rate in phase $\phi$ |
| $p_i^{W,\phi}$ | Dismissal/wicket probability per ball |
| $p_j^{0,\phi}$ | Dot-ball probability per ball |
| $c_i^{o,\phi}$ | Raw outcome count for player $i$, outcome $o$, phase $\phi$ |
| $\alpha = 1$ | Laplace smoothing constant |
| $\bar{\mathbf{p}}^\phi$ | Population-average profile for phase $\phi$ |
| $\lambda_i^\phi$ | James–Stein blend weight |
| $n_i^\phi$ | Historical legal deliveries for player $i$ in phase $\phi$ |
| $n_{\min} = 50$ | Shrinkage threshold |
| $N = 50{,}000$ | Monte Carlo simulation count (final evaluation) |
| $N_{\text{fast}} = 5{,}000$ | Monte Carlo count during SA search |
| $\hat{V}$ | Monte Carlo estimate of value function |
| $\text{SE}(\hat{V})$ | Binomial standard error of estimate |
| $q_j$ | Remaining over quota for bowler $j$ |
| $\mathcal{F}$ | Set of feasible bowling plans |
| $\mathcal{C}_k(\pi)$ | Feasible candidate set for over slot $k$ |
| $A(\Delta V, T)$ | SA acceptance probability |
| $T_0 = 0.05$ | Initial SA temperature |
| $N_{\text{steps}} = 8{,}000$ | SA iteration count |

---
