# T20 Cricket: Optimising Batting Order and Bowling Plans

T20 cricket is as much about strategy as it is about skill and talent. In the modern era of T20, the margin between winning and losing often comes down to a single decision made in the heat of the moment — which batsman to send in after a wicket falls, or which bowler to bring on to defend a total in the final overs.

When a batting team is chasing a target, the captain has to decide who comes in next. Send in someone too defensive and you lose the run rate. Send in someone too aggressive and you risk losing a wicket at the wrong time. The right choice depends on how many runs are still needed, how many balls are left, and how many wickets are in hand — not just on who the best batsman is in general.

The bowling side faces a similar dilemma. With several bowlers still having overs left in their quota, the captain must decide who bowls when. A bowler who takes wickets regularly might be expensive in runs. A bowler with a great economy rate might not create enough pressure. And there are hard rules to follow — no bowler can bowl more than four overs, and the same bowler cannot bowl two overs in a row.

This paper approaches both problems using a technique called Markov Decision Programming. The idea is to build a value function — essentially a table that tells you, for any combination of runs remaining, balls remaining, and wickets in hand, what the probability of winning (or successfully defending) is. The value at any point in the innings depends only on the current state, not on how you got there. This is the Markov property — it does not matter whether you reached 50 runs needed off 30 balls via boundaries or singles; the situation is the same either way. At each delivery, one of seven things can happen: a wicket falls, or between 0 and 6 runs are scored. This tree-like structure collapses into a compact table, making the computation tractable.

Player profiles are built from ball-by-ball data covering 1,161 IPL matches from 2008 to 2025, sourced from [Cricsheet.org](https://cricsheet.org) via the [yorkr R package](https://cran.r-project.org/package=yorkr). The dataset comprises 273,735 total delivery records; after filtering for legal deliveries (excluding wides and no-balls) approximately 264,800 legal balls remain for profile estimation. All matches involving the target teams (KKR–MI on 29 March 2026 and GT–PBKS on 31 March 2026) are excluded to prevent any look-ahead bias. Crucially, profiles are split by phase — Powerplay, Middle overs, and Death overs — because a bowler's economy rate in overs 1–6 can be very different from what they concede in overs 16–20. Win and defend probabilities are estimated by simulating 50,000 innings for each candidate batting order or bowling plan.

The framework is applied to two matches from the 2026 IPL season. In the KKR vs Mumbai Indians match, we look at the batting order MI should have used when chasing 73 off 44 balls after Rohit Sharma's dismissal. In the GT vs Punjab Kings match, we look at the bowling plan GT should have deployed to defend 80 off 60 balls. In both cases, the optimal approach differed meaningfully from what actually happened on the field — and the gap in win/defend probability was large enough to have changed the outcome.

The paper is submitted to the Journal of Quantitative Analysis in Sports. The code and data for both case studies are in this repository.

## References
1. [Using Linear Programming (LP) for optimizing bowling change or batting lineup in T20 cricket](https://gigadom.in/2017/09/28/using-linear-programming-lp-for-optimizing-bowling-change-or-batting-lineup-in-t20-cricket/)
2. **Introducing cricket package yorkr** [1](https://gigadom.in/2016/04/02/introducing-cricket-package-yorkr-part-1-beaten-by-sheer-pace/),[2](https://gigadom.in/2016/04/03/introducing-cricket-package-yorkr-part-2-trapped-leg-before-wicket/),[3](https://gigadom.in/2016/04/07/introducing-cricket-package-yorkr-part-3-foxed-by-flight/),[4](https://gigadom.in/2016/04/12/introducing-cricket-package-yorkrpart-4-in-the-block-hole/)
3. [Natural language processing: What would Shakespeare say?](https://gigadom.in/2015/10/02/natural-language-processing-what-would-shakespeare-say/)


