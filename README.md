# Castle strategy simulations

Python simulations and optimizers for two allocation games: distribute exactly **100 integer soldiers** across five castles worth **1, 2, 3, 4 and 5 points**.

The code and analysis were developed with AI assistance. No actual applicant submissions were available. Opponent populations, strategic shares and counter-strategy frequencies are explicit modeling assumptions. The results are conditional recommendations, not forecasts or universally optimal answers.

## The two scoring objectives

**Game 1 — head to head.** Play every other applicant. At each castle, the higher bid wins its value; ties split it. Maximize average castle points, out of 15.

**Game 2 — four-way battle.** In each reshuffle, compete against three distinct other applicants. The highest castle bid wins; tied leaders split its value. Rank the four players by total castle points, then award **9, 4, 2 and 0** finishing points. Tied finishes share the payouts of the places they occupy. Maximize average finishing payout, out of 9.

These are different objectives. More expected castle points need not produce a higher expected finishing payout in Game 2.

## Run locally

Development and validation used Python 3.14.6 and NumPy 2.5.1.

```sh
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m unittest discover -s tests -v
python3 -u outputs/live_simulation.py
```

The last command runs 1,000 rounds per scenario, prints progress immediately, and writes its results beside the script. It usually finishes quickly; there are no artificial delays.

To run against real applicant allocations, provide a CSV with five integer columns summing to 100 per row. Rows represent **other** applicants; exclude your own submission. An optional header is supported.

```sh
python3 -u outputs/live_simulation.py --opponents applicants.csv
```

With a CSV, Game 1 evaluates every row exactly once, so its match count follows the number of applicants. Game 2 runs 1,000 reshuffles and draws three distinct row identities per group. Different applicants can legitimately submit identical allocations. For supplied data, there is no held-out population: Game 2 uses independent reshuffles of the same known population.

## Reproduce the successive experiments

Run commands from this repository's root. Each command overwrites the corresponding result files in `outputs/`. Run later experiments after earlier ones, or use the included snapshots of their results.

```sh
# 1. Two-player self-play plus hypothetical applicant populations
python3 -u outputs/blotto_ai.py --iterations 50000 --samples 100000

# 2. Search the four-player finishing-payout objective
python3 -u outputs/four_way_ai.py --train 8000 --test 200000

# 3. A short run using 1,000 rounds per scenario
python3 -u outputs/live_simulation.py

# 4. Add plausible strong fixed answers and their counters
python3 -u outputs/predicted_opponents_simulation.py

# 5. Eight adaptive counter rounds, five training seeds, ten fresh test seeds
python3 -u outputs/adaptive_simulation.py

# 6. Battle published picks from four exact-match public repositories
python3 -u outputs/public_opponent_battle.py
```

The offline `outputs/castle-allocation-lab.html` is an interactive **Game 1** tool. Open it directly in a browser. Its embedded results are a snapshot; rerunning Python does not rewrite the HTML. Import a newly generated `results.json` to update it. It does not calculate Game 2 payouts.

## How the strategy changed

1. Start with even, proportional, value-focused and random opponent models.
2. For Game 1, use exact dynamic programming to find the best response to an empirical opponent distribution. It is equivalent to evaluating all **4,598,126** legal allocations for that additive objective.
3. For Game 2, score whole groups and search random allocations, every two-castle split, and local troop transfers. Castle marginals alone cannot determine a rank-based payoff.
4. Add copied strong answers, nearby variations and learned counters. Test strategic shares of 0%, 25%, 50% and 75%; these shares are assumptions, not measured frequencies.
5. Let opponents adapt again, select candidates using training data, freeze them, then evaluate on independent populations and group draws. Report the tradeoffs rather than selecting the best-looking test result.

See [the detailed process](docs/process.md) and [the concise application answer](docs/strategy-answer.md).

## Public-repository comparison

Four external repositories publish concrete picks for these exact questions. Sources and their allocations are pinned to commits in `outputs/public_opponents.json`; we evaluate the allocation facts with our own scorer and do not execute third-party code.

- **Game 1:** `4,18,31,45,2` is an exact best response to the four published Game 1 picks, scoring 10, 14, 10 and 10: **11/15** average. Our previous adaptive-model pick averages 7/15 against those same picks.
- **Game 2:** `19,9,0,62,10` averages **7.75/9** against all four possible groups of three distinct sources: two outright firsts and two tied firsts. Exhaustively checking all 4,598,126 legal allocations proves this is optimal for that literal four-source field; **495 moves tie** at the optimum. Our previous adaptive-model pick averages 0.5/9 there.

These are targeted counters, not universal recommendations. The report separately tests fields in which some applicants copy the public picks and others use the ordinary synthetic model. A targeted counter can perform worse when its targets are rare. Public repositories are not a representative sample of actual applicant submissions.

Read [the sourced battle report](docs/public-repository-battle.md) for the rival answers, exact group outcomes, fresh 1,000-round checks, ten-seed sensitivity results and limitations.

## Adaptive-opponent results before the public-repository comparison

Under the **expanded adaptive-opponent model**, with equal weight on the four strategic-share scenarios:

| Game | Selected allocation, castles 1–5 | Independent test average | Previous recommendation on the same test data |
| --- | --- | --- | --- |
| 1 | `2, 3, 5, 41, 49` | 9.783 / 15 | 8.897 / 15 |
| 2 | `1, 41, 0, 58, 0` | 6.548 / 9 | 5.166 / 9 |

Both improved over the previous recommendation in all ten test seeds **within that expanded model**. Each seed/scenario contains 1,000 rounds. This is not evidence that those allocations beat every possible strategy or predict actual applicant behavior.

The shortlisted choices with the highest minimum training-scenario average were `2,15,2,37,44` for Game 1 and `20,0,0,0,80` for Game 2. They trade average score for more even performance across the modeled scenarios. This is not a global minimax guarantee.

## Files and validation

- `outputs/*.py`: runnable simulations and optimizers; local imports require these scripts to stay together.
- `outputs/*results.json`: machine-readable snapshots, seeds, assumptions, selected moves and evaluation metrics.
- `outputs/strategy_mix.csv`: the Game 1 self-play policy, sampled by probability rather than uniformly over distinct rows.
- `docs/`: process explanation, historical reports, and a ready-to-paste strategy answer.
- `tests/test_scoring.py`: budget validity, castle ties, finishing ties, score conservation, exact dynamic-program checks and sampled-policy checks.

No credentials, applicant submissions or attachment screenshots are included. No external model API is required. These are algorithmic simulations: dynamic programming, Monte Carlo sampling, heuristic search and fictitious play.

Game 1's learned mixed policy has a checked worst-case **expected** score of 7.443/15 in the initial run, close to the symmetric game's 7.5 benchmark. That guarantee belongs to the full private randomized policy, not to one fixed allocation. No corresponding equilibrium guarantee is claimed for Game 2.

Global optimality is established only for the stated Game 1 empirical additive objectives. Game 2 search is heuristic. Cross-seed confidence intervals describe variation inside the assumed model, not uncertainty about real human choices. New counter-strategies can change the recommendation again.
