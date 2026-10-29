# Game 2: four-way battle

Recommended fixed move for the assumed crowd: **0, 0, 36, 0, 64**, in castle order 1–5.

The search optimizes the actual finishing payouts (9, 4, 2, 0), including both castle ties and tied rankings. It does not optimize average castle points.

## Independent test results

The chosen move was selected using training and validation groups before the final test set of 200,000 groups was evaluated.

- **Selected Game 2 move** — [0, 0, 36, 0, 64]: **7.5166/9** expected finishing payout; **71.80%** outright first; **7.31%** tied for first.
- **Game 1 recommendation** — [1, 1, 26, 32, 40]: **5.7968/9** expected finishing payout; **49.75%** outright first; **5.02%** tied for first.
- **Earlier suggestion** — [4, 11, 21, 28, 36]: **4.0378/9** expected finishing payout; **27.76%** outright first; **5.92%** tied for first.
- **Even split** — [20, 20, 20, 20, 20]: **1.7519/9** expected finishing payout; **4.97%** outright first; **3.02%** tied for first.
- **Proportional** — [7, 13, 20, 27, 33]: **3.2556/9** expected finishing payout; **19.75%** outright first; **5.23%** tied for first.
- **Two high castles** — [0, 0, 0, 45, 55]: **7.2261/9** expected finishing payout; **67.53%** outright first; **5.04%** tied for first.

The selected move improves on Game 1's recommendation by **1.7198 finishing points per group** on the same test groups. An approximate 95% paired Monte Carlo interval for that gain is [1.7030, 1.7366]. This interval covers simulation noise conditional on the synthetic population pools; it does not quantify uncertainty about real applicants.

## Why concentrate on castles 3 and 5?

Winning these two castles outright gives 8 of the 15 castle points, guaranteeing first place because all other players together have only 7. A 36/64 split trades off reliable ownership of the 5-point castle against enough strength at the 3-point castle. The exact split is sensitive to the opponent assumptions; nearby splits performed similarly on validation.

Under this model, concentrating on castles 4 and 5 earns more average castle points than the selected move, yet a lower finishing payout. That is why reusing the Game 1 objective would give the wrong answer.

## Search and assumptions

The search evaluated **7,921 distinct allocations** against 8,000 fixed training groups. Initial candidates included random allocations with multiple concentration levels, every two-castle integer split, and the earlier recommendations. Multiple diverse starts were refined by moving troops between castles at decreasing step sizes. The top 40 training candidates were compared on 40,000 independent validation groups. Only the validation-selected move and predefined baselines were then scored on the final test split.

Each of the three opponents independently draws a family uniformly from the six Game 1 synthetic families, then a full legal allocation from that family. Draws use replacement to model a large applicant population.

The six hypothetical opponent families are defined in blotto_ai.py and documented in README.md. No real submissions were supplied. Treat this as a model-based recommendation, not a proven universal optimum. There is no equilibrium or worst-case guarantee for this four-player rank-payout game.

Each simulation contains a fixed candidate plus three independently drawn opponents. All allocations use exactly 100 soldiers. Opponents are drawn with replacement as an approximation to a large applicant population. Actual reshuffling samples distinct applicants; their submissions would allow a more faithful simulation.

## Sensitivity: all opponents drawn from one family

- Even-ish: chosen move **9.000/9**; Game 1 move **8.999/9**; castles 4+5 baseline **9.000/9**.
- Value proportional: chosen move **8.999/9**; Game 1 move **8.394/9**; castles 4+5 baseline **8.999/9**.
- High castles focused: chosen move **7.256/9**; Game 1 move **2.355/9**; castles 4+5 baseline **6.129/9**.
- Broad random: chosen move **7.437/9**; Game 1 move **6.563/9**; castles 4+5 baseline **7.727/9**.
- Uniform integer allocations: chosen move **7.355/9**; Game 1 move **6.427/9**; castles 4+5 baseline **7.615/9**.
- Sacrifice castles: chosen move **5.483/9**; Game 1 move **3.075/9**; castles 4+5 baseline **4.688/9**.

These are fixed-move stress tests, not separately optimized strategies for each family. Against broadly random opponents, the castles 4+5 baseline outperforms the selected move; the recommendation depends on the assumed crowd.

## Reproduce

Keep four_way_ai.py and blotto_ai.py in the same folder. Python 3 and NumPy are required.

```sh
python3 four_way_ai.py --train 8000 --test 200000 --seed 20261028
```

This overwrites four_way_results.json. The source includes scoring tests for every finishing position, two/three/four-player ranking ties, four-way castle ties, conservation of all castle points and all finishing payouts, and agreement between the fast training evaluator and direct full-game scoring.

Use four_way_results.json for all parameters, the top validation candidates, test results, and sensitivity results. The Game 1 HTML tool still models Game 1 and should not be used to calculate Game 2 payouts.
