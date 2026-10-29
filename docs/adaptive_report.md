# Adaptive opponents: continuation

Both games use the screenshot rules. This run adds counters to the latest recommended answers, then eight rounds of replies to increasingly strategic simulated populations. It freezes that opponent model, selects answers on five training seeds, and evaluates them on ten independent test seeds.

Each seed/scenario contains 1,000 rounds. The main objective equally weights populations containing 0%, 25%, 50% and 75% strong opponents. An additional 100% strategic stress test is reported separately. These proportions and the uniform weights on strategic anchors are hypotheses, not forecasts of real applicants.

## Game 1

Frozen recommendation: **[2, 3, 5, 41, 49]**.

Ten-seed test average: **9.7830**, compared with latest recommendation [5, 17, 1, 35, 42] at **8.8970**, and original [1, 1, 26, 32, 40] at **8.6169**. Units: castle points out of 15.

- 0% strong opponents: new **8.9356**; latest **9.1806**; original **9.7500**.
- 25% strong opponents: new **9.4995**; latest **8.9835**; original **8.9958**.
- 50% strong opponents: new **10.0599**; latest **8.7909**; original **8.2483**.
- 75% strong opponents: new **10.6371**; latest **8.6330**; original **7.4737**.

Gain over latest recommendation: 0.8860; approximate 95% paired interval across ten seeds: [0.8543164300111581, 0.9176585699888418]. Better in 10/10 seeds. This interval measures seed variation within the assumed model, not uncertainty about human choices.

A counter to exact copies of the latest answer is [1, 18, 2, 36, 43], with counter score 14.0000. This assumes all opponents choose that exact move, not that this is what actual applicants will do.

The candidate with the highest minimum training-scenario average was [2, 15, 2, 37, 44]; its independent equal-scenario test average is 9.2238. This is a shortlist comparison, not a minimax guarantee.

All-strategic test scores: {'selected': 11.1834, 'latest': 8.4179, 'original': 6.779649999999999, 'cautious': 9.30465}

Adaptation rounds:

- Round 1, 25% strong: [2, 15, 2, 37, 44] (training 9.7655).
- Round 2, 50% strong: [8, 5, 3, 38, 46] (training 10.3655).
- Round 3, 75% strong: [5, 5, 4, 39, 47] (training 11.0020).
- Round 4, 100% strong: [2, 6, 4, 40, 48] (training 11.5320).
- Round 5, 25% strong: [2, 6, 4, 40, 48] (training 9.6565).
- Round 6, 50% strong: [2, 3, 5, 41, 49] (training 10.1915).
- Round 7, 75% strong: [2, 3, 5, 41, 49] (training 10.6390).
- Round 8, 100% strong: [2, 3, 5, 41, 49] (training 11.2405).

Final strategic anchors: [[1, 1, 26, 32, 40], [4, 11, 21, 28, 36], [7, 13, 20, 27, 33], [24, 2, 0, 33, 41], [0, 12, 22, 29, 37], [3, 14, 21, 28, 34], [1, 2, 25, 31, 41], [1, 1, 25, 34, 39], [1, 1, 26, 33, 39], [1, 2, 27, 32, 38], [5, 17, 1, 35, 42], [1, 18, 2, 36, 43], [2, 15, 2, 37, 44], [8, 5, 3, 38, 46], [5, 5, 4, 39, 47], [2, 6, 4, 40, 48], [2, 3, 5, 41, 49]]

## Game 2

Frozen recommendation: **[1, 41, 0, 58, 0]**.

Ten-seed test average: **6.5484**, compared with latest recommendation [8, 0, 43, 49, 0] at **5.1660**, and original [0, 0, 36, 0, 64] at **4.9377**. Units: finishing payout out of 9.

- 0% strong opponents: new **6.2500**; latest **6.3958**; original **7.5817**.
- 25% strong opponents: new **6.5186**; latest **5.5642**; original **5.4985**.
- 50% strong opponents: new **6.5782**; latest **4.6659**; original **3.9311**.
- 75% strong opponents: new **6.8469**; latest **4.0380**; original **2.7396**.

Gain over latest recommendation: 1.3825; approximate 95% paired interval across ten seeds: [1.3069979540069103, 1.4579270459930895]. Better in 10/10 seeds. This interval measures seed variation within the assumed model, not uncertainty about human choices.

A counter to exact copies of the latest answer is [0, 0, 36, 0, 64], with counter score 9.0000. This assumes all opponents choose that exact move, not that this is what actual applicants will do.

The candidate with the highest minimum training-scenario average was [20, 0, 0, 0, 80]; its independent equal-scenario test average is 6.4793. This is a shortlist comparison, not a minimax guarantee.

All-strategic test scores: {'selected': 6.9671, 'latest': 3.3453749999999998, 'original': 1.998825, 'cautious': 6.77555}

Adaptation rounds:

- Round 1, 25% strong: [0, 0, 49, 51, 0] (training 7.5030).
- Round 2, 50% strong: [0, 3, 45, 52, 0] (training 7.3235).
- Round 3, 75% strong: [9, 36, 0, 55, 0] (training 8.0085).
- Round 4, 100% strong: [12, 38, 50, 0, 0] (training 8.6505).
- Round 5, 25% strong: [0, 22, 0, 0, 78] (training 6.8790).
- Round 6, 50% strong: [1, 43, 0, 56, 0] (training 7.1315).
- Round 7, 75% strong: [0, 43, 0, 57, 0] (training 7.4360).
- Round 8, 100% strong: [13, 7, 1, 0, 79] (training 7.7030).

Final strategic anchors: [[0, 0, 36, 0, 64], [0, 0, 35, 0, 65], [0, 0, 38, 0, 62], [0, 0, 0, 45, 55], [0, 0, 0, 40, 60], [1, 1, 37, 1, 60], [1, 1, 0, 33, 65], [7, 18, 2, 0, 73], [6, 6, 8, 3, 77], [8, 0, 43, 49, 0], [0, 0, 36, 0, 64], [0, 0, 49, 51, 0], [0, 3, 45, 52, 0], [9, 36, 0, 55, 0], [12, 38, 50, 0, 0], [0, 22, 0, 0, 78], [1, 43, 0, 56, 0], [0, 43, 0, 57, 0], [13, 7, 1, 0, 79]]

## Reproduce and limits

```sh
python3 -u outputs/adaptive_simulation.py
```

Keep the prior scripts and predicted_opponents_results.json in outputs/. Requires Python 3 and NumPy. Every seed, anchor, score, scenario, and selection rule is saved in adaptive_results.json.

Correct scoring, valid budgets and distinct group identities are checked. The Game 1 training best response is exact for its empirical objective. Game 2 uses heuristic search; global optimality is not established. Fixed answers can still be countered and these experiments do not establish equilibrium. A change in recommendation is conditional on the expanded opponent model; it does not invalidate earlier results under their own assumptions.
