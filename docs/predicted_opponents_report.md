# Rerun against plausible strong answers

All opponent answers and strategic shares are hypotheses, not observed applicant submissions. The screenshot rules are unchanged. Each scenario evaluates 1,000 head-to-head opponents or 1,000 groups of four. Groups select three distinct applicant identities from a fixed population.

The four scenarios contain 0%, 25%, 50% and 75% strong opponents. Ordinary opponents use the six previously documented synthetic families. Strong opponents draw uniformly from the listed anchors, usually unchanged, with occasional small legal troop transfers. Two additional Game 2 anchors were learned by responding to the strategic pool.

Recommendations maximize the equal-weight average over these four TRAINING scenarios. That weighting is an assumption. Choices are frozen before independent test scores are computed. Different shares can favor different answers.

## Game 1

Chosen allocation: **[5, 17, 1, 35, 42]**.

Plausible strong opponent answers: [[1, 1, 26, 32, 40], [4, 11, 21, 28, 36], [7, 13, 20, 27, 33], [24, 2, 0, 33, 41], [0, 12, 22, 29, 37], [3, 14, 21, 28, 34], [1, 2, 25, 31, 41], [1, 1, 25, 34, 39], [1, 1, 26, 33, 39], [1, 2, 27, 32, 38]]

Equal-scenario independent test average: **10.2101**, versus **9.5541** for the previous recommendation. These are expected castle points /15 in Game 1 and expected finishing payouts /9 in Game 2.

- 0% strong: new **9.1120**, previous **9.6545**. New-score Monte Carlo standard error: 0.0614.
- 25% strong: new **9.9325**, previous **9.6420**. New-score Monte Carlo standard error: 0.0667.
- 50% strong: new **10.5740**, previous **9.5135**. New-score Monte Carlo standard error: 0.0631.
- 75% strong: new **11.2220**, previous **9.4065**. New-score Monte Carlo standard error: 0.0576.

Scenario-specific training answers (in increasing strategic-share order): [[1, 2, 26, 33, 38], [0, 2, 22, 34, 42], [5, 17, 1, 35, 42], [7, 15, 1, 35, 42]]

The best minimum-training-score choice among the shortlisted candidates is [0, 2, 22, 34, 42]. This is not a minimax guarantee.

## Game 2

Chosen allocation: **[8, 0, 43, 49, 0]**.

Plausible strong opponent answers: [[0, 0, 36, 0, 64], [0, 0, 35, 0, 65], [0, 0, 38, 0, 62], [0, 0, 0, 45, 55], [0, 0, 0, 40, 60], [1, 1, 37, 1, 60], [1, 1, 0, 33, 65], [7, 18, 2, 0, 73], [6, 6, 8, 3, 77]]

Equal-scenario independent test average: **7.3754**, versus **4.8291** for the previous recommendation. These are expected castle points /15 in Game 1 and expected finishing payouts /9 in Game 2.

- 0% strong: new **6.4250**, previous **7.6670**. New-score Monte Carlo standard error: 0.0982.
- 25% strong: new **7.0365**, previous **5.3900**. New-score Monte Carlo standard error: 0.0933.
- 50% strong: new **7.6300**, previous **3.6855**. New-score Monte Carlo standard error: 0.0805.
- 75% strong: new **8.4100**, previous **2.5740**. New-score Monte Carlo standard error: 0.0549.

Scenario-specific training answers (in increasing strategic-share order): [[0, 0, 33, 0, 67], [0, 0, 44, 56, 0], [8, 3, 40, 49, 0], [8, 0, 43, 49, 0]]

The best minimum-training-score choice among the shortlisted candidates is [0, 0, 44, 56, 0]. This is not a minimax guarantee.

## Limits and reproduction

No applicant data, forecast calibration, or equilibrium guarantee. Game 1 best responses use exact budget dynamic programming. Game 2 uses candidate search and local troop transfers, so global optimality is not established. Simulation noise from 1,000 rounds and uncertainty about human choices are separate sources of uncertainty.

Run with all supporting scripts in the same folder:

```sh
python3 -u outputs/predicted_opponents_simulation.py
```

Seed: 20261030. Full anchors, assumptions, scenario specialists, scores and standard errors are in predicted_opponents_results.json.
