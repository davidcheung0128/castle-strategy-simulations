# Castle allocation AI — completed simulation

For the assumed applicant population, the strongest fixed allocation found was **1, 1, 26, 32, 40**, in castle order 1–5. It scored **9.775 / 15** on independent test opponents, compared with **9.160 / 15** for the earlier suggestion of 4, 11, 21, 28, 36. That is an improvement of about **0.615 points**, or **6.7%**, under this particular opponent model.

This is an exact optimum against the empirical training mixture, not a universal best move. Nearby allocations can perform similarly; the exact integers depend on the population assumptions. No real applicant submissions were available.

## Use the simulator

Open **castle-allocation-lab.html** in a browser. It is a standalone offline tool; no server, login, API key, or external service is needed.

- Adjust how common you think each opponent family will be, then select **Optimize allocation**.
- Inspect any allocation's expected scores and a best possible counter.
- Draw an entire allocation from the learned randomized policy.
- Compare against the earlier suggestion, proportional allocation, and equal split.

The browser calculates exact best responses with dynamic programming. Its default results and learned policy come from the completed Python run. Browser draws are fresh random selections from that policy.

## What the AI did

This is an algorithmic learning agent, not an LLM. It needs no language-model service.

1. Ran **50,000 symmetric fictitious-play iterations**: at each step, find an exact best response to the empirical history, then add that move to the history.
2. Learned a mixed strategy over **2,201 distinct legal allocations**, with frequencies stored in strategy_mix.csv. The initial move plus 50,000 updates give 50,001 history entries.
3. Checked its final policy against the best possible opponent allocation, considering **all 4,598,126** legal allocations through an equivalent dynamic program.
4. Generated **100,000 opponents per family per split**: six hypothetical families, with separate training and test seeds, totaling **1.2 million opponent draws**.
5. Found the exact fixed best response to an equally weighted mixture of the six empirical training populations; evaluated it on the independent test split.

Seed: **20261027**. Training population seed: 20261028. Test population seed: 20261029. Randomized policy initialization and tie choices are deterministic for reproduction; the policy must still be sampled privately when actually used.

## Protection against an adaptive opponent

The learned randomized policy's **worst-case expected score is 7.443441 / 15**. The exact best response to that policy earns 7.556559 / 15. Since the game is symmetric and constant-sum, its minimax value is 7.5 points, placing this policy within **0.056559 points** of that benchmark.

This bound is computed from the full final policy, not estimated from sampled opponents. It applies in expectation over a private random choice from the full policy. It does not promise that every sampled allocation earns 7.443441 points. Do not turn the average allocation into one fixed recommendation and expect the same guarantee.

The earlier fixed suggestion can be held to **1 point** by [0, 12, 22, 29, 37]. The newly optimized fixed move can be held to **3 points** by a best response. These counters assume the opponent knows your fixed move; they are vulnerability checks, not predictions of applicant choices.

## Opponent assumptions

Each family gets one-sixth weight by default. These are explicit hypothetical models, not observed human frequencies.

- **Even-ish:** multinomial troop counts with equal castle probabilities.
- **Value proportional:** multinomial counts with probabilities [1, 2, 3, 4, 5] / 15.
- **High castles focused:** Dirichlet shares with parameters [0.3, 0.5, 3, 4, 5], rounded to integer troops.
- **Broad random:** Dirichlet shares with parameters [1, 1, 1, 1, 1], rounded to integer troops.
- **Uniform integer allocations:** sample each legal integer allocation with equal probability, using stars and bars. This differs slightly from rounding uniform continuous shares.
- **Sacrifice castles:** randomly abandon one or two castles, then divide troops using Dirichlet(2, 2, 2, 2, 2) shares multiplied by castle values and renormalized.

All moves use exactly 100 soldiers. A tie awards half the castle's value to each player. All reported scores are expected points out of 15, not probabilities of winning a matchup.

## Run it again

Requirements: Python 3 and NumPy. NumPy was already available in the execution environment; no packages were installed.

```sh
python3 blotto_ai.py --iterations 50000 --samples 100000 --seed 20261027
```

The command runs built-in correctness checks, trains the agent, and overwrites results.json and strategy_mix.csv in the script's folder. It does not rewrite the HTML. Use **Load another results.json** in the HTML tool to inspect a new run.

- **blotto_ai.py:** complete runnable training and simulation source.
- **results.json:** parameters, independent evaluation results, checkpoints, empirical train/test marginals, and the full learned policy.
- **strategy_mix.csv:** whole allocations and their sampling probabilities. Sample a row in proportion to its probability, not uniformly across distinct rows.
- **castle-allocation-lab.html:** interactive offline optimizer and policy sampler, with the completed run embedded.

## Validation and limits

Python self-checks cover tied matches, complementary scores, allocation validation, budget enforcement, a known linear optimum, and an exhaustive 5,151-allocation comparison of the dynamic program on a restricted game. Browser JavaScript was executed in Node with a minimal DOM stub to check initialization, scoring, optimization, policy sampling, invalid allocations, zero weights, and reset. The independently implemented Python and JavaScript optimizers produced the same recommended allocation and rounded test score.

Browser visual inspection was blocked because the browser tool disallows local file URLs. The rendered layout was not visually verified. The mathematical checks and Node interaction checks passed.

For a one-shot submission against the assumed crowd, use the fixed recommendation. For protection against strategic opponents, use a private draw from the learned policy. Real applicant allocations would let the optimizer target the actual population rather than these assumptions.
