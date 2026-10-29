# Process and interpretation

This is the full development record: encode the rules, test synthetic opponents, let those opponents adapt, then benchmark concrete published answers. The work used AI assistance to build and review Python simulations. Every stage changed the opponent assumptions explicitly; a later recommendation is not a correction to the arithmetic of an earlier experiment.

## Encode the rules before searching

Every player allocates exactly 100 nonnegative integer soldiers among castles worth 1 through 5 points. Winning margins do not affect the value of a castle. Equal highest bids share the castle value among the tied leaders.

Game 1 adds castle points and averages them over head-to-head matchups. Game 2 first adds castle points within a group of four, then ranks players and pays 9, 4, 2 and 0. A two-way tie for first receives (9+4)/2 = 6.5 each; a four-way tie receives 15/4 = 3.75 each. Optimizing castle points is therefore insufficient for Game 2.

The four-player scorer represents castle points in twelfths. This makes every possible 2-, 3- and 4-player castle split an exact integer and avoids floating-point errors when deciding tied rankings.

## Establish baselines and opponent assumptions

No real applicant submissions were available. Ordinary opponents were generated from six explicit families: even-ish multinomial allocations, value-proportional multinomial allocations, high-castle-focused Dirichlet shares, broadly random simplex allocations, uniformly sampled integer allocations, and strategies that sacrifice one or two castles.

The models preserve the 100-soldier budget for each whole allocation. Group simulations draw distinct applicant identities; they do not draw castle bids independently and accidentally violate a player's budget.

## Search Game 1

For an opponent's empirical marginal distribution at each castle, compute the expected points earned by every bid from 0 to 100. Dynamic programming allocates the budget among castles to maximize their sum. Because the payoff is additive, this finds the exact empirical best response without explicitly listing every one of the 4,598,126 allocations.

Separate symmetric fictitious play repeatedly best-responds to the history of earlier moves. Its output is a distribution over whole allocations. An exact best response to that final distribution supplies an exploitability check. It does not turn any one sampled move into a guaranteed winner.

## Search Game 2

Each candidate is scored against complete three-opponent groups under the ranking-payout rule. Candidate generation covers random allocations with different concentration levels and all two-castle integer splits. Diverse promising starts are improved by moving troops between castles at decreasing step sizes.

This is a heuristic search, not an exhaustive proof of optimality. Common group draws are reused across candidates to make comparisons less noisy.

## Add stronger opponents

The early recommendation against the ordinary synthetic crowd was 1/1/26/32/40 for Game 1 and 0/0/36/0/64 for Game 2. The latter targets castles 3 and 5; winning both outright gives 8 of the 15 castle points, guaranteeing first place.

Those choices are predictable once copied. The next experiment added strong fixed answers, nearby variations, exact Game 1 counters and learned Game 2 replies. With equal weight on populations containing 0%, 25%, 50% and 75% strategic players, the recommendations changed to 5/17/1/35/42 and 8/0/43/49/0.

These shares were chosen as sensitivity scenarios, not estimated from applicant data. More simulated rounds cannot resolve uncertainty about which scenario describes the real population.

## Continue adaptation and test independently

The adaptive experiment added a counter to copies of each latest answer and eight further rounds of learned replies. It froze this expanded strategic population before using five training seeds to choose candidates. Ten fresh test seeds then evaluated the frozen choices. Each seed/scenario contained 1,000 rounds.

The choices under that equal-scenario objective were 2/3/5/41/49 and 1/41/0/58/0. The test averages were 9.783 castle points for Game 1 and 6.548 finishing points for Game 2. Both improved over the immediately preceding answers in all ten test seeds under the expanded model.

The minimum-scenario shortlist alternatives were 2/15/2/37/44 and 20/0/0/0/80. They favor stability across the modeled scenarios over the highest equal-scenario average. These are finite-shortlist comparisons, not global minimax solutions.

## Test against published answers

### Find sources that actually match the questions

We searched public GitHub repositories and READMEs using terms such as `castle challenge`, `blotto`, `sequoia castle`, and `100 soldiers castles`. Generic Blotto projects were not sufficient: some use ten castles, different values, or a different win condition. We checked each candidate repository for the same five castle values, 100-soldier integer budget, castle-tie rule, head-to-head objective, and four-player 9/4/2/0 finishing payouts.

Four external repositories published concrete picks for both exact games. Their names, published allocations, source file paths, source line numbers and commit hashes are in [public_opponents.json](../outputs/public_opponents.json). The [battle report](public-repository-battle.md#sources-and-fixed-picks) links directly to those immutable versions. We checked the allocations in the pinned files rather than relying on search snippets. Other simulators that matched the rules but did not publish a concrete pick were not included in the fixed-pick benchmark.

Only the allocation facts were extracted. We ran our own scoring and optimization code, not the external repositories' programs, and did not copy their implementations. Attribution is retained for every source. A published choice is an observable benchmark; whether it was actually submitted, copied by other applicants, or subsequently changed is unknown.

### Define the benchmark before measuring it

The literal benchmark gives each of the four external repositories one entrant and equal weight. We use their Game 1 picks for Game 1 and their Game 2 picks for Game 2. Our previous adaptive-model choices remain frozen baselines. We do not compare scores reported by different authors under incompatible opponent models; all picks are rescored under the same rules and field.

- In Game 1, face each of the four published picks exactly once and average the castle points.
- In Game 2, choose three distinct rivals from those four entrants. There are `C(4,3) = 4` possible opponent groups. Evaluate every group exactly; their ordering does not affect our payout.

The old Game 1 answer averaged 7/15 against this field. The old Game 2 answer averaged 0.5/9. Those results exposed how well an answer can fit a synthetic population yet fail against a different set of opponents.

### Solve the Game 1 counter exactly

For castle value `v` and proposed bid `b`, its reward is `v × (fraction of rival bids below b + half the fraction equal to b)`. An exact budget dynamic program finds the allocation maximizing the sum of these five rewards. This solves the same optimization as checking all `C(104,4) = 4,598,126` nonnegative integer allocations of 100 soldiers.

The resulting counter is **4/18/31/45/2**, averaging **11/15**. The numbers have a concrete explanation:

- **18** beats all four published bids on castle 2, whose maximum is 17.
- **31** beats all four bids on castle 3, whose maximum is 30.
- **45** beats all four bids on castle 4, whose maximum is 44.
- Those three castles supply 9 points against every rival.
- **4** additionally wins castle 1 against three of the four rivals. Against the remaining rival, the final **2** soldiers win castle 5 instead.

The four exact matchup scores are **10, 14, 10 and 10**. Thus it beats every one of the four fixed answers, not merely their average. That property is specific to this source list.

### Exhaustively solve the small Game 2 field

The broad synthetic Game 2 experiments needed heuristic search because scoring millions of allocations against thousands of groups is costly. The public benchmark has only four groups, making a complete search practical.

`allocation_batches()` in [public_opponent_battle.py](../outputs/public_opponent_battle.py) enumerates every legal allocation once, in bounded batches. For each castle and bid, a lookup records the resulting scores of all four players against each fixed opponent triple. Adding the five lookups gives full castle totals, after which the ranking scorer applies the exact finishing payouts. The scan keeps the maximum average and counts every allocation attaining it.

All **4,598,126 allocations** were checked. The optimum is **7.75/9**, attained by **495 different allocations**. We report the first lexicographic example, **19/9/0/62/10**, rather than claiming its exact split is uniquely superior.

Its structure follows the published bids:

- **62 on castle 4** beats the largest published bid there, 61.
- **19 on castle 1** beats the largest published bid there, 18.
- **9 on castle 2** takes it when the rival bidding 34 is absent; it also beats the castle-5-focused rival's bid of 8.
- The remaining **10 on castle 5** takes that castle when the rival bidding 70 is absent.

Across the four groups it receives **6.5, 9, 9 and 6.5**: two ties for first and two outright firsts. The mean is exactly `(6.5 + 9 + 9 + 6.5) / 4 = 7.75`. Exhaustive enumeration establishes optimality only for this fixed, equally weighted four-source benchmark.

A separate 1,000-reshuffle check of the frozen answer averaged **7.7325/9**. That small difference is sampling noise. The exact four-group mean remains the benchmark result; the simulated mean was not used to select the answer.

### Check whether the counter is too specialized

We then asked a different question: what if only some applicants copy the public answers? Each sensitivity population contains 1,000 applicant identities. The public-copy shares are 0%, 25%, 50%, 75% and 100%, with the four published picks represented as evenly as possible. The rest come from the earlier ordinary synthetic model.

Unlike the literal benchmark, multiple identities can copy the same source. A Game 2 group still draws three distinct identities, but its three allocations need not come from three different repositories. This distinction matters: the 100%-copy population is not the same game as one entrant per repository.

For each share, a scenario-specific answer is trained on 1,000 opponents or groups and frozen before evaluation. Ten new seeds then provide 1,000 rounds each. All candidates use the same test samples for fair comparisons. Game 1 training is exact; this larger Game 2 search is heuristic.

The targeted counters are not always the strongest choice in these broader fields. With **0% public copies**, the literal Game 2 counter averages about **4.005/9**, versus **7.493/9** for the scenario-trained choice. With **50% copies**, the scenario-trained **0/35/0/62/3** averages **6.910/9**, versus **5.713/9** for the literal-field counter. The [battle report](public-repository-battle.md#what-if-only-some-applicants-copy-the-published-answers) preserves all shares and scores.

We did not measure the prevalence of copying among actual applicants. These are sensitivity tests, not forecasts. Finding a counter to four public answers does not justify assuming the entire tournament consists of those four answers.

## Reproduce the public comparison

From the repository root, with the pinned NumPy dependency installed:

```sh
python3 -u outputs/public_opponent_battle.py
python3 -m unittest discover -s tests -v
```

The script reads the committed source manifest, prints search progress and writes `outputs/public_battle_results.json` plus `docs/public-repository-battle.md`. It does not require GitHub credentials or fetch changing external code at runtime. The manifest identifies the source versions used for the published picks.

The test suite checks the small-budget enumerator for duplicate-free coverage, compares the vectorized batch scorer with direct game scoring, verifies the four exact counter outcomes, and covers the earlier budget, tie, conservation and optimizer checks. The completed suite contains **eight tests**.

## What the results support

The scoring checks passed for budget validity, tied castles, tied finishes, conservation of total castle points and total rank payouts, and agreement between direct and cached scoring. The Game 1 optimizer was also checked against an independently enumerated restricted game.

Independent test data reduce the risk of reporting a move that only looks good on its search samples. Ten-seed comparisons describe variation inside the model. They do not establish how real applicants will behave, an equilibrium for Game 2, or a universally best fixed allocation. Exact enumeration proves the result of a specified finite benchmark; it cannot prove that benchmark represents the actual applicant pool.

The strategy changed for a substantive reason: a move that exploits one population can be exploited after that population adapts. The useful outcome is a reproducible way to examine those assumptions, not an unlimited sequence of increasingly confident fixed answers.
