# Process and interpretation

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

The final experiment added a counter to copies of each latest answer and eight further rounds of learned replies. It froze this expanded strategic population before using five training seeds to choose candidates. Ten fresh test seeds then evaluated the frozen choices. Each seed/scenario contained 1,000 rounds.

The final choices under the equal-scenario objective were 2/3/5/41/49 and 1/41/0/58/0. The test averages were 9.783 castle points for Game 1 and 6.548 finishing points for Game 2. Both improved over the immediately preceding answers in all ten test seeds under the expanded model.

The minimum-scenario shortlist alternatives were 2/15/2/37/44 and 20/0/0/0/80. They favor stability across the modeled scenarios over the highest equal-scenario average. These are finite-shortlist comparisons, not global minimax solutions.

## What the results support

The scoring checks passed for budget validity, tied castles, tied finishes, conservation of total castle points and total rank payouts, and agreement between direct and cached scoring. The Game 1 optimizer was also checked against an independently enumerated restricted game.

Independent test data reduce the risk of reporting a move that only looks good on its search samples. Ten-seed comparisons describe variation inside the model. They do not establish how real applicants will behave, an equilibrium for Game 2, or a universally best fixed allocation.

The strategy changed for a substantive reason: a move that exploits one population can be exploited after that population adapts. The useful outcome is a reproducible way to examine those assumptions, not an unlimited sequence of increasingly confident fixed answers.
