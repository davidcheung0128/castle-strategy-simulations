# Combat against published repository answers

Four external repositories publish concrete picks for the exact five-castle, 100-soldier challenge. Their stated scoring matches these questions. We extracted only the published allocation facts and evaluated them with our own scorer; no third-party code was executed or vendored. Sources are pinned to commits in `outputs/public_opponents.json`. Public picks are not evidence that the authors submitted them or that the wider applicant field follows them.

## Sources and fixed picks

| Source | Game 1 | Game 2 |
| --- | --- | --- |
| [nischalon10/poker-bot-tournament](https://github.com/nischalon10/poker-bot-tournament/blob/a24dbbed99cb09289538563c26133e07433e8c99/README.md) | `[1, 0, 21, 31, 47]` | `[0, 0, 48, 52, 0]` |
| [rahulr-1006/castle_game](https://github.com/rahulr-1006/castle_game/blob/49a7353496f83244149c51f243cb4bfd297578ec/README.md) | `[8, 17, 30, 44, 1]` | `[18, 8, 2, 2, 70]` |
| [qbeka/No-Limit-a-bot-building-tournament-simulations](https://github.com/qbeka/No-Limit-a-bot-building-tournament-simulations/blob/665a58f6a6b087ad4de199a4d1ac37ad4fe806d8/README.md) | `[3, 5, 27, 2, 63]` | `[2, 34, 3, 59, 2]` |
| [PartyD1/castle-game](https://github.com/PartyD1/castle-game/blob/f6ce78d2ae5ddcdbd0e7d9712ba05ceb8eb9435e/outputs/REPORT.md) | `[2, 3, 5, 37, 53]` | `[0, 0, 39, 61, 0]` |

## Game 1: exact direct comparison

Our existing answer `[2, 3, 5, 41, 49]` scores **7.0000/15** on average against the four published Game 1 picks. The exact empirical best response is **`[4, 18, 31, 45, 2]`**, averaging **11.0000/15**. Dynamic programming covers all 4,598,126 legal moves for this additive objective.

- nischalon10/poker-bot-tournament: current 12.00/15; targeted counter 10.00/15.
- rahulr-1006/castle_game: current 5.00/15; targeted counter 14.00/15.
- qbeka/No-Limit-a-bot-building-tournament-simulations: current 4.00/15; targeted counter 10.00/15.
- PartyD1/castle-game: current 7.00/15; targeted counter 10.00/15.

## Game 2: exhaustive small-field solution

Assume one entrant from each of the four repositories, with equal weight. Our entrant faces every possible group of three distinct external entrants. There are four such triples. This is a deliberately small, literal test field; it is not a model of actual tournament prevalence.

Our existing answer `[1, 41, 0, 58, 0]` earns **0.5000/9** across those four groups. An exhaustive scan of **4,598,126 legal allocations** finds **`[19, 9, 0, 62, 10]`**, earning **7.7500/9**. There are **495** allocations tied at this exact optimum; the first lexicographic example is shown, not a uniquely superior choice.

- Opponents nischalon10/poker-bot-tournament, rahulr-1006/castle_game, qbeka/No-Limit-a-bot-building-tournament-simulations: current 0.00/9, targeted 6.50/9; castle-point totals (ours first) [5.0, 3.0, 5.0, 2.0].
- Opponents nischalon10/poker-bot-tournament, rahulr-1006/castle_game, PartyD1/castle-game: current 0.00/9, targeted 9.00/9; castle-point totals (ours first) [7.0, 3.0, 5.0, 0.0].
- Opponents nischalon10/poker-bot-tournament, qbeka/No-Limit-a-bot-building-tournament-simulations, PartyD1/castle-game: current 0.00/9, targeted 9.00/9; castle-point totals (ours first) [10.0, 3.0, 2.0, 0.0].
- Opponents rahulr-1006/castle_game, qbeka/No-Limit-a-bot-building-tournament-simulations, PartyD1/castle-game: current 2.00/9, targeted 6.50/9; castle-point totals (ours first) [5.0, 5.0, 2.0, 3.0].

A separate 1,000-group simulation of that frozen optimum averaged 7.7325/9. The exact four-group result above is authoritative; this Monte Carlo draw is a check, not the selection criterion.

## What if only some applicants copy the published answers?

A separate sensitivity model creates 1,000 applicant identities and assigns 0%, 25%, 50%, 75% or 100% to exact copies of the four public picks in near-equal proportions. The rest use our earlier ordinary synthetic population. Three distinct identities are drawn within a Game 2 group, but different identities can copy the same source. This differs from the one-author-per-entrant literal test above. Copy shares are assumptions, not observed frequencies.

For each game/share, a scenario-specific answer is trained on 1,000 opponents or groups, frozen, then evaluated on ten fresh seeds of 1,000 rounds each. Game 1 training optimization is exact; Game 2 mixed-field optimization is heuristic. No test result is used to select the scenario-specific answer.

### Game 1

| Public-copy share | Existing answer | Literal-field counter | Scenario-trained answer | Scenario-trained score |
| --- | --- | --- | --- | --- |
| 0% | 8.9341 | 8.1102 | `[1, 2, 25, 31, 41]` | 9.7494 |
| 25% | 8.4574 | 8.8297 | `[3, 6, 31, 38, 22]` | 8.9093 |
| 50% | 7.9868 | 9.5575 | `[4, 18, 31, 45, 2]` | 9.5575 |
| 75% | 7.4724 | 10.2733 | `[4, 18, 31, 45, 2]` | 10.2733 |
| 100% | 7.0000 | 11.0000 | `[4, 18, 31, 45, 2]` | 11.0000 |

### Game 2

| Public-copy share | Existing answer | Literal-field counter | Scenario-trained answer | Scenario-trained score |
| --- | --- | --- | --- | --- |
| 0% | 6.3704 | 4.0054 | `[0, 0, 34, 0, 66]` | 7.4929 |
| 25% | 4.7978 | 4.7754 | `[0, 29, 0, 0, 71]` | 6.7134 |
| 50% | 3.5091 | 5.7128 | `[0, 35, 0, 62, 3]` | 6.9096 |
| 75% | 2.6082 | 6.8266 | `[0, 35, 0, 63, 2]` | 7.2385 |
| 100% | 2.0192 | 8.2868 | `[19, 11, 6, 62, 2]` | 8.5270 |

## Reproduce

```sh
python3 -u outputs/public_opponent_battle.py
python3 -m unittest discover -s tests -v
```

The exhaustive Game 2 optimum is proven only for the stated four-source finite field. It is not a universally optimal move. Published repositories are a small, selected sample, and real applicants may change their submissions or use mixed strategies. The additional copy-population simulations show sensitivity, not the actual likelihood of those populations. All earlier reports remain historical results under their original assumptions.
