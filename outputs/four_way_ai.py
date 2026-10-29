"""Game 2: search for expected FINISHING payout, not expected castle points.

Run: python3 four_way_ai.py --train 8000 --test 200000
Requires NumPy and blotto_ai.py in the same folder. No external AI/API needed.
Search is heuristic; results are not a proven global optimum or equilibrium.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from blotto_ai import populations, rounded_shares, validate

VALUES = np.arange(1, 6)
PREFIX = np.array([0, 9, 13, 15, 15], dtype=float)


def outcome(scores):
    """Castle scores in twelfths make all 2/3/4-player ties exact integers."""
    above = (scores[:, 1:] > scores[:, :1]).sum(axis=1)
    equal = (scores[:, 1:] == scores[:, :1]).sum(axis=1)
    payout = (PREFIX[above + equal + 1] - PREFIX[above]) / (equal + 1)
    return payout, above, equal


def game_scores(allocation, groups):
    bids = np.concatenate((np.broadcast_to(validate(allocation), (len(groups), 1, 5)), groups), axis=1)
    winners = bids == bids.max(axis=1, keepdims=True)
    return (winners * (12 * VALUES / winners.sum(axis=1))[:, None, :]).sum(axis=2).astype(int)


def evaluate(allocation, groups):
    scores = game_scores(allocation, groups)
    payout, above, equal = outcome(scores)
    return {"allocation": list(map(int, allocation)), "mean_payout": float(payout.mean()),
            "standard_error": float(payout.std(ddof=1) / np.sqrt(len(payout))),
            "outright_first_probability": float(((above == 0) & (equal == 0)).mean()),
            "shared_first_probability": float(((above == 0) & (equal > 0)).mean()),
            "mean_castle_points": float(scores[:, 0].mean() / 12)}


def groups_from_population(n, seed):
    rng = np.random.default_rng(seed)
    families = populations(max(1000, n // 2), seed + 1)
    arrays = np.array(list(families.values()))
    choice = rng.integers(len(arrays), size=(n, 3))
    index = rng.integers(arrays.shape[1], size=(n, 3))
    return arrays[choice, index]


class TrainingEvaluator:
    """Cache exact per-castle group scores for every possible bid."""
    def __init__(self, groups):
        n = len(groups)
        self.lookup = np.empty((5, 101, n, 4), dtype=np.int16)
        for castle in range(5):
            opponents = groups[:, :, castle]
            maximum = opponents.max(axis=1)
            winners = opponents == maximum[:, None]
            count = winners.sum(axis=1)
            for bid in range(101):
                ties = count + (bid == maximum)
                share = 12 * (castle + 1) // ties
                self.lookup[castle, bid, :, 0] = np.where(bid < maximum, 0, np.where(bid > maximum, 12 * (castle + 1), share))
                self.lookup[castle, bid, :, 1:] = winners * (bid <= maximum)[:, None] * share[:, None]
        self.cache = {}

    def __call__(self, allocation):
        key = tuple(map(int, allocation))
        if key not in self.cache:
            scores = sum(self.lookup[i, bid] for i, bid in enumerate(key))
            self.cache[key] = float(outcome(scores)[0].mean())
        return self.cache[key]


def search(groups, seed):
    rng = np.random.default_rng(seed)
    objective = TrainingEvaluator(groups)
    # Cover random full-budget moves, concentrated moves, and all two-castle splits.
    seeds = [[1, 1, 26, 32, 40], [4, 11, 21, 28, 36], [20] * 5, [7, 13, 20, 27, 33]]
    for alpha in [np.ones(5) * 0.3, np.ones(5), VALUES, VALUES * 3]:
        seeds.extend(rounded_shares(rng.dirichlet(alpha, size=1200)).tolist())
    for i in range(5):
        for j in range(i + 1, 5):
            for amount in range(0, 101):
                a = [0] * 5
                a[i], a[j] = amount, 100 - amount
                seeds.append(a)
    for a in seeds:
        objective(a)
    # Multiple starts and coarse-to-fine troop transfers reduce local traps.
    ranked = sorted(objective.cache, key=objective.cache.get, reverse=True)
    starts, supports = [], set()
    for a in ranked:
        support = tuple(x > 0 for x in a)
        if support not in supports or (len(starts) < 50 and all(sum(abs(x-y) for x,y in zip(a,b)) >= 16 for b in starts)):
            starts.append(a)
            supports.add(support)
    starts = starts[:65]
    for start in starts:
        current = np.array(start)
        for step in [12, 6, 3, 1]:
            for _ in range(30):
                best, best_value = current.copy(), objective(current)
                for donor in range(5):
                    if current[donor] < step:
                        continue
                    for recipient in range(5):
                        if donor == recipient:
                            continue
                        proposal = current.copy()
                        proposal[donor] -= step
                        proposal[recipient] += step
                        value = objective(proposal)
                        if value > best_value + 1e-12:
                            best, best_value = proposal, value
                if np.array_equal(current, best):
                    break
                current = best
    top = sorted(objective.cache, key=objective.cache.get, reverse=True)[:40]
    print(f"Searched {len(objective.cache):,} distinct moves against {len(groups):,} training groups", flush=True)
    return top, objective


def check():
    equal = np.full((1, 3, 5), 20)
    scores = game_scores([20] * 5, equal)
    assert np.all(scores == 45)
    assert outcome(scores)[0][0] == 3.75
    # Rank payouts, including ties across two, three and four places.
    cases = np.array([[8, 7, 6, 5], [7, 8, 6, 5], [6, 8, 7, 5], [5, 8, 7, 6],
                      [8, 8, 6, 5], [7, 8, 7, 5], [6, 8, 7, 6], [8, 8, 8, 5], [6, 6, 6, 6]])
    assert np.allclose(outcome(cases)[0], [9, 4, 2, 0, 6.5, 3, 1, 5, 3.75])
    rng = np.random.default_rng(12)
    groups = rng.multinomial(100, np.ones(5) / 5, size=(100, 3))
    lookup = TrainingEvaluator(groups)
    for a in rng.multinomial(100, np.ones(5) / 5, size=20):
        s = game_scores(a, groups)
        assert np.all(s.sum(axis=1) == 180)
        assert abs(lookup(a) - outcome(s)[0].mean()) < 1e-12
        # All four players' finish payouts must total 15 in every group.
        payouts = sum(outcome(s[:, [i] + [j for j in range(4) if j != i]])[0] for i in range(4))
        assert np.allclose(payouts, 15)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train', type=int, default=8000)
    p.add_argument('--test', type=int, default=200000)
    p.add_argument('--seed', type=int, default=20261028)
    args = p.parse_args()
    if args.train < 100 or args.test < 100:
        p.error('train and test must each be at least 100')
    check()
    train = groups_from_population(args.train, args.seed)
    candidates, objective = search(train, args.seed + 10)
    validation = groups_from_population(40000, args.seed + 20)
    ranked = sorted((evaluate(a, validation) for a in candidates), key=lambda r: r['mean_payout'], reverse=True)
    selected = ranked[0]['allocation']
    print('Validation-selected move:', selected, ranked[0]['mean_payout'], flush=True)
    # The final test split is not used to select a winner.
    test = groups_from_population(args.test, args.seed + 30)
    baselines = {'Selected Game 2 move': selected, 'Game 1 recommendation': [1, 1, 26, 32, 40],
                 'Earlier suggestion': [4, 11, 21, 28, 36], 'Even split': [20] * 5,
                 'Proportional': [7, 13, 20, 27, 33], 'Two high castles': [0, 0, 0, 45, 55]}
    rows = [{'name': name, **evaluate(a, test)} for name, a in baselines.items()]
    selected_payoffs = outcome(game_scores(selected, test))[0]
    baseline_payoffs = outcome(game_scores(baselines['Game 1 recommendation'], test))[0]
    difference = selected_payoffs - baseline_payoffs
    sensitivity = {}
    families = populations(50000, args.seed + 40)
    rng = np.random.default_rng(args.seed + 41)
    for name, pop in families.items():
        groups = pop[rng.integers(len(pop), size=(30000, 3))]
        sensitivity[name] = {label: evaluate(a, groups)['mean_payout'] for label, a in baselines.items()}
    result = {'game': 2, 'seed': args.seed, 'training_groups': args.train, 'validation_groups': len(validation),
              'final_test_groups': args.test, 'searched_allocations': len(objective.cache),
              'opponent_model': 'Each of the three opponents independently draws a family uniformly from the six Game 1 synthetic families, then a full legal allocation from that family. Draws use replacement to model a large applicant population.',
              'selected_allocation': selected, 'final_test_results': rows,
              'gain_vs_game1': {'mean': float(difference.mean()), 'paired_standard_error': float(difference.std(ddof=1)/np.sqrt(len(difference)))},
              'validation_top_candidates': ranked[:10], 'homogeneous_family_sensitivity': sensitivity,
              'limitations': ['No real applicant data.', 'Heuristic search, not an exhaustive global optimum.',
                             'No equilibrium or worst-case guarantee for this four-player rank-payout game.',
                             'Monte Carlo standard errors measure simulation noise, not uncertainty about human behavior.']}
    folder = Path(__file__).resolve().parent
    (folder / 'four_way_results.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
