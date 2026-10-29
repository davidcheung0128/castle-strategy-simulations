"""Run both screenshot games with exactly 1,000 rounds per evaluated move.

python3 -u live_simulation.py
Optional: --opponents applicants.csv (five integer columns; optional header).
NumPy, blotto_ai.py and four_way_ai.py must be available in this folder.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path
import numpy as np
from blotto_ai import (best_response, marginals_of, payoffs, populations,
                      rounded_shares, validate)
from four_way_ai import TrainingEvaluator, check as check_four_way, game_scores, outcome


def say(message=''):
    print(message, flush=True)


def make_pool(seed):
    """One fixed population, not a new opponent distribution for every move."""
    rng = np.random.default_rng(seed)
    families = populations(1000, seed)
    # Near-equal family representation in a fixed 1,000-applicant population.
    labels = np.arange(1000) % len(families)
    rng.shuffle(labels)
    arrays = np.array(list(families.values()))
    return arrays[labels, rng.integers(1000, size=1000)]


def load_pool(path):
    rows = []
    for line, row in enumerate(csv.reader(Path(path).open()), 1):
        if not row or all(not x.strip() for x in row):
            continue
        try:
            a = [int(x) for x in row]
        except ValueError:
            if line == 1:
                continue
            raise ValueError(f'Invalid integer at CSV line {line}')
        rows.append(validate(a))
    if len(rows) < 3:
        raise ValueError('CSV needs at least three other applicants')
    return np.array(rows)


def draw_groups(pool, seed):
    """Shuffle candidate's group 1,000 times: three DISTINCT pool members."""
    rng = np.random.default_rng(seed)
    indices = np.array([rng.choice(len(pool), size=3, replace=False) for _ in range(1000)])
    assert np.all(indices[:, 0] != indices[:, 1])
    assert np.all(indices[:, 0] != indices[:, 2])
    assert np.all(indices[:, 1] != indices[:, 2])
    return pool[indices]


def duel_scores(a, pool):
    a = validate(a)
    return (((a > pool) + (a == pool) / 2) * np.arange(1, 6)).sum(axis=1)


def show_running(a, opponents, game):
    if game == 1:
        points = duel_scores(a, opponents)
        maximum = 15
    else:
        points = outcome(game_scores(a, opponents))[0]
        maximum = 9
    say(f'  Allocation {list(map(int, a))}; fresh evaluation:')
    for n in range(100, len(points) + 1, 100):
        say(f'    rounds {n:4d}/{len(points)} | running mean {points[:n].mean():.4f}/{maximum}')
    if len(points) % 100:
        say(f'    rounds {len(points)}/{len(points)} | mean {points.mean():.4f}/{maximum}')
    return {'allocation': list(map(int, a)), 'rounds': len(points),
            'mean': float(points.mean()),
            'standard_error': float(points.std(ddof=1)/np.sqrt(len(points)))}


def search_four_way(groups, seed):
    rng = np.random.default_rng(seed)
    evaluator = TrainingEvaluator(groups)
    candidates = [[0, 0, 36, 0, 64], [1, 1, 26, 32, 40], [20] * 5]
    for alpha in [np.ones(5)*.3, np.ones(5), np.arange(1, 6)]:
        candidates.extend(rounded_shares(rng.dirichlet(alpha, size=400)).tolist())
    for i in range(5):
        for j in range(i+1, 5):
            for x in range(101):
                a = [0]*5
                a[i], a[j] = x, 100-x
                candidates.append(a)
    leader, value = None, -1

    def visit(a):
        nonlocal leader, value
        old_count = len(evaluator.cache)
        result = evaluator(a)
        if result > value + 1e-12:
            leader, value = list(map(int, a)), result
            say(f'  NEW LEADER after {len(evaluator.cache):,} moves: {leader} -> {value:.4f}/9')
        if old_count != len(evaluator.cache) and len(evaluator.cache) % 100 == 0:
            say(f'  Search: {len(evaluator.cache):,} moves x 1,000 groups | leader {value:.4f}/9 {leader}')
        return result

    for a in candidates:
        visit(a)
    # Diverse starts prevent the top twenty seeds from being one narrow cluster.
    starts, supports = [], set()
    for a in sorted(evaluator.cache, key=evaluator.cache.get, reverse=True):
        support = tuple(x > 0 for x in a)
        if support not in supports or (len(starts) < 35 and all(sum(abs(x-y) for x,y in zip(a,b)) >= 16 for b in starts)):
            starts.append(a)
            supports.add(support)
    for start in starts[:50]:
        current = np.array(start)
        for step in [8, 4, 2, 1]:
            for _ in range(20):
                best, best_value = current.copy(), visit(current)
                for donor in range(5):
                    if current[donor] < step:
                        continue
                    for recipient in range(5):
                        if donor == recipient:
                            continue
                        a = current.copy()
                        a[donor] -= step
                        a[recipient] += step
                        candidate_value = visit(a)
                        if candidate_value > best_value + 1e-12:
                            best, best_value = a, candidate_value
                if np.array_equal(best, current):
                    break
                current = best
    return leader, value, len(evaluator.cache)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=20261029)
    parser.add_argument('--opponents', type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    say('CASTLE SIMULATION — ACTUAL SCREENSHOT RULES')
    say('Castle values [1,2,3,4,5]; exactly 100 integer soldiers in EVERY allocation.')
    say('No artificial display delays. Progress is printed as computation happens.')
    check_four_way()
    say('PASS: castle ties, ranking ties, point conservation, fast/direct scoring.')
    if args.opponents:
        train = test = load_pool(args.opponents)
        source = f'Provided applicant CSV, {len(train)} other applicants'
        say('ACTUAL DATA: '+source)
        say('Game 1 evaluates EVERY other applicant; count follows CSV, not a 1,000-draw approximation.')
        say('Game 2 draws three distinct other applicants per group, 1,000 reshuffles.')
        say('With a CSV, the same known applicant population is used for train and test; Game 2 reshuffles differ.')
    else:
        train, test = make_pool(args.seed), make_pool(args.seed+1)
        source = 'Hypothetical 1,000-applicant population, near-equal mixture of six documented families'
        say('ASSUMPTION: '+source)
        say('No actual applicant submissions supplied. Fresh independent test population avoids search-set reporting.')
    say('\nGAME 1: HEAD TO HEAD')
    say('Play every other applicant. Higher bid wins castle value; tie splits it. Average points /15.')
    g1, train_value = best_response(payoffs(marginals_of(train)))
    assert abs(duel_scores(g1, train).mean() - train_value) < 1e-10
    say(f'Exact training optimum over all 4,598,126 moves: {g1.tolist()} -> {train_value:.4f}/15')
    game1 = show_running(g1, test, 1)
    game1['previous_recommendation_mean'] = float(duel_scores([1,1,26,32,40], test).mean())
    say('\nGAME 2: FOUR-WAY BATTLE')
    say('Three DISTINCT opponents per reshuffle, 1,000 groups for EVERY searched move.')
    say('Highest castle bid wins; ties share value. Rank by castle points; payouts [9,4,2,0].')
    say('Tied finishes split the payouts of occupied places. Maximize average finishing PAYOUT /9.')
    groups = draw_groups(train, args.seed+2)
    move, training_value, searched = search_four_way(groups, args.seed+3)
    say(f'Best FOUND move after {searched:,} distinct moves: {move} -> {training_value:.4f}/9 training')
    say('Frozen choice now evaluated on 1,000 NEW groups (not used to choose the move).')
    test_groups = draw_groups(test, args.seed+4)
    game2 = show_running(move, test_groups, 2)
    payout, above, equal = outcome(game_scores(move, test_groups))
    game2['outright_first_rate'] = float(((above == 0) & (equal == 0)).mean())
    game2['tied_first_rate'] = float(((above == 0) & (equal > 0)).mean())
    game2['previous_recommendation_mean'] = float(outcome(game_scores([0,0,36,0,64], test_groups))[0].mean())
    game2['game1_allocation_mean'] = float(outcome(game_scores(g1, test_groups))[0].mean())
    result = {'seed': args.seed, 'opponent_source': source, 'exact_rules': True,
              'game1': game1, 'game2': game2, 'game2_candidates': searched,
              'game1_training_mean': train_value, 'game2_training_mean': training_value,
              'limits': ['Actual scores depend on actual applicant submissions.',
                         'Game 1 optimum is exact for the training population only.',
                         'Game 2 uses heuristic search; no global-optimality or equilibrium claim.',
                         '1,000 rounds have sampling noise; neighboring allocations may be statistically indistinguishable.']}
    output = Path(__file__).resolve().parent / 'live_1000_results.json'
    output.write_text(json.dumps(result, indent=2))
    say('\nCOMPLETE — INDEPENDENT EVALUATION (castle order 1 -> 5)')
    say(f"  GAME 1: {game1['allocation']} | {game1['mean']:.4f}/15 | {game1['rounds']} matchups")
    say(f"  GAME 2: {game2['allocation']} | {game2['mean']:.4f}/9 | {game2['rounds']} groups")
    say(f"  GAME 2 outright first: {100*game2['outright_first_rate']:.1f}%; tied first: {100*game2['tied_first_rate']:.1f}%")
    say('  These are model-based answers, not guaranteed best against unknown applicants.')
    say(f'  Elapsed: {time.perf_counter()-started:.2f}s. Results: {output}')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        sys.exit(1)
