"""Stress-test plausible strong opponent answers under both screenshot rules.

Run: python3 -u predicted_opponents_simulation.py
Needs NumPy and the other simulation scripts in this folder. Exactly 1,000
groups per Game 2 candidate; 1,000 opponents per Game 1 scenario. All strategic
shares are hypothetical sensitivity settings, not estimated applicant frequencies.
"""
import contextlib
import io
import json
from pathlib import Path
import numpy as np
from blotto_ai import best_response, marginals_of, payoffs, validate
from four_way_ai import check, game_scores, outcome
from live_simulation import make_pool, draw_groups, duel_scores, search_four_way

SEED = 20261030
SHARES = [0, .25, .5, .75]
PREVIOUS = {1: [1, 1, 26, 32, 40], 2: [0, 0, 36, 0, 64]}


def log(message):
    print(message, flush=True)


def quiet_search(groups, seed):
    with contextlib.redirect_stdout(io.StringIO()):
        return search_four_way(groups, seed)


def strategic_pool(moves, seed):
    rng = np.random.default_rng(seed)
    pool = np.array(moves)[rng.integers(len(moves), size=1000)].copy()
    # Keep 65% exact anchors; nearby legal bids model differing rounding choices.
    for row in pool:
        if rng.random() < .35:
            donor, recipient = rng.choice(5, size=2, replace=False)
            amount = min(int(row[donor]), int(rng.integers(1, 4)))
            row[donor] -= amount
            row[recipient] += amount
    assert pool.shape == (1000, 5) and np.all(pool >= 0)
    assert np.all(pool.sum(axis=1) == 100)
    return pool


def scenario(moves, share, seed):
    base, strong = make_pool(seed), strategic_pool(moves, seed+1)
    rng = np.random.default_rng(seed+2)
    pool = np.concatenate((base[:1000-int(share*1000)], strong[:int(share*1000)]))
    rng.shuffle(pool)
    assert len(pool) == 1000 and np.all(pool.sum(axis=1) == 100)
    return pool


def stats(move, sample, game):
    if game == 1:
        scores = duel_scores(move, sample)
        result = {}
    else:
        scores, above, ties = outcome(game_scores(move, sample))
        result = {'outright_first_rate': float(((above == 0) & (ties == 0)).mean()),
                  'tied_first_rate': float(((above == 0) & (ties > 0)).mean())}
    return {'mean': float(scores.mean()), 'standard_error': float(scores.std(ddof=1)/np.sqrt(1000)), **result}


def main():
    check()
    log('STRONG-OPPONENT RERUN: exact screenshot rules; 1,000 rounds per scenario')
    log('Shares 0%, 25%, 50%, 75% are assumptions. No actual applicant data.')
    log('Independent train/test populations and independent group reshuffles.')
    # Game 1 anchors: familiar strategies plus exact counters to each fixed move.
    g1 = [PREVIOUS[1], [4,11,21,28,36], [7,13,20,27,33]]
    for anchor in list(g1):
        reply, _ = best_response(payoffs(marginals_of(np.array([anchor]))))
        g1.append(reply.tolist())
    for seed in range(4):
        reply, _ = best_response(payoffs(marginals_of(make_pool(SEED+seed))))
        g1.append(reply.tolist())
    # Game 2 anchors reflect high-value concentration, copying and countering it.
    g2 = [PREVIOUS[2], [0,0,35,0,65], [0,0,38,0,62],
          [0,0,0,45,55], [0,0,0,40,60], [1,1,37,1,60], [1,1,0,33,65]]
    # Verify two proposed counters against three exact copies of the old answer.
    clones = np.broadcast_to(np.array(PREVIOUS[2]), (1000,3,5))
    for counter in g2[-2:]:
        assert np.all(outcome(game_scores(counter, clones))[0] == 9)
    # Two reasoning levels learn replies to the earlier strategic population.
    for level in range(2):
        groups = draw_groups(strategic_pool(g2, SEED+100+level), SEED+110+level)
        reply, value, count = quiet_search(groups, SEED+120+level)
        g2.append(reply)
        log(f'Learned Game 2 counter level {level+1}: {reply}; training payout {value:.3f}/9 ({count:,} searched moves)')
    anchors = {1: list(dict.fromkeys(map(tuple, g1))), 2: list(dict.fromkeys(map(tuple, g2)))}
    for game, moves in anchors.items():
        for a in moves:
            validate(a)
        log(f'Game {game} plausible strong anchors: '+str(moves))
    result = {'seed': SEED, 'rounds_per_scenario': 1000,
              'shares': SHARES, 'predicted_answers_are_hypotheses': True,
              'strong_anchor_lists': {str(g): [list(a) for a in moves] for g,moves in anchors.items()},
              'strong_model': 'Uniform draws over listed anchors; 65% unchanged, 35% attempt a random legal transfer of 1–3 soldiers (zero-troop donors can leave a move unchanged).',
              'games': {}}
    for game in [1, 2]:
        log(f'\nGAME {game}: training scenario-specific answers')
        train_samples, test_samples, specialists = [], [], []
        for i, share in enumerate(SHARES):
            train = scenario(anchors[game], share, SEED+1000*game+10*i)
            test = scenario(anchors[game], share, SEED+1000*game+10*i+3)
            if game == 2:
                train = draw_groups(train, SEED+1000*game+10*i+5)
                test = draw_groups(test, SEED+1000*game+10*i+6)
                move, value, count = quiet_search(train, SEED+1000*game+10*i+7)
            else:
                move, value = best_response(payoffs(marginals_of(train)))
                move = move.tolist()
            specialists.append(move)
            train_samples.append(train)
            test_samples.append(test)
            log(f'  {share:.0%} strong opponents: trained answer {move}; objective {value:.4f}')
        candidates = [PREVIOUS[game], *specialists, *anchors[game]]
        if game == 1:
            # Exact optimum for an equally weighted mixture of training scenarios.
            pooled = np.concatenate(train_samples)
            move, _ = best_response(payoffs(marginals_of(pooled)))
            candidates.append(move.tolist())
        else:
            # A fresh 1,000-group search with equal representation of scenarios.
            pooled = np.concatenate([s[:250] for s in train_samples])
            move, _, _ = quiet_search(pooled, SEED+9999)
            candidates.append(move)
        candidates = list(dict.fromkeys(map(tuple, candidates)))
        training = np.array([[stats(a,s,game)['mean'] for s in train_samples] for a in candidates])
        # Freeze the choice before looking at ANY independent test score.
        choice = candidates[int(np.argmax(training.mean(axis=1)))]
        cautious = candidates[int(np.argmax(training.min(axis=1)))]
        log(f'  Frozen equal-scenario recommendation: {choice}')
        log(f'  Best minimum training score among shortlisted moves: {cautious}')
        evaluated = []
        for a in candidates:
            scores = [stats(a,s,game) for s in test_samples]
            evaluated.append({'allocation': list(a), 'scenario_test_scores': scores,
                              'equal_scenario_test_average': float(np.mean([s['mean'] for s in scores]))})
        log('  INDEPENDENT TEST (chosen move / earlier move):')
        chosen = next(row for row in evaluated if row['allocation'] == list(choice))
        old = next(row for row in evaluated if row['allocation'] == PREVIOUS[game])
        for share, new, previous in zip(SHARES, chosen['scenario_test_scores'], old['scenario_test_scores']):
            log(f"    {share:.0%} strong: {new['mean']:.4f} / {previous['mean']:.4f}")
        result['games'][str(game)] = {'previous': PREVIOUS[game], 'scenario_specialists': specialists,
            'selected_on_training': list(choice), 'cautious_shortlist_choice': list(cautious),
            'selection_objective': 'Equal-weight mean over four training scenarios; weights are assumptions.',
            'test_evaluations': evaluated}
    output = Path(__file__).resolve().parent
    (output/'predicted_opponents_results.json').write_text(json.dumps(result, indent=2))
    write_report(result, output)
    log('\nSaved predicted_opponents_results.json and predicted_opponents_report.md')
    log('No guarantee against actual applicants. Game 2 and shortlist robustness searches are heuristic.')


def write_report(result, output):
    lines = ['# Rerun against plausible strong answers', '',
             'All opponent answers and strategic shares are hypotheses, not observed applicant submissions. The screenshot rules are unchanged. Each scenario evaluates 1,000 head-to-head opponents or 1,000 groups of four. Groups select three distinct applicant identities from a fixed population.', '',
             'The four scenarios contain 0%, 25%, 50% and 75% strong opponents. Ordinary opponents use the six previously documented synthetic families. Strong opponents draw uniformly from the listed anchors, usually unchanged, with occasional small legal troop transfers. Two additional Game 2 anchors were learned by responding to the strategic pool.', '',
             'Recommendations maximize the equal-weight average over these four TRAINING scenarios. That weighting is an assumption. Choices are frozen before independent test scores are computed. Different shares can favor different answers.', '']
    for game in [1,2]:
        r = result['games'][str(game)]
        selected = next(x for x in r['test_evaluations'] if x['allocation']==r['selected_on_training'])
        old = next(x for x in r['test_evaluations'] if x['allocation']==r['previous'])
        lines += [f'## Game {game}', '', f"Chosen allocation: **{r['selected_on_training']}**.", '',
                  'Plausible strong opponent answers: '+str(result['strong_anchor_lists'][str(game)]), '',
                  f"Equal-scenario independent test average: **{selected['equal_scenario_test_average']:.4f}**, versus **{old['equal_scenario_test_average']:.4f}** for the previous recommendation. These are expected castle points /15 in Game 1 and expected finishing payouts /9 in Game 2.", '']
        for i, share in enumerate(SHARES):
            new, before = selected['scenario_test_scores'][i], old['scenario_test_scores'][i]
            lines.append(f"- {share:.0%} strong: new **{new['mean']:.4f}**, previous **{before['mean']:.4f}**. New-score Monte Carlo standard error: {new['standard_error']:.4f}.")
        lines += ['', 'Scenario-specific training answers (in increasing strategic-share order): '+str(r['scenario_specialists']), '',
                  'The best minimum-training-score choice among the shortlisted candidates is '+str(r['cautious_shortlist_choice'])+'. This is not a minimax guarantee.', '']
    lines += ['## Limits and reproduction', '',
              'No applicant data, forecast calibration, or equilibrium guarantee. Game 1 best responses use exact budget dynamic programming. Game 2 uses candidate search and local troop transfers, so global optimality is not established. Simulation noise from 1,000 rounds and uncertainty about human choices are separate sources of uncertainty.', '',
              'Run with all supporting scripts in the same folder:', '',
              '```sh', 'python3 -u outputs/predicted_opponents_simulation.py', '```', '',
              'Seed: '+str(SEED)+'. Full anchors, assumptions, scenario specialists, scores and standard errors are in predicted_opponents_results.json.']
    (output/'predicted_opponents_report.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
