"""Continue the analysis: adapt opponents, freeze choices, test ten fresh seeds.

Run: python3 -u outputs/adaptive_simulation.py
Requires NumPy and the earlier Python scripts/results in this folder.
Every per-scenario sample has 1,000 rounds. No claim to predict actual applicants.
"""
import json
from pathlib import Path
import numpy as np
from blotto_ai import best_response, marginals_of, payoffs, validate
from four_way_ai import check, game_scores, outcome
from live_simulation import draw_groups
from predicted_opponents_simulation import scenario, quiet_search, stats

ROOT = Path(__file__).resolve().parent
SEED = 20261101
SHARES = [0, .25, .5, .75]
LATEST = {1: [5,17,1,35,42], 2: [8,0,43,49,0]}
ORIGINAL = {1: [1,1,26,32,40], 2: [0,0,36,0,64]}


def log(message):
    print(message, flush=True)


def samples(anchors, game, seeds, shares=SHARES):
    result = []
    for seed in seeds:
        runs = []
        for index, share in enumerate(shares):
            pool = scenario(anchors, share, seed+index*10)
            runs.append(draw_groups(pool, seed+index*10+5) if game==2 else pool)
        result.append(runs)
    return result


def evaluation(a, runs, game):
    return np.array([[stats(a, sample, game)['mean'] for sample in run] for run in runs])


def paired_summary(new, old):
    # Independent unit is an entire new population/seed, not its individual matches.
    differences = (new-old).mean(axis=1)
    mean = float(differences.mean())
    half_width = 2.262 * float(differences.std(ddof=1)/np.sqrt(len(differences)))
    return {'mean_gain': mean, 'approx_95pct_interval_across_ten_seeds': [mean-half_width, mean+half_width],
            'better_seed_count': int(np.sum(differences>0)), 'seed_count': len(differences)}


def main():
    check()
    previous = json.loads((ROOT/'predicted_opponents_results.json').read_text())
    result = {'seed': SEED, 'adaptive_rounds': 8, 'training_seeds': 5, 'test_seeds': 10,
              'rounds_per_seed_per_scenario': 1000, 'strong_shares': SHARES,
              'opponent_model': previous['strong_model'], 'games': {},
              'limitations': ['All opponent mixes are hypothetical; no applicant submissions.',
                  'Game 2 search and minimum-score shortlist selection are heuristic.',
                  'Cross-seed intervals measure variation inside this assumed model, not forecast accuracy.',
                  'Fixed moves can still be countered; adaptation is not an equilibrium guarantee.']}
    log('ADAPTIVE CONTINUATION: 8 counter rounds, 5 training seeds, 10 untouched test seeds')
    log('1,000 rounds per scenario; exact screenshot scoring and ties. All strategic shares are assumptions.')
    for game in [1,2]:
        log(f'\nGAME {game}')
        anchors = [tuple(a) for a in previous['strong_anchor_lists'][str(game)]]
        anchors.append(tuple(LATEST[game]))
        anchors = list(dict.fromkeys(anchors))
        # First expose the latest answer's vulnerability when three opponents copy it.
        if game == 1:
            clone_counter, clone_score = best_response(payoffs(marginals_of(np.array([LATEST[game]]))))
            counter = clone_counter.tolist()
        else:
            clones = np.broadcast_to(np.array(LATEST[game]), (1000,3,5))
            counter, clone_score, _ = quiet_search(clones, SEED+game)
        anchors.append(tuple(counter))
        log(f'  Counter to copies of latest answer: {counter}, earns {clone_score:.4f}')
        rounds = []
        for level in range(8):
            # 25/50/75/100% strategic mixes teach replies at different depths.
            share = [.25,.5,.75,1.0][level%4]
            pool = scenario(anchors, share, SEED+game*1000+level*10)
            if game==1:
                move, value = best_response(payoffs(marginals_of(pool)))
                move = move.tolist()
            else:
                groups = draw_groups(pool, SEED+game*1000+level*10+1)
                move, value, _ = quiet_search(groups, SEED+game*1000+level*10+2)
            if tuple(move) not in anchors:
                anchors.append(tuple(move))
            rounds.append({'round': level+1, 'strong_share': share, 'reply': move, 'training_score': value})
            log(f'  Counter round {level+1}/8 ({share:.0%} strategic): {move}, score {value:.4f}')
        for a in anchors:
            validate(a)
        # Freeze this opponent model after adaptation; test data are not used here.
        training_seeds = [SEED+10000*game+100*j for j in range(5)]
        train = samples(anchors, game, training_seeds)
        candidates = [tuple(LATEST[game]), tuple(ORIGINAL[game]), *anchors]
        if game==1:
            pooled = np.concatenate([sample for run in train for sample in run])
            move, _ = best_response(payoffs(marginals_of(pooled)))
            candidates.append(tuple(move))
        else:
            # Five independent search starts, each with 1,000 balanced training groups.
            for j in range(5):
                pooled = np.concatenate([run[i][j*50:(j+1)*50] for run in train for i in range(4)])
                assert pooled.shape==(1000,3,5)
                move, value, _ = quiet_search(pooled, SEED+90000+j)
                candidates.append(tuple(move))
                log(f'  Balanced search {j+1}/5: {move}, training score {value:.4f}')
        candidates = list(dict.fromkeys(candidates))
        training_scores = [evaluation(a, train, game) for a in candidates]
        chosen = candidates[int(np.argmax([a.mean() for a in training_scores]))]
        cautious = candidates[int(np.argmax([a.mean(axis=0).min() for a in training_scores]))]
        log(f'  FROZEN recommendation before testing: {chosen}')
        log(f'  Best minimum scenario average among training shortlist: {cautious}')
        test_seeds = [SEED+500000+10000*game+100*j for j in range(10)]
        test = samples(anchors, game, test_seeds)
        evaluated = {}
        for label,a in [('selected',chosen),('latest',LATEST[game]),('original',ORIGINAL[game]),('cautious',cautious)]:
            values = evaluation(a,test,game)
            evaluated[label] = {'allocation': list(a), 'per_seed_scenario_means': values.tolist(),
                'mean': float(values.mean()), 'scenario_means': values.mean(axis=0).tolist(),
                'seed_mean_std': float(values.mean(axis=1).std(ddof=1))}
        for j in range(10):
            log(f"  Untouched test seed {j+1:2}/10: new {np.mean(evaluated['selected']['per_seed_scenario_means'][j]):.4f}, latest {np.mean(evaluated['latest']['per_seed_scenario_means'][j]):.4f}")
        gain_latest = paired_summary(np.array(evaluated['selected']['per_seed_scenario_means']), np.array(evaluated['latest']['per_seed_scenario_means']))
        gain_original = paired_summary(np.array(evaluated['selected']['per_seed_scenario_means']), np.array(evaluated['original']['per_seed_scenario_means']))
        # One additional all-strategic stress test, separate from the main objective.
        stress = samples(anchors,game,[SEED+800000+game*1000+j*100 for j in range(10)],shares=[1.0])
        stress_results = {label:float(evaluation(row['allocation'],stress,game).mean()) for label,row in evaluated.items()}
        result['games'][str(game)] = {'anchors_after_adaptation':[list(a) for a in anchors],
            'counter_to_latest_copies': counter, 'counter_score_against_copies': clone_score,
            'adaptive_history':rounds, 'candidate_count':len(candidates),
            'training_seed_values':training_seeds,'test_seed_values':test_seeds,
            'selection':'Equal-weight mean over four scenarios and five training seeds; frozen before independent testing.',
            'test_results':evaluated,'gain_vs_latest':gain_latest,'gain_vs_original':gain_original,
            'all_strategic_stress_test_means':stress_results}
        log(f"  RESULT new {evaluated['selected']['mean']:.4f}; latest {evaluated['latest']['mean']:.4f}; original {evaluated['original']['mean']:.4f}")
    (ROOT/'adaptive_results.json').write_text(json.dumps(result,indent=2))
    report(result)
    log('\nDone. Saved adaptive_results.json and adaptive_report.md')


def report(result):
    lines=['# Adaptive opponents: continuation', '',
        'Both games use the screenshot rules. This run adds counters to the latest recommended answers, then eight rounds of replies to increasingly strategic simulated populations. It freezes that opponent model, selects answers on five training seeds, and evaluates them on ten independent test seeds.', '',
        'Each seed/scenario contains 1,000 rounds. The main objective equally weights populations containing 0%, 25%, 50% and 75% strong opponents. An additional 100% strategic stress test is reported separately. These proportions and the uniform weights on strategic anchors are hypotheses, not forecasts of real applicants.', '']
    for game in [1,2]:
        g=result['games'][str(game)]
        s=g['test_results']
        lines += [f'## Game {game}', '',f"Frozen recommendation: **{s['selected']['allocation']}**.", '',
            f"Ten-seed test average: **{s['selected']['mean']:.4f}**, compared with latest recommendation {s['latest']['allocation']} at **{s['latest']['mean']:.4f}**, and original {s['original']['allocation']} at **{s['original']['mean']:.4f}**. Units: {'castle points out of 15' if game==1 else 'finishing payout out of 9'}.", '']
        for i,share in enumerate(SHARES):
            lines.append(f"- {share:.0%} strong opponents: new **{s['selected']['scenario_means'][i]:.4f}**; latest **{s['latest']['scenario_means'][i]:.4f}**; original **{s['original']['scenario_means'][i]:.4f}**.")
        gain=g['gain_vs_latest']
        lines += ['',f"Gain over latest recommendation: {gain['mean_gain']:.4f}; approximate 95% paired interval across ten seeds: {gain['approx_95pct_interval_across_ten_seeds']}. Better in {gain['better_seed_count']}/10 seeds. This interval measures seed variation within the assumed model, not uncertainty about human choices.", '',
            f"A counter to exact copies of the latest answer is {g['counter_to_latest_copies']}, with counter score {g['counter_score_against_copies']:.4f}. This assumes all opponents choose that exact move, not that this is what actual applicants will do.", '',
            f"The candidate with the highest minimum training-scenario average was {s['cautious']['allocation']}; its independent equal-scenario test average is {s['cautious']['mean']:.4f}. This is a shortlist comparison, not a minimax guarantee.", '',
            'All-strategic test scores: '+str(g['all_strategic_stress_test_means']), '',
            'Adaptation rounds:', '']
        for r in g['adaptive_history']:
            lines.append(f"- Round {r['round']}, {r['strong_share']:.0%} strong: {r['reply']} (training {r['training_score']:.4f}).")
        lines += ['',f"Final strategic anchors: {g['anchors_after_adaptation']}",'']
    lines += ['## Reproduce and limits','',
        '```sh','python3 -u outputs/adaptive_simulation.py','```','',
        'Keep the prior scripts and predicted_opponents_results.json in outputs/. Requires Python 3 and NumPy. Every seed, anchor, score, scenario, and selection rule is saved in adaptive_results.json.', '',
        'Correct scoring, valid budgets and distinct group identities are checked. The Game 1 training best response is exact for its empirical objective. Game 2 uses heuristic search; global optimality is not established. Fixed answers can still be countered and these experiments do not establish equilibrium. A change in recommendation is conditional on the expanded opponent model; it does not invalidate earlier results under their own assumptions.']
    (ROOT/'adaptive_report.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    main()
