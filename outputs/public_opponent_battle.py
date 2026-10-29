"""Combat published answers using our scorer, without running third-party code.

Run from repo root: python3 -u outputs/public_opponent_battle.py
Sources and exact commit URLs: public_opponents.json. Game 1 is exact. Game 2
exhaustively scans every legal move against the four distinct-source triples.
An additional model allows copies of those answers among ordinary applicants.
"""
import itertools
import json
import math
from pathlib import Path
import numpy as np
from blotto_ai import best_response, marginals_of, payoffs, validate
from four_way_ai import TrainingEvaluator, check, game_scores, outcome, PREFIX
from live_simulation import make_pool, draw_groups, duel_scores
from predicted_opponents_simulation import quiet_search, stats

ROOT = Path(__file__).resolve().parent
SEED = 20261102
CURRENT = {1:[2,3,5,41,49], 2:[1,41,0,58,0]}


def log(message):
    print(message, flush=True)


def allocation_batches(budget=100, batch_size=50000):
    """Enumerate once each, in lexicographic order; keep memory bounded."""
    buffer = np.empty((batch_size,5),dtype=np.int16)
    used=0
    for a in range(budget+1):
        for b in range(budget-a+1):
            for c in range(budget-a-b+1):
                rest=budget-a-b-c
                size=rest+1
                if used+size>batch_size:
                    yield buffer[:used].copy()
                    used=0
                block=buffer[used:used+size]
                block[:,:3]=[a,b,c]
                block[:,3]=np.arange(size)
                block[:,4]=rest-block[:,3]
                used+=size
    if used:
        yield buffer[:used].copy()


def batch_payouts(allocations, lookup):
    """Return one finishing payout per candidate per fixed opponent triple."""
    scores=lookup[0,allocations[:,0]].copy()
    for i in range(1,5):
        scores+=lookup[i,allocations[:,i]]
    above=(scores[:,:,1:]>scores[:,:,:1]).sum(axis=2)
    ties=(scores[:,:,1:]==scores[:,:,:1]).sum(axis=2)
    return (PREFIX[above+ties+1]-PREFIX[above])/(ties+1)


def exact_four_way(groups):
    lookup=TrainingEvaluator(groups).lookup
    best_value=-1
    best_count=0
    best_examples=[]
    visited=0
    checkpoint=500000
    for batch in allocation_batches():
        means=batch_payouts(batch,lookup).mean(axis=1)
        maximum=float(means.max())
        if maximum>best_value:
            best_value=maximum
            best_count=0
            best_examples=[]
        if maximum==best_value:
            matches=np.flatnonzero(means==best_value)
            best_count+=len(matches)
            best_examples.extend(batch[matches[:max(0,10-len(best_examples))]].tolist())
        visited+=len(batch)
        if visited>=checkpoint:
            log(f'  Exhaustive Game 2: {visited:,}/4,598,126 moves; best {best_value:.4f}/9')
            checkpoint+=500000
    assert visited==math.comb(104,4)
    winner=validate(best_examples[0])
    direct=float(outcome(game_scores(winner,groups))[0].mean())
    assert direct==best_value
    return {'allocation':winner.tolist(),'mean':best_value,'allocations_checked':visited,
            'number_of_tied_optima':best_count,'first_ten_optimal_allocations':best_examples}


def pool_with_public_answers(moves,share,seed):
    rng=np.random.default_rng(seed)
    ordinary=make_pool(seed)
    count=int(1000*share)
    # Public answers are copied exactly; identities remain separate applicants.
    public=np.array(moves)[np.arange(count)%len(moves)]
    rng.shuffle(public)
    pool=np.concatenate((ordinary[:1000-count],public))
    rng.shuffle(pool)
    assert pool.shape==(1000,5) and np.all(pool.sum(axis=1)==100)
    return pool


def test_helpers():
    small=np.concatenate(list(allocation_batches(budget=5,batch_size=20)))
    assert len(small)==math.comb(9,4)
    assert len(set(map(tuple,small)))==len(small) and np.all(small.sum(axis=1)==5)
    rng=np.random.default_rng(8)
    groups=rng.multinomial(100,np.ones(5)/5,size=(7,3))
    lookup=TrainingEvaluator(groups).lookup
    moves=rng.multinomial(100,np.ones(5)/5,size=20)
    fast=batch_payouts(moves,lookup)
    for i,a in enumerate(moves):
        assert np.array_equal(fast[i],outcome(game_scores(a,groups))[0])


def main():
    check()
    test_helpers()
    manifest=json.loads((ROOT/'public_opponents.json').read_text())
    sources=manifest['sources']
    assert len(sources)>=3
    for source in sources:
        for game in [1,2]:
            validate(source[f'game{game}'])
        assert len(source['commit'])==40
    log('PUBLIC REPOSITORY BATTLE — published fixed picks, not inferred submissions')
    for source in sources:
        log(f"  {source['repo']}: G1 {source['game1']}, G2 {source['game2']}")
    rivals1=np.array([s['game1'] for s in sources])
    attack1,value1=best_response(payoffs(marginals_of(rivals1)))
    scores1=duel_scores(CURRENT[1],rivals1)
    log(f'\nGAME 1 current {CURRENT[1]} -> {scores1.tolist()}, mean {scores1.mean():.4f}/15')
    log(f'GAME 1 exact targeted counter {attack1.tolist()} -> {duel_scores(attack1,rivals1).tolist()}, mean {value1:.4f}/15')
    pairwise=[]
    for s in sources:
        counter,best=best_response(payoffs(marginals_of(np.array([s['game1']]))))
        pairwise.append({'source':s['repo'],'current_score':float(duel_scores(CURRENT[1],np.array([s['game1']]))[0]),
            'targeted_counter':counter.tolist(),'targeted_counter_score':best})
    rivals2=np.array([s['game2'] for s in sources])
    # One entrant per repository. Enumerate all C(4,3)=4 distinct-source groups.
    triples=list(itertools.combinations(range(len(sources)),3))
    groups=rivals2[np.array(triples)]
    current2=outcome(game_scores(CURRENT[2],groups))[0]
    log(f'\nGAME 2 current {CURRENT[2]} -> {current2.tolist()}, exact mean {current2.mean():.4f}/9')
    attack2=exact_four_way(groups)
    log(f"GAME 2 exact targeted optimum {attack2['allocation']} -> {attack2['mean']:.4f}/9; {attack2['number_of_tied_optima']} tied optimal moves")
    # 1,000 reshuffles check the exact four-group expectation; selection was exact.
    simulation_groups=draw_groups(rivals2,SEED)
    simulation=stats(attack2['allocation'],simulation_groups,2)
    exact_group_rows=[]
    for index,triple in enumerate(triples):
        score=game_scores(attack2['allocation'],groups[index:index+1])[0]
        payout=float(outcome(score[None,:])[0][0])
        exact_group_rows.append({'opponents':[sources[i]['repo'] for i in triple],
            'counter_castle_scores':(score/12).tolist(),'counter_payout':payout,
            'current_payout':float(current2[index])})
    result={'seed':SEED,'sources':sources,
        'published_picks_are_not_verified_submissions':True,
        'literal_field_assumption':'Equal weight on four repositories, one entrant per source; no jitter.',
        'game1':{'current':CURRENT[1],'current_mean':float(scores1.mean()),'current_scores':scores1.tolist(),
            'targeted_counter':attack1.tolist(),'targeted_counter_mean':value1,
            'targeted_counter_scores':duel_scores(attack1,rivals1).tolist(),'per_source':pairwise},
        'game2':{'current':CURRENT[2],'current_mean':float(current2.mean()),
            'targeted_optimum':attack2,'exact_groups':exact_group_rows,
            'independent_1000_group_simulation':simulation},'mixed_fields':{}}
    # Different question: what if some of a large field copy the published picks?
    for game in [1,2]:
        log(f'\nGAME {game} sensitivity: public-copy shares among otherwise ordinary simulated applicants')
        moves=[s[f'game{game}'] for s in sources]
        tailored=attack1.tolist() if game==1 else attack2['allocation']
        rows=[]
        for i,share in enumerate([0,.25,.5,.75,1.0]):
            pool=pool_with_public_answers(moves,share,SEED+game*1000+i*10)
            if game==1:
                chosen,_=best_response(payoffs(marginals_of(pool)))
                chosen=chosen.tolist()
            else:
                training=draw_groups(pool,SEED+game*1000+i*10+1)
                chosen,_,_=quiet_search(training,SEED+game*1000+i*10+2)
            # Frozen training choice; ten fresh 1,000-round evaluations.
            tests=[]
            for j in range(10):
                p=pool_with_public_answers(moves,share,SEED+100000+game*10000+i*1000+j*10)
                tests.append(draw_groups(p,SEED+100000+game*10000+i*1000+j*10+1) if game==2 else p)
            evaluated={}
            for label,a in [('current',CURRENT[game]),('literal_counter',tailored),('scenario_trained',chosen)]:
                values=[stats(a,t,game)['mean'] for t in tests]
                evaluated[label]={'allocation':list(a),'mean':float(np.mean(values)),'ten_seed_scores':values}
            log(f"  {share:.0%} copies: current {evaluated['current']['mean']:.4f}; targeted {evaluated['literal_counter']['mean']:.4f}; scenario-trained {chosen} {evaluated['scenario_trained']['mean']:.4f}")
            rows.append({'public_copy_share':share,'evaluations':evaluated})
        result['mixed_fields'][str(game)]=rows
    (ROOT/'public_battle_results.json').write_text(json.dumps(result,indent=2))
    write_report(result)
    log('\nSaved public_battle_results.json and docs/public-repository-battle.md')


def write_report(r):
    lines=['# Combat against published repository answers','',
        'Four external repositories publish concrete picks for the exact five-castle, 100-soldier challenge. Their stated scoring matches these questions. We extracted only the published allocation facts and evaluated them with our own scorer; no third-party code was executed or vendored. Sources are pinned to commits in `outputs/public_opponents.json`. Public picks are not evidence that the authors submitted them or that the wider applicant field follows them.','',
        '## Sources and fixed picks','',
        '| Source | Game 1 | Game 2 |','| --- | --- | --- |']
    for s in r['sources']:
        lines.append(f"| [{s['repo']}]({s['source_url']}) | `{s['game1']}` | `{s['game2']}` |")
    g1,g2=r['game1'],r['game2']
    lines+=['','## Game 1: exact direct comparison','',
        f"Our existing answer `{g1['current']}` scores **{g1['current_mean']:.4f}/15** on average against the four published Game 1 picks. The exact empirical best response is **`{g1['targeted_counter']}`**, averaging **{g1['targeted_counter_mean']:.4f}/15**. Dynamic programming covers all 4,598,126 legal moves for this additive objective.",'']
    for s,current,counter in zip(r['sources'],g1['current_scores'],g1['targeted_counter_scores']):
        lines.append(f"- {s['repo']}: current {current:.2f}/15; targeted counter {counter:.2f}/15.")
    a=g2['targeted_optimum']
    lines+=['','## Game 2: exhaustive small-field solution','',
        'Assume one entrant from each of the four repositories, with equal weight. Our entrant faces every possible group of three distinct external entrants. There are four such triples. This is a deliberately small, literal test field; it is not a model of actual tournament prevalence.','',
        f"Our existing answer `{g2['current']}` earns **{g2['current_mean']:.4f}/9** across those four groups. An exhaustive scan of **{a['allocations_checked']:,} legal allocations** finds **`{a['allocation']}`**, earning **{a['mean']:.4f}/9**. There are **{a['number_of_tied_optima']:,}** allocations tied at this exact optimum; the first lexicographic example is shown, not a uniquely superior choice.",'']
    for group in g2['exact_groups']:
        lines.append(f"- Opponents {', '.join(group['opponents'])}: current {group['current_payout']:.2f}/9, targeted {group['counter_payout']:.2f}/9; castle-point totals (ours first) {group['counter_castle_scores']}.")
    sim=g2['independent_1000_group_simulation']
    lines+=['',f"A separate 1,000-group simulation of that frozen optimum averaged {sim['mean']:.4f}/9. The exact four-group result above is authoritative; this Monte Carlo draw is a check, not the selection criterion.",'',
        '## What if only some applicants copy the published answers?','',
        'A separate sensitivity model creates 1,000 applicant identities and assigns 0%, 25%, 50%, 75% or 100% to exact copies of the four public picks in near-equal proportions. The rest use our earlier ordinary synthetic population. Three distinct identities are drawn within a Game 2 group, but different identities can copy the same source. This differs from the one-author-per-entrant literal test above. Copy shares are assumptions, not observed frequencies.','',
        'For each game/share, a scenario-specific answer is trained on 1,000 opponents or groups, frozen, then evaluated on ten fresh seeds of 1,000 rounds each. Game 1 training optimization is exact; Game 2 mixed-field optimization is heuristic. No test result is used to select the scenario-specific answer.','']
    for game,rows in r['mixed_fields'].items():
        lines += [f'### Game {game}','', '| Public-copy share | Existing answer | Literal-field counter | Scenario-trained answer | Scenario-trained score |','| --- | --- | --- | --- | --- |']
        for row in rows:
            e=row['evaluations']
            lines.append(f"| {row['public_copy_share']:.0%} | {e['current']['mean']:.4f} | {e['literal_counter']['mean']:.4f} | `{e['scenario_trained']['allocation']}` | {e['scenario_trained']['mean']:.4f} |")
        lines.append('')
    lines+=['## Reproduce','', '```sh','python3 -u outputs/public_opponent_battle.py','python3 -m unittest discover -s tests -v','```','',
        'The exhaustive Game 2 optimum is proven only for the stated four-source finite field. It is not a universally optimal move. Published repositories are a small, selected sample, and real applicants may change their submissions or use mixed strategies. The additional copy-population simulations show sensitivity, not the actual likelihood of those populations. All earlier reports remain historical results under their original assumptions.']
    (ROOT.parent/'docs/public-repository-battle.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    main()
