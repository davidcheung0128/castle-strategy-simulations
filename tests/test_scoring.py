"""Small correctness checks for the game rules and optimizer."""
import csv
import json
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'outputs'))
from blotto_ai import best_response, payoffs, marginals_of, self_check, validate
from four_way_ai import check, game_scores, outcome
from live_simulation import draw_groups, duel_scores


class RulesTests(unittest.TestCase):
    def test_two_player_optimizer_and_scoring(self):
        self_check()

    def test_four_player_ties_and_conservation(self):
        check()

    def test_head_to_head_tie_and_known_winner(self):
        self.assertEqual(duel_scores([20]*5, np.array([[20]*5]))[0], 7.5)
        self.assertEqual(duel_scores([4,11,21,28,36], np.array([[20]*5]))[0], 12)

    def test_group_members_are_distinct(self):
        # Every identity has a distinct valid move; drawing all three must include each.
        pool=np.array([[100,0,0,0,0],[0,100,0,0,0],[0,0,100,0,0]])
        groups=draw_groups(pool,42)
        self.assertEqual(groups.shape,(1000,3,5))
        for group in groups:
            self.assertEqual(len(set(map(tuple,group))),3)

    def test_saved_mixed_policy_certificate(self):
        result=json.loads((ROOT/'outputs/results.json').read_text())
        counts=np.zeros((5,101)); total=0
        with (ROOT/'outputs/strategy_mix.csv').open() as f:
            for row in csv.DictReader(f):
                a=validate([int(row[f'castle_{i}']) for i in range(1,6)])
                count=int(row['count']); total+=count
                counts[np.arange(5),a]+=count
        self.assertEqual(total,result['learned_policy']['iterations']+1)
        _,upper=best_response(payoffs(counts/total))
        self.assertAlmostEqual(upper,result['learned_policy']['opponent_best_response_score'])

    def test_saved_adaptive_results_are_consistent(self):
        result=json.loads((ROOT/'outputs/adaptive_results.json').read_text())
        for game in result['games'].values():
            self.assertFalse(set(game['training_seed_values']) & set(game['test_seed_values']))
            for row in game['test_results'].values():
                validate(row['allocation'])
                scores=np.array(row['per_seed_scenario_means'])
                self.assertEqual(scores.shape,(10,4))
                self.assertAlmostEqual(scores.mean(),row['mean'])


if __name__=='__main__':
    unittest.main()
