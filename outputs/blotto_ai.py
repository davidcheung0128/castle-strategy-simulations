"""100-soldier, five-castle optimizer. Requires Python 3 and NumPy.

Run: python3 blotto_ai.py --iterations 20000 --samples 50000
Outputs results.json and strategy_mix.csv beside this script.
No LLM/API is needed: exact dynamic programming supplies the learning agent.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

VALUES = np.arange(1, 6)
BUDGET = 100


def validate(allocation):
    x = np.asarray(allocation)
    if x.shape != (5,) or not np.all(np.isfinite(x)) or np.any(x < 0) or np.any(x != np.floor(x)) or x.sum() != BUDGET:
        raise ValueError("Use five nonnegative integers summing to 100")
    return x.astype(int)


def payoffs(marginals):
    """Expected castle points for each integer bid, including half-point ties."""
    return VALUES[:, None] * (np.cumsum(marginals, axis=1) - marginals / 2)


SPENT = np.arange(101)[:, None]
BIDS = np.arange(101)[None, :]
REMAINING = np.maximum(SPENT - BIDS, 0)
FEASIBLE = BIDS <= SPENT


def best_response(rewards):
    """Exact maximum over all 4,598,126 feasible allocations, via budget DP."""
    dp = np.full(101, -np.inf)
    dp[0] = 0
    choices = []
    for reward in rewards:
        scores = np.where(FEASIBLE, dp[REMAINING] + reward[None, :], -np.inf)
        choice = np.argmax(scores, axis=1)
        dp = scores[np.arange(101), choice]
        choices.append(choice)
    remaining, allocation = 100, []
    for choice in reversed(choices):
        bid = int(choice[remaining])
        allocation.append(bid)
        remaining -= bid
    return validate(allocation[::-1]), float(dp[100])


def marginals_of(allocations):
    return np.array([np.bincount(allocations[:, i], minlength=101) for i in range(5)]) / len(allocations)


def score(allocation, marginals):
    return float(payoffs(marginals)[np.arange(5), validate(allocation)].sum())


def self_play(iterations, seed):
    """Symmetric fictitious play: best respond to the history, then add the move.

    The empirical mixture is the learned policy, not its mean allocation.
    Each checkpoint's exact best response bounds exploitability over ALL moves.
    """
    rng = np.random.default_rng(seed)
    counts = np.zeros((5, 101))
    history = np.empty((iterations + 1, 5), dtype=int)
    history[0] = rng.multinomial(100, VALUES / 15)
    counts[np.arange(5), history[0]] += 1
    checkpoints = []
    for t in range(1, iterations + 1):
        move, upper = best_response(payoffs(counts / t))
        history[t] = move
        counts[np.arange(5), move] += 1
        if t % 1000 == 0:
            print(f"Self-play {t:,}: opponent best-response score {upper:.5f}/15", flush=True)
            checkpoints.append({"iterations": t, "opponent_best_response": upper})
    distribution = counts / len(history)
    counter, upper = best_response(payoffs(distribution))
    return history, distribution, counter, upper, checkpoints


def rounded_shares(shares):
    raw = shares * 100
    bids = np.floor(raw).astype(int)
    order = np.argsort(-(raw - bids), axis=1)
    for i in range(5):
        rows = np.where(100 - bids.sum(axis=1) > 0)[0]
        bids[rows, order[rows, i]] += 1
    assert np.all(bids.sum(axis=1) == 100)
    return bids


def populations(n, seed):
    """Explicit hypothetical populations; these are NOT observed applicant data."""
    rng = np.random.default_rng(seed)
    result = {}
    result["Even-ish"] = rng.multinomial(100, np.ones(5) / 5, size=n)
    result["Value proportional"] = rng.multinomial(100, VALUES / 15, size=n)
    result["High castles focused"] = rounded_shares(rng.dirichlet([0.3, 0.5, 3, 4, 5], size=n))
    result["Broad random"] = rounded_shares(rng.dirichlet(np.ones(5), size=n))
    # Draw one of the C(104,4) integer allocations uniformly (stars and bars).
    bars = np.sort(np.argpartition(rng.random((n, 104)), 4, axis=1)[:, :4], axis=1)
    result["Uniform integer allocations"] = np.diff(np.column_stack((np.full(n, -1), bars, np.full(n, 104))), axis=1) - 1
    # A human may deliberately give up one or two castles, then split troops.
    shares = rng.dirichlet(np.ones(5) * 2, size=n) * VALUES
    for row in shares:
        row[rng.choice(5, size=int(rng.integers(1, 3)), replace=False)] = 0
    result["Sacrifice castles"] = rounded_shares(shares / shares.sum(axis=1, keepdims=True))
    for population in result.values():
        assert np.all(population >= 0) and np.all(population.sum(axis=1) == 100)
    return result


def self_check():
    x = np.array([4, 11, 21, 28, 36])
    marginal = marginals_of(np.array([x]))
    assert score(x, marginal) == 7.5
    assert score(x, marginals_of(np.array([[20] * 5]))) == 12
    y, upper = best_response(payoffs(marginal))
    assert score(y, marginal) == upper and upper >= 12
    assert score(x, marginals_of(np.array([y]))) + upper == 15
    # Unaffordable reward thresholds must not create a free-budget solution.
    rng = np.random.default_rng(4)
    rewards = rng.random((5, 101))
    rewards[:, 6:] = -1e6
    # Only bids <=5 are allowed, so exactly 100 cannot be spent.
    _, value = best_response(rewards)
    assert value < -1e5
    # Linear rewards: all troops must go to the highest coefficient.
    move, value = best_response(VALUES[:, None] * np.arange(101))
    assert move.tolist() == [0, 0, 0, 0, 100] and value == 500
    # Exhaustive comparison with three active castles (5,151 legal moves).
    rewards = rng.random((5, 101))
    rewards[3:, 1:] = -1e6
    move, value = best_response(rewards)
    brute = max(rewards[0, a] + rewards[1, b] + rewards[2, 100-a-b]
                + rewards[3, 0] + rewards[4, 0]
                for a in range(101) for b in range(101-a))
    assert abs(value - brute) < 1e-10
    for bad in ([0] * 5, [0, 0, 0, -1, 101], [1.5, 1.5, 1, 1, 95]):
        try:
            validate(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid allocation accepted")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--samples", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=20261027)
    args = parser.parse_args()
    if args.iterations < 1 or args.samples < 10:
        parser.error("iterations must be positive; samples must be at least 10")
    self_check()
    history, equilibrium, counter, upper, checkpoints = self_play(args.iterations, args.seed)
    train = {k: marginals_of(v) for k, v in populations(args.samples, args.seed + 1).items()}
    test = {k: marginals_of(v) for k, v in populations(args.samples, args.seed + 2).items()}
    train_mix = np.mean(list(train.values()), axis=0)
    optimized, train_score = best_response(payoffs(train_mix))
    candidates = {"Earlier suggestion": [4, 11, 21, 28, 36], "Even split": [20] * 5,
                  "Proportional": [7, 13, 20, 27, 33], "Optimized synthetic mix": optimized.tolist()}
    specialists = {name: best_response(payoffs(m))[0].tolist() for name, m in train.items()}
    rows = []
    for name, allocation in candidates.items():
        adversary, max_score = best_response(payoffs(marginals_of(np.array([allocation]))))
        rows.append({"name": name, "allocation": allocation,
                     "test_scores": {k: score(allocation, m) for k, m in test.items()},
                     "test_mix_score": float(np.mean([score(allocation, m) for m in test.values()])),
                     "vs_learned_policy": score(allocation, equilibrium),
                     "worst_case_score": 15 - max_score, "counter": adversary.tolist()})
    mix_scores = {name: float((equilibrium * payoffs(m)).sum()) for name, m in test.items()}
    unique, frequencies = np.unique(history, axis=0, return_counts=True)
    output = Path(__file__).resolve().parent
    with (output / "strategy_mix.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["castle_1", "castle_2", "castle_3", "castle_4", "castle_5", "count", "probability"])
        for allocation, count in zip(unique, frequencies):
            writer.writerow([*allocation.tolist(), int(count), float(count / len(history))])
    rng = np.random.default_rng(args.seed + 3)
    sampled = history[rng.integers(len(history))].tolist()
    result = {"seed": args.seed, "samples_per_family_per_split": args.samples,
              "training_and_test_seeds_independent": True, "population_weights": "equal across six hypothetical families",
              "candidates": rows, "specialists": specialists,
              "training_marginals": {k: v.tolist() for k, v in train.items()},
              "test_marginals": {k: v.tolist() for k, v in test.items()},
              "policy_support": unique.tolist(), "policy_counts": frequencies.tolist(),
              "learned_policy": {"iterations": args.iterations, "support_size": len(unique),
                  "opponent_best_response": counter.tolist(), "opponent_best_response_score": upper,
                  "worst_case_expected_score": 15 - upper, "exploitability": upper - 7.5,
                  "self_play_score": 7.5, "mean_allocation": history.mean(axis=0).tolist(),
                  "sampled_allocation": sampled, "test_scores": mix_scores, "checkpoints": checkpoints},
              "limitations": ["Synthetic populations are assumptions, not applicant observations.",
                  "The mixed policy guarantee is in expectation over its random draw, not for each fixed allocation.",
                  "The synthetic optimum is exact for the training empirical distribution; validation uses independent samples.",
                  "A single sampled move can be countered if revealed before opponents choose."]}
    (output / "results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"candidates": rows, "learned_policy": {
        k: v for k, v in result["learned_policy"].items() if k != "checkpoints"}}, indent=2))
    print(f"Saved {output / 'results.json'} and {output / 'strategy_mix.csv'}")


if __name__ == "__main__":
    main()
