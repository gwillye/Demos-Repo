"""Multi-Touch Attribution — who gets credit for the conversion?

A customer touches several marketing channels before converting (Search ->
Social -> Email -> buy). Which channel deserves the credit? This compares the
classic heuristics (first-touch, last-touch, linear) with a **Markov-chain
removal-effect** model that estimates each channel's *causal* contribution.

Synthetic journeys are generated with a known dominant channel so the result is
verifiable. Run:  python attribution.py
"""
from __future__ import annotations
import numpy as np

CHANNELS = ["Search", "Social", "Display", "Email", "Video"]
# higher weight => more likely to push a journey to conversion (ground truth)
WEIGHT = {"Search": 1.6, "Social": 0.5, "Display": 0.2, "Email": 0.9, "Video": 0.4}
N_JOURNEYS = 6000


def gen_journeys(seed: int = 7):
    rng = np.random.default_rng(seed)
    pick_p = np.array([0.30, 0.24, 0.18, 0.16, 0.12])  # how often each channel appears
    journeys, labels = [], []
    for _ in range(N_JOURNEYS):
        length = rng.integers(1, 6)
        path = list(rng.choice(CHANNELS, size=length, p=pick_p))
        score = -1.2 + sum(WEIGHT[c] for c in dict.fromkeys(path))  # unique channels
        p_conv = 1 / (1 + np.exp(-score))
        converted = rng.random() < p_conv
        journeys.append(path)
        labels.append(bool(converted))
    return journeys, labels


def heuristics(journeys, labels):
    first = dict.fromkeys(CHANNELS, 0.0)
    last = dict.fromkeys(CHANNELS, 0.0)
    linear = dict.fromkeys(CHANNELS, 0.0)
    for path, conv in zip(journeys, labels):
        if not conv:
            continue
        first[path[0]] += 1
        last[path[-1]] += 1
        for c in path:
            linear[c] += 1 / len(path)
    return {"first_touch": _norm(first), "last_touch": _norm(last), "linear": _norm(linear)}


def _norm(d):
    s = sum(d.values())
    return {k: (v / s if s else 0.0) for k, v in d.items()}


def markov_removal(journeys, labels):
    states = ["start"] + CHANNELS + ["conv", "null"]
    idx = {s: i for i, s in enumerate(states)}
    n = len(states)
    counts = np.zeros((n, n))
    for path, conv in zip(journeys, labels):
        prev = "start"
        for c in path:
            counts[idx[prev], idx[c]] += 1
            prev = c
        counts[idx[prev], idx["conv" if conv else "null"]] += 1
    row = counts.sum(axis=1, keepdims=True)
    P = np.divide(counts, row, out=np.zeros_like(counts), where=row != 0)
    P[idx["conv"], idx["conv"]] = 1.0
    P[idx["null"], idx["null"]] = 1.0

    def cvr(matrix):
        # absorption probability into 'conv' from 'start'
        trans = ["start"] + CHANNELS
        t = [idx[s] for s in trans]
        Q = matrix[np.ix_(t, t)]
        r = matrix[np.ix_(t, [idx["conv"]])]
        x = np.linalg.solve(np.eye(len(t)) - Q, r)
        return float(x[0, 0])

    base = cvr(P)
    removal = {}
    for c in CHANNELS:
        Pc = P.copy()
        Pc[idx[c], :] = 0.0
        Pc[idx[c], idx["null"]] = 1.0   # remove channel: its journeys fail
        removal[c] = max(0.0, 1 - cvr(Pc) / base)
    return _norm(removal), base


def main() -> None:
    journeys, labels = gen_journeys()
    cr = sum(labels) / len(labels)
    print(f"{len(journeys)} journeys, conversion rate {cr:.1%}\n")

    models = heuristics(journeys, labels)
    markov, base = markov_removal(journeys, labels)
    models["markov"] = markov

    print(f"{'channel':<9}" + "".join(f"{m:>13}" for m in models))
    for c in CHANNELS:
        print(f"{c:<9}" + "".join(f"{models[m][c]:>12.1%} " for m in models))

    winner = max(markov, key=markov.get)
    print(f"\nMarkov base conversion prob: {base:.3f}")
    print(f"Top channel (Markov causal credit): {winner} ({markov[winner]:.1%})")
    _self_check(models, winner)


def _self_check(models, winner) -> None:
    for name, d in models.items():
        assert abs(sum(d.values()) - 1.0) < 1e-6, f"{name} credits don't sum to 1"
        assert all(v >= 0 for v in d.values()), f"{name} has negative credit"
    # Search has the highest ground-truth weight -> should win the causal (Markov) model
    assert winner == "Search", f"expected Search to win Markov, got {winner}"
    print("self-check: OK (all models normalized, non-negative, Markov recovers the driver)")


if __name__ == "__main__":
    main()
