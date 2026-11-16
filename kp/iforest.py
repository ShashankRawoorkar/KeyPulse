"""Isolation Forest, implemented from scratch in numpy (Liu, Ting & Zhou, 2008).

Idea: anomalies are "few and different", so random axis-aligned splits isolate them
in fewer steps than normal points. Average path length over many random trees -> score.
"""
from __future__ import annotations

import math

import numpy as np

EULER_GAMMA = 0.5772156649


def avg_path_length(n: int) -> float:
    """c(n): average path length of an unsuccessful BST search over n points."""
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    return 2.0 * (math.log(n - 1) + EULER_GAMMA) - 2.0 * (n - 1) / n


class IsolationForest:
    def __init__(self, n_trees: int = 200, sample_size: int = 256, seed: int = 0):
        self.n_trees = n_trees
        self.sample_size = sample_size
        self.seed = seed
        self.trees: list = []
        self.psi = 0

    def fit(self, X: np.ndarray) -> "IsolationForest":
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.seed)
        n = len(X)
        self.psi = min(self.sample_size, n)
        limit = max(1, math.ceil(math.log2(max(self.psi, 2))))
        self.trees = [
            self._build(X[rng.choice(n, self.psi, replace=False)], 0, limit, rng)
            for _ in range(self.n_trees)
        ]
        return self

    def _build(self, X, depth, limit, rng):
        if depth >= limit or len(X) <= 1:
            return ("leaf", len(X))
        spread = X.max(axis=0) - X.min(axis=0)
        candidates = np.flatnonzero(spread > 0)
        if len(candidates) == 0:
            return ("leaf", len(X))
        q = int(rng.choice(candidates))
        p = rng.uniform(X[:, q].min(), X[:, q].max())
        mask = X[:, q] < p
        return ("node", q, p,
                self._build(X[mask], depth + 1, limit, rng),
                self._build(X[~mask], depth + 1, limit, rng))

    def _path(self, x, node, depth):
        while node[0] == "node":
            _, q, p, left, right = node
            node = left if x[q] < p else right
            depth += 1
        return depth + avg_path_length(node[1])

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """s(x) in (0, 1]: ~1 = clearly anomalous, <=0.5 = normal."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        c = avg_path_length(self.psi) or 1.0
        mean_h = np.array([np.mean([self._path(x, t, 0) for t in self.trees]) for x in X])
        return 2.0 ** (-mean_h / c)
