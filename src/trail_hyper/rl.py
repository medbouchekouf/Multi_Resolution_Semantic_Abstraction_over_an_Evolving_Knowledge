"""Small REINFORCE policy over bounded candidate merge actions.

This is intentionally lightweight: replace it with a HyperGNN+PPO policy while
keeping the candidate-action and reward contracts used by the coarsener.
"""
from __future__ import annotations

import numpy as np


class LinearMergePolicy:
    def __init__(self, seed: int = 0, learning_rate: float = 0.02) -> None:
        self.rng = np.random.default_rng(seed)
        self.weights = self.rng.normal(0, 0.05, size=4)
        self.learning_rate = learning_rate
        self.last_features: np.ndarray | None = None
        self.last_probs: np.ndarray | None = None
        self.last_action: int | None = None

    @staticmethod
    def features(candidates: list[tuple[int, int, float, float]]) -> np.ndarray:
        return np.array([[s, m, s * m, abs(s - m)] for _, _, s, m in candidates], dtype=float)

    def __call__(self, candidates: list[tuple[int, int, float, float]]) -> int:
        x = self.features(candidates)
        logits = x @ self.weights
        probs = np.exp(logits - logits.max()); probs /= probs.sum()
        action = int(self.rng.choice(len(candidates), p=probs))
        self.last_features, self.last_probs, self.last_action = x, probs, action
        return action

    def reinforce(self, reward: float) -> None:
        if self.last_features is None or self.last_probs is None or self.last_action is None:
            return
        expected = self.last_probs @ self.last_features
        grad = self.last_features[self.last_action] - expected
        self.weights += self.learning_rate * reward * grad

