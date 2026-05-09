"""ベースラインのランダムポリシー。"""
from __future__ import annotations

import random

from envs.base import Observation
from .base import Policy


class RandomPolicy(Policy):
    name = "random"

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def act(self, observation: Observation) -> int:
        return self._rng.choice(observation.legal_actions)
