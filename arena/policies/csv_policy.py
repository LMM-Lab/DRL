"""CSV(Q-table)で学習したテーブル系モデルのポリシー。

CSV フォーマット (long形式):
    state_key,action,value
    XOX..O...,4,0.83
    XOX..O...,7,-0.12
    ...

合法手のうち最大の `value` を持つ行動を選ぶ。state_key 未学習なら legal の中から
ランダム(deterministic 用に seed 可)。
"""
from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path

from envs.base import Observation
from .base import Policy


class CsvPolicy(Policy):
    def __init__(self, path: str | Path, name: str = "csv", seed: int | None = None) -> None:
        self.name = name
        self._table: dict[str, dict[int, float]] = defaultdict(dict)
        self._rng = random.Random(seed)
        self._load(Path(path))

    def _load(self, path: Path) -> None:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            required = {"state_key", "action", "value"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(
                    f"csv must have header {sorted(required)}, got {reader.fieldnames}"
                )
            for row in reader:
                key = row["state_key"]
                action = int(row["action"])
                value = float(row["value"])
                self._table[key][action] = value

    def act(self, observation: Observation) -> int:
        legal = observation.legal_actions
        entries = self._table.get(observation.state_key)
        if not entries:
            return self._rng.choice(legal)
        best_action = -1
        best_value = float("-inf")
        for a in legal:
            v = entries.get(a)
            if v is None:
                continue
            if v > best_value:
                best_value = v
                best_action = a
        if best_action < 0:
            return self._rng.choice(legal)
        return best_action
