"""PMML(Predictive Model Markup Language)で保存された分類器を使うポリシー。

`pypmml` がインストールされている場合のみ使用可能。フラット化した観測(`obs.array.flatten()`)
を入力に取り、`probability(<action>)` または `predicted_<target>` を出力する分類器を期待する。

`feature_names` は meta.json で指定可能。未指定なら `f0, f1, ...` を使う。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from envs.base import Observation
from .base import Policy


class PmmlPolicy(Policy):
    def __init__(
        self,
        path: str | Path,
        name: str = "pmml",
        feature_names: list[str] | None = None,
        n_actions: int | None = None,
    ) -> None:
        try:
            from pypmml import Model  # type: ignore
        except ImportError as e:
            raise ImportError(
                "pypmml is required for PmmlPolicy. install with `pip install pypmml`"
            ) from e

        self.name = name
        self._model = Model.load(str(path))
        self._feature_names = feature_names
        self._n_actions = n_actions

    def act(self, observation: Observation) -> int:
        flat = observation.array.flatten().tolist()
        names = self._feature_names or [f"f{i}" for i in range(len(flat))]
        record = dict(zip(names, flat))
        result = self._model.predict(record)

        # 1. probability(<action>) フィールドを優先的に使う
        n_act = self._n_actions or (max(observation.legal_actions) + 1)
        scores = np.full(n_act, fill_value=-1e30, dtype=np.float64)
        any_prob = False
        for k, v in result.items():
            if not k.startswith("probability("):
                continue
            try:
                idx = int(k[len("probability("):-1])
            except ValueError:
                continue
            if 0 <= idx < n_act:
                scores[idx] = float(v)
                any_prob = True
        if any_prob:
            masked = scores.copy()
            for i in range(n_act):
                if i not in observation.legal_actions:
                    masked[i] = -1e30
            return int(np.argmax(masked))

        # 2. predicted_<target> から直接行動が返るケース
        for k, v in result.items():
            if k.startswith("predicted_"):
                action = int(v)
                if action in observation.legal_actions:
                    return action
                break

        # 3. fallback: legal の最初の手
        return observation.legal_actions[0]
