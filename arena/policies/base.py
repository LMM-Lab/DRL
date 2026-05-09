"""Policy抽象クラス。

Envとは独立しており、Observationから合法手の中から一つを選ぶことだけが責務。
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from envs.base import Observation


class Policy(ABC):
    name: str = "policy"

    @abstractmethod
    def act(self, observation: Observation) -> int:
        """合法手の中から1つを選んで返す。違法な行動を返すと即敗北になる。"""

    def reset(self) -> None:
        """1ゲームの開始時に呼ばれる。状態を持つPolicy(MCTS等)向け。"""
        return None

    def close(self) -> None:
        """セッション全体の終了時に呼ばれる。"""
        return None

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r}>"
