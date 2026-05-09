"""2人ゼロ和・完全情報・交互手番ゲーム共通の抽象基底クラス。

全ての対戦環境はこのインターフェースを満たすこと。Policyとアリーナはこの抽象に
のみ依存する(具体的なゲームに依存しない)。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class Observation:
    """Policyに渡される観測。

    array     : ONNX等に流すcanonical(現プレイヤー視点)テンソル。shape=(C, H, W) など。
    state_key : テーブル系(CSV/PMML)で状態キーとして使う文字列。canonical済みであること。
    current_player : 0 or 1。
    legal_actions  : このターンで合法な行動のリスト。
    """

    array: np.ndarray
    state_key: str
    current_player: int
    legal_actions: list[int]


@dataclass
class StepResult:
    obs: Observation
    reward: float           # 直前手を打ったプレイヤー視点の即時報酬
    done: bool
    info: dict


class Env(ABC):
    # 環境固有メタデータ(クラス変数で宣言)
    name: str = "abstract"
    n_actions: int = 0
    obs_shape: tuple[int, ...] = ()
    n_players: int = 2

    @abstractmethod
    def reset(self) -> Observation: ...

    @abstractmethod
    def step(self, action: int) -> StepResult: ...

    @abstractmethod
    def legal_actions(self) -> list[int]: ...

    @property
    @abstractmethod
    def current_player(self) -> int: ...

    @property
    @abstractmethod
    def winner(self) -> int | None:
        """勝者を返す。0/1=プレイヤー、-1=引き分け、None=未決着。"""

    @property
    @abstractmethod
    def done(self) -> bool: ...

    @abstractmethod
    def render(self) -> str: ...

    @abstractmethod
    def clone(self) -> "Env":
        """完全コピー(MCTSやAlphaZeroで使う)。"""

    @abstractmethod
    def observation(self) -> Observation: ...

    # --- 共通ヘルパ ---
    def action_name(self, action: int) -> str:
        """行動の人間可読表記(デフォルトは数値)。CLIの表示や入力で使用。"""
        return str(action)

    def parse_action(self, text: str) -> int:
        """人間入力を行動に変換(デフォルトはintキャスト)。"""
        return int(text.strip())

    def legal_mask(self) -> np.ndarray:
        mask = np.zeros(self.n_actions, dtype=np.float32)
        for a in self.legal_actions():
            mask[a] = 1.0
        return mask
