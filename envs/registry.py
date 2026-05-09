"""環境の名前→クラス登録テーブル。CLI/ローダーから `make(name)` で取得する。"""
from __future__ import annotations

from typing import Callable

from .base import Env

REGISTRY: dict[str, Callable[[], Env]] = {}


def register(name: str, factory: Callable[[], Env]) -> None:
    if name in REGISTRY:
        raise ValueError(f"env '{name}' already registered")
    REGISTRY[name] = factory


def make(name: str) -> Env:
    if name not in REGISTRY:
        raise KeyError(f"unknown env '{name}'. registered: {sorted(REGISTRY)}")
    return REGISTRY[name]()


# 環境を import するだけで登録されるようにする
def _auto_register() -> None:
    from .tictactoe.env import TicTacToeEnv
    from .othello.env import OthelloEnv
    from .connect4.env import Connect4Env

    if "tictactoe" not in REGISTRY:
        register("tictactoe", TicTacToeEnv)
    if "othello" not in REGISTRY:
        register("othello", OthelloEnv)
    if "connect4" not in REGISTRY:
        register("connect4", Connect4Env)


_auto_register()
