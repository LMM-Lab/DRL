"""三目並べ(Tic-tac-toe)環境。"""
from __future__ import annotations

import numpy as np

from ..base import Env, Observation, StepResult


_LINES = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),    # 横
    (0, 3, 6), (1, 4, 7), (2, 5, 8),    # 縦
    (0, 4, 8), (2, 4, 6),               # 斜め
)


class TicTacToeEnv(Env):
    name = "tictactoe"
    n_actions = 9
    obs_shape = (2, 3, 3)
    n_players = 2

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> Observation:
        self._board = np.zeros(9, dtype=np.int8)   # 0=空, 1=先手, -1=後手
        self._current = 0
        self._winner: int | None = None
        self._done = False
        return self.observation()

    def step(self, action: int) -> StepResult:
        if self._done:
            raise RuntimeError("step called after game over")
        if not (0 <= action < 9) or self._board[action] != 0:
            raise ValueError(f"illegal action {action}")
        mover = self._current
        sign = 1 if mover == 0 else -1
        self._board[action] = sign
        reward = 0.0
        if self._check_win(sign):
            self._winner = mover
            self._done = True
            reward = 1.0
        elif (self._board != 0).all():
            self._winner = -1
            self._done = True
        else:
            self._current = 1 - mover
        return StepResult(self.observation(), reward, self._done, {"winner": self._winner})

    def _check_win(self, sign: int) -> bool:
        return any(all(self._board[i] == sign for i in line) for line in _LINES)

    def legal_actions(self) -> list[int]:
        if self._done:
            return []
        return [i for i in range(9) if self._board[i] == 0]

    @property
    def current_player(self) -> int:
        return self._current

    @property
    def winner(self) -> int | None:
        return self._winner

    @property
    def done(self) -> bool:
        return self._done

    def observation(self) -> Observation:
        sign = 1 if self._current == 0 else -1
        own = (self._board == sign).astype(np.float32).reshape(3, 3)
        opp = (self._board == -sign).astype(np.float32).reshape(3, 3)
        array = np.stack([own, opp], axis=0)
        chars = []
        for v in self._board:
            if v == sign:
                chars.append("X")
            elif v == -sign:
                chars.append("O")
            else:
                chars.append(".")
        return Observation(
            array=array,
            state_key="".join(chars),
            current_player=self._current,
            legal_actions=self.legal_actions(),
        )

    def render(self) -> str:
        sym = lambda v: "X" if v == 1 else "O" if v == -1 else "."
        rows = [" ".join(sym(self._board[r * 3 + c]) for c in range(3)) for r in range(3)]
        rows.append(f"to move: {'X' if self._current == 0 else 'O'}  (action: 0..8 or r,c)")
        return "\n".join(rows)

    def clone(self) -> "TicTacToeEnv":
        e = TicTacToeEnv.__new__(TicTacToeEnv)
        e._board = self._board.copy()
        e._current = self._current
        e._winner = self._winner
        e._done = self._done
        return e

    def action_name(self, action: int) -> str:
        return f"({action // 3},{action % 3})"

    def parse_action(self, text: str) -> int:
        text = text.strip()
        if "," in text:
            r, c = text.split(",")
            return int(r) * 3 + int(c)
        return int(text)
