"""Connect Four(四目並べ・コネクトフォー)環境。6行×7列。"""
from __future__ import annotations

import numpy as np

from ..base import Env, Observation, StepResult


class Connect4Env(Env):
    name = "connect4"
    n_actions = 7
    obs_shape = (2, 6, 7)
    n_players = 2

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> Observation:
        self._board = np.zeros((6, 7), dtype=np.int8)   # row 0 が一番下
        self._heights = np.zeros(7, dtype=np.int8)
        self._current = 0
        self._winner: int | None = None
        self._done = False
        return self.observation()

    def step(self, action: int) -> StepResult:
        if self._done:
            raise RuntimeError("step called after game over")
        if not (0 <= action < 7) or self._heights[action] >= 6:
            raise ValueError(f"illegal action {action}")
        mover = self._current
        sign = 1 if mover == 0 else -1
        r = int(self._heights[action])
        self._board[r, action] = sign
        self._heights[action] += 1
        reward = 0.0
        if self._check_win(r, action, sign):
            self._winner = mover
            self._done = True
            reward = 1.0
        elif (self._heights == 6).all():
            self._winner = -1
            self._done = True
        else:
            self._current = 1 - mover
        return StepResult(self.observation(), reward, self._done, {"winner": self._winner})

    def _check_win(self, r: int, c: int, sign: int) -> bool:
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            count = 1
            for d in (1, -1):
                rr, cc = r + dr * d, c + dc * d
                while 0 <= rr < 6 and 0 <= cc < 7 and self._board[rr, cc] == sign:
                    count += 1
                    rr += dr * d
                    cc += dc * d
            if count >= 4:
                return True
        return False

    def legal_actions(self) -> list[int]:
        if self._done:
            return []
        return [c for c in range(7) if self._heights[c] < 6]

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
        own = (self._board == sign).astype(np.float32)
        opp = (self._board == -sign).astype(np.float32)
        array = np.stack([own, opp], axis=0)
        chars = []
        # 表示しやすさのため上の行(row 5)から書き出す
        for r in range(5, -1, -1):
            for c in range(7):
                v = int(self._board[r, c])
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
        lines = []
        for r in range(5, -1, -1):
            lines.append(" ".join(sym(int(self._board[r, c])) for c in range(7)))
        lines.append(" ".join(str(c) for c in range(7)))
        lines.append(f"to move: {'X' if self._current == 0 else 'O'}  (action: 0..6)")
        return "\n".join(lines)

    def clone(self) -> "Connect4Env":
        e = Connect4Env.__new__(Connect4Env)
        e._board = self._board.copy()
        e._heights = self._heights.copy()
        e._current = self._current
        e._winner = self._winner
        e._done = self._done
        return e

    def parse_action(self, text: str) -> int:
        return int(text.strip())
