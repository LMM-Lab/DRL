"""オセロ(リバーシ)環境。8x8、パスあり。"""
from __future__ import annotations

import numpy as np

from ..base import Env, Observation, StepResult


_DIRS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
PASS = 64


class OthelloEnv(Env):
    name = "othello"
    n_actions = 65   # 0..63 = 着手, 64 = パス
    obs_shape = (2, 8, 8)
    n_players = 2

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> Observation:
        self._board = np.zeros((8, 8), dtype=np.int8)
        self._board[3, 3] = -1
        self._board[3, 4] = 1
        self._board[4, 3] = 1
        self._board[4, 4] = -1
        self._current = 0    # 黒(=1)が先手
        self._winner: int | None = None
        self._done = False
        self._last_pass = False
        return self.observation()

    @staticmethod
    def _sign(player: int) -> int:
        return 1 if player == 0 else -1

    def _flips(self, r: int, c: int, sign: int) -> list[tuple[int, int]]:
        if self._board[r, c] != 0:
            return []
        flips: list[tuple[int, int]] = []
        for dr, dc in _DIRS:
            line: list[tuple[int, int]] = []
            rr, cc = r + dr, c + dc
            while 0 <= rr < 8 and 0 <= cc < 8 and self._board[rr, cc] == -sign:
                line.append((rr, cc))
                rr += dr
                cc += dc
            if line and 0 <= rr < 8 and 0 <= cc < 8 and self._board[rr, cc] == sign:
                flips.extend(line)
        return flips

    def _legal_moves_for(self, player: int) -> list[int]:
        sign = self._sign(player)
        moves = []
        for r in range(8):
            for c in range(8):
                if self._board[r, c] == 0 and self._flips(r, c, sign):
                    moves.append(r * 8 + c)
        return moves

    def legal_actions(self) -> list[int]:
        if self._done:
            return []
        moves = self._legal_moves_for(self._current)
        return moves if moves else [PASS]

    def step(self, action: int) -> StepResult:
        if self._done:
            raise RuntimeError("step called after game over")
        legal = self.legal_actions()
        if action not in legal:
            raise ValueError(f"illegal action {action}")
        mover = self._current
        sign = self._sign(mover)
        if action == PASS:
            if self._last_pass:
                self._finish_score()
            else:
                self._last_pass = True
                self._current = 1 - mover
        else:
            r, c = divmod(action, 8)
            flips = self._flips(r, c, sign)
            self._board[r, c] = sign
            for fr, fc in flips:
                self._board[fr, fc] = sign
            self._last_pass = False
            if (self._board != 0).all():
                self._finish_score()
            else:
                self._current = 1 - mover
        reward = 0.0
        if self._done and self._winner is not None:
            if self._winner == mover:
                reward = 1.0
            elif self._winner == -1:
                reward = 0.0
            else:
                reward = -1.0
        return StepResult(self.observation(), reward, self._done, {"winner": self._winner})

    def _finish_score(self) -> None:
        black = int((self._board == 1).sum())
        white = int((self._board == -1).sum())
        if black > white:
            self._winner = 0
        elif white > black:
            self._winner = 1
        else:
            self._winner = -1
        self._done = True

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
        sign = self._sign(self._current)
        own = (self._board == sign).astype(np.float32)
        opp = (self._board == -sign).astype(np.float32)
        array = np.stack([own, opp], axis=0)
        chars = []
        for v in self._board.flatten():
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
        legal = set(self.legal_actions())
        lines = ["  " + " ".join(str(c) for c in range(8))]
        for r in range(8):
            row = []
            for c in range(8):
                v = int(self._board[r, c])
                if v == 0 and (r * 8 + c) in legal:
                    row.append("*")
                else:
                    row.append(sym(v))
            lines.append(f"{r} " + " ".join(row))
        b = int((self._board == 1).sum())
        w = int((self._board == -1).sum())
        mover = "X(black)" if self._current == 0 else "O(white)"
        lines.append(f"to move: {mover}  black={b} white={w}  (* = legal)")
        return "\n".join(lines)

    def clone(self) -> "OthelloEnv":
        e = OthelloEnv.__new__(OthelloEnv)
        e._board = self._board.copy()
        e._current = self._current
        e._winner = self._winner
        e._done = self._done
        e._last_pass = self._last_pass
        return e

    def action_name(self, action: int) -> str:
        if action == PASS:
            return "pass"
        return f"({action // 8},{action % 8})"

    def parse_action(self, text: str) -> int:
        text = text.strip().lower()
        if text in ("pass", "p"):
            return PASS
        if "," in text:
            r, c = text.split(",")
            return int(r) * 8 + int(c)
        return int(text)
