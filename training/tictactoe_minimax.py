"""三目並べを完全解析して Q-table CSV を生成するスクリプト。

`envs/tictactoe/examples/minimax/model.csv` に出力する(既存ファイルは上書き)。
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# パッケージとして実行できなくても import できるようにルートを追加
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from envs.tictactoe.env import TicTacToeEnv  # noqa: E402


def solve(env: TicTacToeEnv, value_memo: dict[str, float], q: dict[str, dict[int, float]]) -> float:
    """current_player 視点での状態価値を返す。env は終局していない前提。"""
    obs = env.observation()
    key = obs.state_key
    if key in value_memo:
        return value_memo[key]
    q.setdefault(key, {})
    best = float("-inf")
    for a in obs.legal_actions:
        nxt = env.clone()
        nxt.step(a)
        if nxt.done:
            if nxt.winner == -1:
                v = 0.0
            else:
                # tic-tac-toe は手番者が勝つ手のみが終局を生むため、勝ち=+1
                v = 1.0
        else:
            # 着手後は相手の手番。相手から見た価値を反転して自分視点に。
            v = -solve(nxt, value_memo, q)
        q[key][a] = v
        if v > best:
            best = v
    value_memo[key] = best
    return best


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=str,
        default=str(ROOT / "envs" / "tictactoe" / "examples" / "minimax" / "model.csv"),
    )
    args = parser.parse_args()

    env = TicTacToeEnv()
    value_memo: dict[str, float] = {}
    q: dict[str, dict[int, float]] = {}
    solve(env, value_memo, q)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["state_key", "action", "value"])
        for key in sorted(q):
            for action in sorted(q[key]):
                w.writerow([key, action, f"{q[key][action]:.6f}"])
                rows += 1
    print(f"wrote {rows} rows for {len(q)} states -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
