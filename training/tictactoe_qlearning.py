"""三目並べの Q-learning による自己対戦学習サンプル。CSV を出力。"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from envs.tictactoe.env import TicTacToeEnv  # noqa: E402


def best_action(q: dict[int, float], legal: list[int], rng: random.Random) -> int:
    best_v = float("-inf")
    best_a: list[int] = []
    for a in legal:
        v = q.get(a, 0.0)
        if v > best_v:
            best_v = v
            best_a = [a]
        elif v == best_v:
            best_a.append(a)
    return rng.choice(best_a)


def select_action(table, key: str, legal: list[int], eps: float, rng: random.Random) -> int:
    if rng.random() < eps:
        return rng.choice(legal)
    return best_action(table[key], legal, rng)


def train(episodes: int, alpha: float, gamma: float, eps_start: float, eps_end: float,
          seed: int) -> dict[str, dict[int, float]]:
    rng = random.Random(seed)
    table: dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(float))

    for ep in range(episodes):
        eps = eps_start + (eps_end - eps_start) * (ep / max(1, episodes - 1))
        env = TicTacToeEnv()
        # 各プレイヤー視点での (state_key, action) 履歴を取り、終局時に bootstrap を遡らせる。
        history: list[tuple[str, int, int]] = []   # (key, action, mover)
        while not env.done:
            obs = env.observation()
            mover = env.current_player
            action = select_action(table, obs.state_key, obs.legal_actions, eps, rng)
            history.append((obs.state_key, action, mover))
            env.step(action)
        # 報酬: mover ごとに +1/-1/0
        if env.winner == -1:
            rewards = {0: 0.0, 1: 0.0}
        else:
            rewards = {env.winner: 1.0, 1 - env.winner: -1.0}

        # 各プレイヤー視点で逆順に Q を更新する
        future: dict[int, float] = {0: 0.0, 1: 0.0}
        for key, action, mover in reversed(history):
            target = rewards[mover] + gamma * future[mover]
            old = table[key][action]
            new = old + alpha * (target - old)
            table[key][action] = new
            future[mover] = max(table[key].values())
            # 終端報酬は最終ステップで一度しか足さない。
            rewards[mover] = 0.0
    return {k: dict(v) for k, v in table.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100_000)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, required=True,
                        help="提出物ディレクトリ。中に model.csv と meta.json を書く。")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--author", type=str, default="anonymous")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or out_dir.name

    print(f"training {args.episodes} episodes...")
    table = train(args.episodes, args.alpha, args.gamma, args.eps_start, args.eps_end, args.seed)

    csv_path = out_dir / "model.csv"
    rows = 0
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["state_key", "action", "value"])
        for key in sorted(table):
            for action in sorted(table[key]):
                w.writerow([key, action, f"{table[key][action]:.6f}"])
                rows += 1
    meta = {
        "env": "tictactoe",
        "type": "csv",
        "file": "model.csv",
        "author": args.author,
        "name": name,
        "description": (
            f"Q-learning self-play, {args.episodes} episodes, alpha={args.alpha}, "
            f"gamma={args.gamma}, eps {args.eps_start} -> {args.eps_end}"
        ),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {rows} rows / {len(table)} states -> {csv_path}")
    print(f"wrote meta.json -> {out_dir / 'meta.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
