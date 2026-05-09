"""rl-arena CLI。

使用例:

    # 環境一覧
    python -m arena.cli list-envs

    # モデル vs モデル(2戦、座席入れ替え)
    python -m arena.cli play --env tictactoe --p1 random --p2 tictactoe:examples/qtable_demo --games 2

    # 人間 vs モデル
    python -m arena.cli human --env tictactoe --opponent tictactoe:examples/qtable_demo

    # トーナメント
    python -m arena.cli tournament --env tictactoe --dir submissions/tictactoe --games 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from envs import make as make_env
from envs.registry import REGISTRY

from .policies.human_policy import HumanPolicy
from .policy_loader import load_policy, resolve_path
from .runner import (
    run_match,
    run_tournament,
    save_match_jsonl,
    save_tournament_json,
)


def _format_match(result, env_name: str, seat_names: tuple[str, str]) -> str:
    if result.illegal_player is not None:
        loser = seat_names[result.illegal_player]
        winner = seat_names[1 - result.illegal_player]
        reason = f"illegal/error: {result.error_message}"
        return f"[{env_name}] {seat_names[0]} vs {seat_names[1]}: {winner} wins ({loser} {reason})"
    if result.winner == -1:
        return f"[{env_name}] {seat_names[0]} vs {seat_names[1]}: draw ({result.n_moves} moves)"
    winner = seat_names[result.winner]
    return f"[{env_name}] {seat_names[0]} vs {seat_names[1]}: {winner} wins ({result.n_moves} moves)"


def cmd_list_envs(args: argparse.Namespace) -> int:
    print("registered envs:")
    for name in sorted(REGISTRY):
        env = REGISTRY[name]()
        print(f"  - {name}: n_actions={env.n_actions}, obs_shape={env.obs_shape}")
    return 0


def cmd_play(args: argparse.Namespace) -> int:
    env_name = args.env
    p1 = load_policy(args.p1, env_name=env_name)
    p2 = load_policy(args.p2, env_name=env_name)
    results = []
    for g in range(args.games):
        env = make_env(env_name)
        if args.swap and g % 2 == 1:
            policies = (p2.policy, p1.policy)
            seat_names = (p2.policy.name, p1.policy.name)
        else:
            policies = (p1.policy, p2.policy)
            seat_names = (p1.policy.name, p2.policy.name)
        result = run_match(env, policies, render=args.render)
        print(_format_match(result, env_name, seat_names))
        results.append(result)
    if args.log:
        save_match_jsonl(results, args.log)
        print(f"saved {len(results)} games to {args.log}")
    return 0


def cmd_human(args: argparse.Namespace) -> int:
    env_name = args.env
    opp = load_policy(args.opponent, env_name=env_name)
    env = make_env(env_name)
    human = HumanPolicy(env)
    if args.first == "me":
        policies = (human, opp.policy)
        seat_names = (human.name, opp.policy.name)
    else:
        policies = (opp.policy, human)
        seat_names = (opp.policy.name, human.name)
    result = run_match(env, policies, render=False)
    print(env.render())
    print(_format_match(result, env_name, seat_names))
    return 0


def cmd_tournament(args: argparse.Namespace) -> int:
    env_name = args.env
    base = Path(args.dir)
    if not base.is_absolute():
        base = resolve_path(args.dir)
    if not base.is_dir():
        print(f"directory not found: {base}", file=sys.stderr)
        return 2

    entries = {}
    for sub in sorted(p for p in base.iterdir() if p.is_dir()):
        meta_file = sub / "meta.json"
        if not meta_file.is_file():
            print(f"  skip {sub.name} (no meta.json)")
            continue
        try:
            loaded = load_policy(str(sub), env_name=env_name)
        except Exception as e:   # noqa: BLE001 - 1件失敗しても続行
            print(f"  skip {sub.name}: {e}")
            continue
        entries[loaded.policy.name] = loaded.policy
    if args.include_random:
        from .policies.random_policy import RandomPolicy
        entries.setdefault("random", RandomPolicy())
    if len(entries) < 2:
        print(f"need at least 2 entries, got {len(entries)}: {sorted(entries)}", file=sys.stderr)
        return 2

    print(f"tournament [{env_name}] with {len(entries)} entries: {sorted(entries)}")
    result = run_tournament(
        env_factory=lambda: make_env(env_name),
        entries=entries,
        games_per_pair=args.games,
    )
    print()
    print(f"== standings ({result.games_per_pair} games/pair) ==")
    print(f"{'name':<28} {'W':>3} {'D':>3} {'L':>3} {'score':>6} {'elo':>7}")
    for name, w, d, l, score in result.standings:
        print(f"{name:<28} {w:>3} {d:>3} {l:>3} {score:>6.1f} {result.elo[name]:>7.1f}")
    if args.out:
        save_tournament_json(result, args.out)
        print(f"\nsaved tournament result to {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rl-arena")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list-envs", help="登録された環境一覧")
    p_list.set_defaults(func=cmd_list_envs)

    p_play = sub.add_parser("play", help="モデル同士の対戦")
    p_play.add_argument("--env", required=True)
    p_play.add_argument("--p1", required=True, help="提出物パス、または 'random'")
    p_play.add_argument("--p2", required=True, help="提出物パス、または 'random'")
    p_play.add_argument("--games", type=int, default=1)
    p_play.add_argument("--swap", action="store_true", help="奇数試合目で座席を入れ替える")
    p_play.add_argument("--render", action="store_true")
    p_play.add_argument("--log", type=str, default=None, help="MatchResult を JSONL 保存するパス")
    p_play.set_defaults(func=cmd_play)

    p_human = sub.add_parser("human", help="人間 vs モデル(コンソール対戦)")
    p_human.add_argument("--env", required=True)
    p_human.add_argument("--opponent", required=True)
    p_human.add_argument("--first", choices=["me", "ai"], default="me")
    p_human.set_defaults(func=cmd_human)

    p_tour = sub.add_parser("tournament", help="総当たり戦")
    p_tour.add_argument("--env", required=True)
    p_tour.add_argument("--dir", required=True, help="提出物の親ディレクトリ")
    p_tour.add_argument("--games", type=int, default=2)
    p_tour.add_argument("--include-random", action="store_true")
    p_tour.add_argument("--out", type=str, default=None, help="JSON で結果保存")
    p_tour.set_defaults(func=cmd_tournament)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
