"""対戦実行ロジック。1試合の進行・トーナメント・結果集計。"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable

from envs.base import Env
from .policies.base import Policy


@dataclass
class MoveRecord:
    player: int
    action: int
    state_key: str


@dataclass
class MatchResult:
    winner: int                # 0 or 1, -1 = draw
    illegal_player: int | None  # 違法手を打って即敗北したプレイヤー(無ければ None)
    error_message: str | None
    n_moves: int
    duration_sec: float
    seats: tuple[str, str]      # (player0 name, player1 name)
    moves: list[MoveRecord] = field(default_factory=list)


def run_match(
    env: Env,
    policies: tuple[Policy, Policy] | list[Policy],
    render: bool = False,
    max_moves: int = 10_000,
    record_moves: bool = True,
) -> MatchResult:
    """1試合進行する。`policies[i]` が seat `i` のプレイヤー。"""
    if len(policies) != 2:
        raise ValueError("run_match expects exactly 2 policies")
    env.reset()
    for p in policies:
        p.reset()
    moves: list[MoveRecord] = []
    t0 = time.perf_counter()
    illegal_player: int | None = None
    error_message: str | None = None
    n = 0

    while not env.done and n < max_moves:
        cp = env.current_player
        obs = env.observation()
        try:
            action = policies[cp].act(obs)
        except Exception as e:   # noqa: BLE001 - Policy 例外も即敗北扱い
            illegal_player = cp
            error_message = f"{type(e).__name__}: {e}"
            break
        if action not in obs.legal_actions:
            illegal_player = cp
            error_message = f"illegal action: {action}"
            break
        if record_moves:
            moves.append(MoveRecord(player=cp, action=int(action), state_key=obs.state_key))
        env.step(action)
        n += 1
        if render:
            print(env.render())
            print()

    duration = time.perf_counter() - t0
    if illegal_player is not None:
        winner = 1 - illegal_player
    elif env.winner is None:
        winner = -1
    else:
        winner = env.winner
    return MatchResult(
        winner=winner,
        illegal_player=illegal_player,
        error_message=error_message,
        n_moves=n,
        duration_sec=duration,
        seats=(policies[0].name, policies[1].name),
        moves=moves if record_moves else [],
    )


# --- トーナメント -----------------------------------------------------------

@dataclass
class PairStats:
    p1: str
    p2: str
    games: int = 0
    p1_wins: int = 0
    p2_wins: int = 0
    draws: int = 0
    p1_illegal: int = 0
    p2_illegal: int = 0


@dataclass
class TournamentResult:
    env_name: str
    games_per_pair: int
    pairs: list[PairStats]
    elo: dict[str, float]
    standings: list[tuple[str, int, int, int, float]]   # (name, w, d, l, score)


def _expected(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))


def run_tournament(
    env_factory,
    entries: dict[str, Policy],
    games_per_pair: int = 2,
    render: bool = False,
    elo_k: float = 20.0,
    initial_elo: float = 1500.0,
    swap_seats: bool = True,
) -> TournamentResult:
    """総当たり戦(round-robin)。

    `swap_seats=True` で同じペアの試合を seat を入れ替えて偶数試合行う。
    """
    names = list(entries.keys())
    elo: dict[str, float] = {n: initial_elo for n in names}
    wins: dict[str, int] = {n: 0 for n in names}
    losses: dict[str, int] = {n: 0 for n in names}
    draws: dict[str, int] = {n: 0 for n in names}
    pair_stats: list[PairStats] = []

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            stats = PairStats(p1=a, p2=b)
            for g in range(games_per_pair):
                if swap_seats and g % 2 == 1:
                    seats = (entries[b], entries[a])
                    seat_names = (b, a)
                else:
                    seats = (entries[a], entries[b])
                    seat_names = (a, b)
                env = env_factory()
                result = run_match(env, seats, render=render)
                stats.games += 1
                if result.illegal_player is not None:
                    if seat_names[result.illegal_player] == a:
                        stats.p1_illegal += 1
                    else:
                        stats.p2_illegal += 1
                if result.winner == -1:
                    stats.draws += 1
                    draws[a] += 1
                    draws[b] += 1
                    sa = 0.5
                else:
                    winner_name = seat_names[result.winner]
                    if winner_name == a:
                        stats.p1_wins += 1
                        wins[a] += 1
                        losses[b] += 1
                        sa = 1.0
                    else:
                        stats.p2_wins += 1
                        wins[b] += 1
                        losses[a] += 1
                        sa = 0.0
                ea = _expected(elo[a], elo[b])
                elo[a] += elo_k * (sa - ea)
                elo[b] += elo_k * ((1.0 - sa) - (1.0 - ea))
            pair_stats.append(stats)

    standings = sorted(
        (
            (n, wins[n], draws[n], losses[n], wins[n] + 0.5 * draws[n])
            for n in names
        ),
        key=lambda t: (-t[4], -wins[t[0]], t[0]),
    )
    return TournamentResult(
        env_name=getattr(env_factory(), "name", "unknown"),
        games_per_pair=games_per_pair,
        pairs=pair_stats,
        elo=elo,
        standings=standings,
    )


def save_match_jsonl(results: Iterable[MatchResult], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


def save_tournament_json(result: TournamentResult, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "env": result.env_name,
        "games_per_pair": result.games_per_pair,
        "elo": result.elo,
        "standings": [
            {"name": n, "wins": w, "draws": d, "losses": l, "score": s}
            for (n, w, d, l, s) in result.standings
        ],
        "pairs": [asdict(p) for p in result.pairs],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
