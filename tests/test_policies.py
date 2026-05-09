"""ポリシーとランナーのテスト。"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from arena.policies.csv_policy import CsvPolicy
from arena.policies.random_policy import RandomPolicy
from arena.policy_loader import load_policy
from arena.runner import run_match, run_tournament
from envs import make


def test_random_policy_picks_legal():
    env = make("tictactoe")
    env.reset()
    pol = RandomPolicy(seed=0)
    for _ in range(100):
        env.reset()
        while not env.done:
            obs = env.observation()
            a = pol.act(obs)
            assert a in obs.legal_actions
            env.step(a)


def test_csv_policy_uses_max_value(tmp_path: Path):
    csv_path = tmp_path / "table.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["state_key", "action", "value"])
        # 初期状態 "........." で行動 4 (中央) を強く好む
        w.writerow(["........", "0", "0.1"])   # わざと不正な8文字も書く(無視されるべきではないので別ケース)
        w.writerow([".........", "0", "0.1"])
        w.writerow([".........", "4", "0.9"])
        w.writerow([".........", "8", "0.5"])

    pol = CsvPolicy(csv_path, name="t", seed=0)
    env = make("tictactoe")
    obs = env.reset()
    assert pol.act(obs) == 4


def test_runner_random_vs_random_terminates():
    env = make("tictactoe")
    p1 = RandomPolicy(seed=1)
    p2 = RandomPolicy(seed=2)
    result = run_match(env, (p1, p2))
    assert result.illegal_player is None
    assert result.winner in (-1, 0, 1)
    assert result.n_moves <= 9


def test_tournament_runs():
    entries = {"a": RandomPolicy(seed=1), "b": RandomPolicy(seed=2), "c": RandomPolicy(seed=3)}
    result = run_tournament(
        env_factory=lambda: make("tictactoe"),
        entries=entries,
        games_per_pair=4,
    )
    assert len(result.standings) == 3
    total_games = sum(p.games for p in result.pairs)
    assert total_games == 3 * 4   # 3 pairs * 4 games


def test_load_random_policy():
    loaded = load_policy("random")
    assert loaded.policy.name == "random"


def test_load_example_random_policy():
    loaded = load_policy("tictactoe:examples/random", env_name="tictactoe")
    assert loaded.meta["type"] == "random"
    env = make("tictactoe")
    obs = env.reset()
    assert loaded.policy.act(obs) in obs.legal_actions
