"""環境の基本契約テスト。

全環境共通:
  - reset / step / observation の shape
  - clone の独立性
  - 違法手を打つと ValueError
  - ランダムプレイで終局し winner が {-1, 0, 1} のいずれか
"""
from __future__ import annotations

import random

import numpy as np
import pytest

from envs import REGISTRY, make


@pytest.mark.parametrize("env_name", sorted(REGISTRY))
def test_reset_returns_observation(env_name):
    env = make(env_name)
    obs = env.reset()
    assert obs.array.shape == env.obs_shape
    assert obs.array.dtype == np.float32
    assert isinstance(obs.state_key, str) and len(obs.state_key) > 0
    assert obs.current_player in (0, 1)
    assert len(obs.legal_actions) > 0
    assert env.done is False
    assert env.winner is None


@pytest.mark.parametrize("env_name", sorted(REGISTRY))
def test_random_playthrough(env_name):
    rng = random.Random(0)
    env = make(env_name)
    for _ in range(10):   # 何度も新しいゲームを回す
        env.reset()
        steps = 0
        while not env.done and steps < 1000:
            obs = env.observation()
            action = rng.choice(obs.legal_actions)
            env.step(action)
            steps += 1
        assert env.done, f"{env_name} did not terminate within 1000 steps"
        assert env.winner in (-1, 0, 1)


@pytest.mark.parametrize("env_name", sorted(REGISTRY))
def test_clone_is_independent(env_name):
    env = make(env_name)
    env.reset()
    obs = env.observation()
    a = obs.legal_actions[0]
    twin = env.clone()
    env.step(a)
    # twin should be unaffected
    assert twin.current_player == obs.current_player
    assert twin.done is False
    # observations should still match the original pre-step state
    twin_obs = twin.observation()
    assert twin_obs.state_key == obs.state_key


@pytest.mark.parametrize("env_name", sorted(REGISTRY))
def test_illegal_action_raises(env_name):
    env = make(env_name)
    env.reset()
    obs = env.observation()
    illegal = next((a for a in range(env.n_actions) if a not in obs.legal_actions), None)
    if illegal is None:
        pytest.skip("no illegal action exists at start")
    with pytest.raises((ValueError, IndexError)):
        env.step(illegal)


def test_tictactoe_horizontal_win():
    env = make("tictactoe")
    env.reset()
    # Player 0 (X): 0, 1, 2  Player 1 (O): 3, 4
    for a in [0, 3, 1, 4, 2]:
        env.step(a)
    assert env.done
    assert env.winner == 0


def test_connect4_vertical_win():
    env = make("connect4")
    env.reset()
    # P0: col 0 ×4, P1: col 1 ×3
    for a in [0, 1, 0, 1, 0, 1, 0]:
        env.step(a)
    assert env.done
    assert env.winner == 0


def test_othello_initial_board():
    env = make("othello")
    obs = env.reset()
    assert env.current_player == 0
    # 黒の合法手は4つ(d3 = (2,3) 相当の位置たち)
    assert len(obs.legal_actions) == 4
