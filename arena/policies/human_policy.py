"""コンソール入力で人間が指す Policy。"""
from __future__ import annotations

from envs.base import Env, Observation
from .base import Policy


class HumanPolicy(Policy):
    name = "human"

    def __init__(self, env: Env, prompt: str = "your move> ") -> None:
        # 入力をパース・表示するために env への参照が必要。
        self._env = env
        self._prompt = prompt

    def act(self, observation: Observation) -> int:
        legal = observation.legal_actions
        legal_repr = ", ".join(self._env.action_name(a) for a in legal)
        print(f"\n{self._env.render()}")
        print(f"legal: {legal_repr}")
        while True:
            text = input(self._prompt)
            try:
                action = self._env.parse_action(text)
            except Exception as e:
                print(f"  parse error: {e}")
                continue
            if action not in legal:
                print(f"  illegal action {action}, choose from legal list")
                continue
            return action
