"""対戦環境(運営が用意するゲーム実装)。"""
from __future__ import annotations

from .base import Env, Observation, StepResult
from .registry import REGISTRY, register, make

__all__ = ["Env", "Observation", "StepResult", "REGISTRY", "register", "make"]
