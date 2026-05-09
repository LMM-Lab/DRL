"""ONNXで保存された DNN モデルを推論で動かすポリシー。

入力:
    - 環境の `obs_shape` の先頭にバッチ次元1を付けた float32 配列。
    - 例: tictactoe → (1, 2, 3, 3)、othello → (1, 2, 8, 8)。
出力:
    - shape (1, n_actions) の Q値 / logits。
    - AlphaZero 系で policy/value の2出力がある場合は `policy_index` で policy 出力を指定。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from envs.base import Observation
from .base import Policy


class OnnxPolicy(Policy):
    def __init__(
        self,
        path: str | Path,
        name: str = "onnx",
        policy_index: int = 0,
        input_name: str | None = None,
    ) -> None:
        # onnxruntime はインポートが重いので遅延 import。
        import onnxruntime as ort  # type: ignore

        self.name = name
        self._session = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
        self._input_name = input_name or self._session.get_inputs()[0].name
        self._policy_index = policy_index
        # 入力の dtype を確認(float32 想定だが念のため)
        self._input_dtype = np.float32

    def act(self, observation: Observation) -> int:
        x = observation.array.astype(self._input_dtype, copy=False)[None, ...]
        outputs = self._session.run(None, {self._input_name: x})
        logits = np.asarray(outputs[self._policy_index]).reshape(-1)
        if logits.shape[0] == 0:
            raise RuntimeError("onnx model returned empty output")
        masked = np.full_like(logits, fill_value=-1e30)
        for a in observation.legal_actions:
            if a < masked.shape[0]:
                masked[a] = logits[a]
        return int(np.argmax(masked))
