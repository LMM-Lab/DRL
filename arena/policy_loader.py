"""提出物ディレクトリ(`submissions/<env>/<author>_<name>/`)から Policy をロード。

ディレクトリ直下の `meta.json` を読み、`type` に応じて適切な Policy を返す。

特殊識別子:
    - "random"  : `RandomPolicy` を返す(提出ディレクトリ不要)。
    - "<env>:examples/<example_name>" : `envs/<env>/examples/<example_name>` から読む。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .policies.base import Policy
from .policies.csv_policy import CsvPolicy
from .policies.onnx_policy import OnnxPolicy
from .policies.random_policy import RandomPolicy

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class LoadedPolicy:
    policy: Policy
    meta: dict[str, Any]
    path: Path | None


def _load_pmml(path: Path, meta: dict[str, Any]) -> Policy:
    from .policies.pmml_policy import PmmlPolicy   # 遅延 import
    return PmmlPolicy(
        path=path,
        name=meta.get("name", "pmml"),
        feature_names=meta.get("feature_names"),
        n_actions=meta.get("n_actions"),
    )


def resolve_path(spec: str) -> Path:
    """spec をディレクトリパスに解決。

    受け付ける形式:
      - 絶対/相対パス(例: `submissions/tictactoe/alice_dqn`)
      - `<env>:examples/<example_name>` (例: `tictactoe:examples/random`)
    """
    if ":" in spec and not Path(spec).exists():
        env_name, sub = spec.split(":", 1)
        return REPO_ROOT / "envs" / env_name / sub
    p = Path(spec)
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve() if (Path.cwd() / p).exists() else REPO_ROOT / p
    return p


def load_policy(spec: str, env_name: str | None = None) -> LoadedPolicy:
    """`spec` から Policy をロード。

    `spec == "random"` は提出物が無くても動く特別ケース。それ以外は meta.json を読む。
    `env_name` 指定時は meta.json の env と一致するか検証する。
    """
    if spec == "random":
        return LoadedPolicy(policy=RandomPolicy(), meta={"type": "random", "name": "random"}, path=None)

    path = resolve_path(spec)
    if not path.is_dir():
        raise FileNotFoundError(f"policy directory not found: {path}")
    meta_file = path / "meta.json"
    if not meta_file.is_file():
        raise FileNotFoundError(f"meta.json not found in {path}")
    meta = json.loads(meta_file.read_text(encoding="utf-8"))

    if env_name and meta.get("env") and meta["env"] != env_name:
        raise ValueError(
            f"meta.json env={meta['env']!r} does not match expected env={env_name!r}"
        )

    type_ = meta.get("type")
    name = meta.get("name") or path.name
    file_field = meta.get("file")

    if type_ == "random":
        return LoadedPolicy(policy=RandomPolicy(), meta=meta, path=path)

    if not file_field:
        raise ValueError(f"meta.json missing 'file' field: {meta_file}")
    model_path = path / file_field

    if type_ == "csv":
        return LoadedPolicy(policy=CsvPolicy(model_path, name=name), meta=meta, path=path)
    if type_ == "onnx":
        return LoadedPolicy(
            policy=OnnxPolicy(
                model_path,
                name=name,
                policy_index=int(meta.get("policy_index", 0)),
                input_name=meta.get("input_name"),
            ),
            meta=meta,
            path=path,
        )
    if type_ == "pmml":
        return LoadedPolicy(policy=_load_pmml(model_path, meta), meta=meta, path=path)

    raise ValueError(f"unknown policy type {type_!r} in {meta_file}")
