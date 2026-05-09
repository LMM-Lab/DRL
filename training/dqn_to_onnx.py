"""DQN を PyTorch で学習して ONNX 出力するサンプル(対応: tictactoe / connect4)。

依存: torch。`pip install torch` してから実行。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from envs import make as make_env  # noqa: E402
from envs.base import Env  # noqa: E402


def _build_net(obs_shape: tuple[int, ...], n_actions: int):
    import torch
    import torch.nn as nn

    c, h, w = obs_shape
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(c * h * w, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, n_actions),
    )


def select_action(net, obs_array, legal_actions, eps, rng):
    import torch
    if rng.random() < eps:
        return rng.choice(legal_actions)
    with torch.no_grad():
        q = net(torch.from_numpy(obs_array).float().unsqueeze(0)).squeeze(0).numpy()
    masked = np.full_like(q, fill_value=-1e30)
    for a in legal_actions:
        masked[a] = q[a]
    return int(np.argmax(masked))


def train(env_name: str, steps: int, batch: int, gamma: float, lr: float, seed: int) -> tuple:
    import torch
    import torch.nn.functional as F

    rng = random.Random(seed)
    torch.manual_seed(seed)
    env: Env = make_env(env_name)
    net = _build_net(env.obs_shape, env.n_actions)
    target = _build_net(env.obs_shape, env.n_actions)
    target.load_state_dict(net.state_dict())
    optim = torch.optim.Adam(net.parameters(), lr=lr)

    buf: deque = deque(maxlen=50_000)
    obs = env.reset()
    s = obs.array
    legal = obs.legal_actions
    eps = 1.0
    for step in range(1, steps + 1):
        eps = max(0.05, 1.0 - step / (0.7 * steps))
        a = select_action(net, s, legal, eps, rng)
        result = env.step(a)
        r = result.reward
        done = result.done
        s2 = result.obs.array
        legal2 = result.obs.legal_actions
        buf.append((s, a, r, s2, done, env.legal_mask().copy() if not done else None))
        if done:
            obs = env.reset()
            s, legal = obs.array, obs.legal_actions
        else:
            s, legal = s2, legal2

        if len(buf) >= batch:
            samples = rng.sample(list(buf), batch)
            ss = torch.tensor(np.stack([x[0] for x in samples]), dtype=torch.float32)
            aa = torch.tensor([x[1] for x in samples], dtype=torch.long)
            rr = torch.tensor([x[2] for x in samples], dtype=torch.float32)
            ss2 = torch.tensor(np.stack([x[3] for x in samples]), dtype=torch.float32)
            dd = torch.tensor([x[4] for x in samples], dtype=torch.float32)
            with torch.no_grad():
                qn = target(ss2).max(dim=1).values
                tgt = rr + gamma * (1 - dd) * qn
            qv = net(ss).gather(1, aa.unsqueeze(1)).squeeze(1)
            loss = F.smooth_l1_loss(qv, tgt)
            optim.zero_grad()
            loss.backward()
            optim.step()
        if step % 1000 == 0:
            target.load_state_dict(net.state_dict())
        if step % 5000 == 0:
            print(f"step={step} eps={eps:.3f} buf={len(buf)}")
    return net, env


def export_onnx(net, env: Env, out_path: Path) -> None:
    import torch
    net.eval()
    dummy = torch.zeros((1, *env.obs_shape), dtype=torch.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        net,
        dummy,
        str(out_path),
        input_names=["obs"],
        output_names=["q"],
        opset_version=17,
        dynamic_axes={"obs": {0: "batch"}, "q": {0: "batch"}},
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True, choices=["tictactoe", "connect4", "othello"])
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True, help="提出物ディレクトリ")
    parser.add_argument("--author", default="anonymous")
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    net, env = train(args.env, args.steps, args.batch, args.gamma, args.lr, args.seed)
    out_dir = Path(args.out)
    onnx_path = out_dir / "model.onnx"
    export_onnx(net, env, onnx_path)
    meta = {
        "env": args.env,
        "type": "onnx",
        "file": "model.onnx",
        "author": args.author,
        "name": args.name or out_dir.name,
        "description": f"DQN, {args.steps} steps, gamma={args.gamma}, lr={args.lr}",
        "input_name": "obs",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {onnx_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
