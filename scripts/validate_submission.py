"""提出物のスモーク検証スクリプト(CI からも呼ばれる)。

`submissions/<env>/<dir>/meta.json` を全て見つけて以下を確認:
  1. ロードに成功する
  2. 対 random で指定試合数走らせて、違法手や例外を出さない
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from arena.policies.random_policy import RandomPolicy  # noqa: E402
from arena.policy_loader import load_policy  # noqa: E402
from arena.runner import run_match  # noqa: E402
from envs import make as make_env  # noqa: E402


def find_submissions(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    out = []
    for env_dir in sorted(root.iterdir()):
        if not env_dir.is_dir():
            continue
        for sub in sorted(env_dir.iterdir()):
            if (sub / "meta.json").is_file():
                out.append(sub)
    return out


def _infer_env_name(submission: Path) -> str | None:
    meta_file = submission / "meta.json"
    if meta_file.is_file():
        try:
            import json
            return json.loads(meta_file.read_text(encoding="utf-8")).get("env")
        except Exception:   # noqa: BLE001
            pass
    return submission.parent.name if submission.parent.name not in {"examples"} else None


def validate_one(submission: Path, games: int) -> tuple[bool, str]:
    env_name = _infer_env_name(submission)
    if env_name is None:
        return False, "could not infer env name (meta.json missing 'env')"
    try:
        loaded = load_policy(str(submission), env_name=env_name)
    except Exception as e:   # noqa: BLE001
        return False, f"load failed: {type(e).__name__}: {e}"

    rng_seed = 0
    for g in range(games):
        try:
            env = make_env(env_name)
            opp = RandomPolicy(seed=rng_seed + g)
            seats = (loaded.policy, opp) if g % 2 == 0 else (opp, loaded.policy)
            r = run_match(env, seats)
        except Exception as e:   # noqa: BLE001
            return False, f"run failed (game {g}): {type(e).__name__}: {e}\n{traceback.format_exc()}"
        if r.illegal_player is not None:
            seat_names = (seats[0].name, seats[1].name)
            culprit = seat_names[r.illegal_player]
            if culprit == loaded.policy.name:
                return False, f"submission produced illegal action / error: {r.error_message}"
    return True, f"ok ({games} games vs random)"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="submissions")
    parser.add_argument("--games", type=int, default=4)
    parser.add_argument("--paths", nargs="*", default=None,
                        help="個別ディレクトリを指定(指定時は --root を無視)")
    args = parser.parse_args()

    if args.paths:
        targets = [Path(p) for p in args.paths]
    else:
        targets = find_submissions(Path(args.root))

    if not targets:
        print("no submissions to validate")
        return 0

    print(f"validating {len(targets)} submissions...")
    failures: list[tuple[Path, str]] = []
    for sub in targets:
        ok, msg = validate_one(sub, args.games)
        flag = "OK " if ok else "FAIL"
        print(f"  [{flag}] {sub} -- {msg}")
        if not ok:
            failures.append((sub, msg))
    if failures:
        print(f"\n{len(failures)} failures:")
        for sub, msg in failures:
            print(f"  - {sub}: {msg}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
