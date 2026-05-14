# 使い方

## セットアップ

```bash
# 仮想環境を作成して有効化
python -m venv .venv
# Windows
.venv\Scripts\Activate.ps1
# Mac/Linux
source .venv/bin/activate

# 依存パッケージをインストール
pip install -r requirements.txt
```

---

## モデル同士を対戦させる

```bash
python -m arena.cli play --env <env> --p1 <モデル1> --p2 <モデル2> [オプション]
```

| オプション | 必須 | デフォルト | 内容 |
|---|---|---|---|
| `--env` | ✅ | — | 環境名(`tictactoe` / `othello` / `connect4`) |
| `--p1` | ✅ | — | プレイヤー1のモデル(提出物ディレクトリのパス、または `random`) |
| `--p2` | ✅ | — | プレイヤー2のモデル(提出物ディレクトリのパス、または `random`) |
| `--games` | | `1` | 対戦試合数 |
| `--swap` | | `false` | 奇数試合目で先手後手を入れ替える |
| `--render` | | `false` | 各手の後に盤面を表示する |
| `--log <path>` | | なし | 試合結果を JSONL 形式でファイルに保存 |

---

## 指定のモデルと戦う

```bash
python -m arena.cli human --env <env> --opponent <モデル> [オプション]
```

| オプション | 必須 | デフォルト | 内容 |
|---|---|---|---|
| `--env` | ✅ | — | 環境名 |
| `--opponent` | ✅ | — | 対戦相手のモデル(提出物ディレクトリのパス、または `random`) |
| `--first` | | `me` | `me` = 自分が先手 / `ai` = AI が先手 |

手の入力方法は `行,列` 形式(例: `1,2`)または行動番号(例: `5`)。

---

## 総当たり戦

```bash
python -m arena.cli tournament --env <env> --dir <ディレクトリ> [オプション]
```

| オプション | 必須 | デフォルト | 内容 |
|---|---|---|---|
| `--env` | ✅ | — | 環境名 |
| `--dir` | ✅ | — | 提出物が並ぶ親ディレクトリ(例: `submissions/tictactoe`) |
| `--games` | | `2` | 1ペアあたりの試合数 |
| `--include-random` | | `false` | ランダムエージェントもエントリーに加える |
| `--out <path>` | | なし | トーナメント結果を JSON ファイルに保存 |

---

## モデルを学習させる

### Q-learning 

```bash
python training/tictactoe_qlearning.py --out <出力ディレクトリ>
```

| オプション | デフォルト | 内容 |
|---|---|---|
| `--episodes` | `100000` | 学習エピソード数 |
| `--alpha` | `0.1` | 学習率 |
| `--gamma` | `0.95` | 割引率 |
| `--eps-start` | `1.0` | ε-greedy の初期探索率 |
| `--eps-end` | `0.05` | ε-greedy の最終探索率 |
| `--seed` | `0` | 乱数シード |
| `--out` | 必須 | 出力ディレクトリ(model.csv と meta.json が生成される) |
| `--author` | `anonymous` | meta.json に記録する提出者名 |
| `--name` | ディレクトリ名 | meta.json に記録するモデル名 |

### DQN

```bash
python training/dqn_to_onnx.py --env <env> --out <出力ディレクトリ>
```

| オプション | デフォルト | 内容 |
|---|---|---|
| `--env` | 必須 | 環境名(`tictactoe` / `connect4` / `othello`) |
| `--steps` | `100000` | 学習ステップ数 |
| `--batch` | `64` | バッチサイズ |
| `--gamma` | `0.95` | 割引率 |
| `--lr` | `0.001` | 学習率 |
| `--seed` | `0` | 乱数シード |
| `--out` | 必須 | 出力ディレクトリ(model.onnx と meta.json が生成される) |
| `--author` | `anonymous` | meta.json に記録する提出者名 |
| `--name` | ディレクトリ名 | meta.json に記録するモデル名 |

### minimax (三目並べのみ、完全解析)

```bash
python training/tictactoe_minimax.py
```

オプションなし。`envs/tictactoe/examples/minimax/` に model.csv を生成する。

---

## 提出前の検証

```bash
python scripts/validate_submission.py
```

| オプション | デフォルト | 内容 |
|---|---|---|
| `--root` | `submissions` | 検証対象の親ディレクトリ(全提出物を一括検証) |
| `--games` | `4` | 1モデルあたり対 random で走らせる試合数 |
| `--paths` | なし | 個別ディレクトリを指定(指定時は `--root` を無視) |

提出方法の詳細は [docs/submission_guide.md](submission_guide.md) を参照。
