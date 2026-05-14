# 提出方法

### 1. 自分のブランチを作成

通常通り、自分の作業用ブランチを切る。
```
git checkout -b <your-branch-name>
```

### 2. ディレクトリを作成

`submissions/<env>/<author>_<name>/` というパスでディレクトリを作る。

- `<env>` は `tictactoe` / `othello` / `connect4` のいずれか
- `<author>_<name>` は半角英数字とアンダースコアのみ

ディレクトリ構成のイメージ:

```
submissions/
├── tictactoe/
│   ├── alice_dqn/
│   │   ├── model.onnx
│   │   └── meta.json
│   └── bob_qtable/
│       ├── model.csv
│       └── meta.json
└── othello/
    └── carol_ppo/
        ├── model.onnx
        └── meta.json
```

### 3. モデルファイルを置く

作成したディレクトリに、モデル本体のファイルを置く。形式は CSV か ONNX。

#### CSV (テーブル系)
```
state_key,action,value
```

- `state_key` は `envs/<env>/spec.md` で定義された文字列
- `value` は Q値
- 未知の状態は、対戦時に自動でランダムな行動が選択される

#### ONNX (DNN系)

- 入力: `(1, *obs_shape)` の `float32`
- 出力: `(1, n_actions)` の Q値
- opset 13 以降推奨

PyTorch から書き出す例:

```python
import torch
torch.onnx.export(net, dummy_input, "model.onnx", input_names=["obs"], output_names=["q"], opset_version=17)
```

#### PMML (任意)

- `pypmml` で読み込める形式
- 特徴量はフラット化した観測(`obs.array.flatten()`)で `f0, f1, ...` を期待
- 出力は `probability(<action>)` または `predicted_<target>` を期待

### 4. meta.json を置く

同じディレクトリに `meta.json` を置く。例:

```json
{
  "env": "tictactoe",
  "type": "csv",
  "file": "model.csv",
  "author": "yutaro",
  "name": "yutaro_qtable",
  "description": "200k self-play, alpha=0.1, gamma=0.95"
}
```

各フィールドの意味:

| フィールド | 必須 | 内容 |
|---|---|---|
| `env` | ✅ | `tictactoe` / `othello` / `connect4` のいずれか |
| `type` | ✅ | `csv` / `onnx` / `pmml` / `random` |
| `file` | type が random 以外で必須 | モデルファイル名(ディレクトリ相対) |
| `author` | ✅ | 提出者の handle |
| `name` | ✅ | モデル名(リーダーボード表記) |
| `description` | 任意 | 学習手法・ハイパラなど |
| `policy_index` | ONNX 任意 | 出力が複数ある場合の policy インデックス(default 0) |
| `input_name` | ONNX 任意 | 入力名(未指定なら最初の入力を使う) |
| `feature_names` | PMML 任意 | 特徴量の名前一覧 |

### 5. ローカルで検証する

push する前に、自分の環境で動作確認しておく。

```bash
# 自分のモデルが random と当たって違法手を出さないか
python scripts/validate_submission.py --paths submissions/tictactoe/yutaro_qtable

# 既存の例とも対戦
python -m arena.cli play --env tictactoe \
    --p1 submissions/tictactoe/yutaro_qtable \
    --p2 tictactoe:examples/minimax --games 20
```

### 6. リポジトリへ push して PR を出す

提出物をステージして commit する。

```bash
git add Path/to/file
git commit -m "Add <env>/<author>_<name>"
```

自分のブランチを push する。

```bash
git push origin <your-branch-name>
```

GitHub 上で PR を作成する。

PR を出すと CI(`.github/workflows/validate_submission.yml`)が自動で走り、対 random スモークテストと全提出物総当たり戦が実行される。
全チェックが通れば管理者がマージする。
