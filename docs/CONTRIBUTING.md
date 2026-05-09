# CONTRIBUTING

モデル提出のルールと、新環境を追加する場合の手順。

## モデル提出

### 1. ディレクトリを作る

```
submissions/<env>/<author>_<name>/
    ├── meta.json         # 必須
    └── model.csv | model.onnx | model.pmml   # type に応じてどれか
```

`<author>_<name>` は半角英数字とアンダースコアで。半角スペースや全角文字は避ける(CI 側でディレクトリ名を扱うため)。

### 2. `meta.json` を書く

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

### 3. ファイルを書き出す

#### CSV (テーブル系)

ヘッダ固定、long 形式:

```
state_key,action,value
```

- `state_key` は `envs/<env>/spec.md` で定義された canonical 文字列。
- `value` は Q値 / 期待勝率 / その他「大きいほど良い」スカラー値。
- 違法行動の行があっても無視されるが、書かないほうが軽い。
- 未学習の状態は対戦時に自動でランダム fallback。

#### ONNX (DNN系)

- 入力: `(1, *obs_shape)` の `float32`(各環境の `spec.md` 参照)
- 出力: `(1, n_actions)` の Q値 / logits
- AlphaZero スタイルで policy + value を出す場合は、policy 出力のインデックスを `meta.json.policy_index` で指定
- opset 13 以降推奨

PyTorch から書き出す例:

```python
import torch
torch.onnx.export(net, dummy_input, "model.onnx",
                  input_names=["obs"], output_names=["q"], opset_version=17)
```

#### PMML (任意)

- `pypmml` で読み込める形式
- 特徴量はフラット化した観測(`obs.array.flatten()`)で `f0, f1, ...` を期待
- 出力は `probability(<action>)` または `predicted_<target>` を期待

### 4. ローカルで検証する

```bash
# 自分のモデルが random と当たって違法手を出さないか
python scripts/validate_submission.py --paths submissions/tictactoe/yutaro_qtable

# 既存の例とも対戦
python -m arena.cli play --env tictactoe \
    --p1 submissions/tictactoe/yutaro_qtable \
    --p2 tictactoe:examples/minimax --games 20
```

### 5. PR を出す

- タイトル例: `Add tictactoe/yutaro_qtable`
- CI(`.github/workflows/validate_submission.yml`)が走り、対 random スモークテスト + 全提出物総当たり戦が回る。

## 新しい環境を追加する

1. `envs/<env_name>/env.py` を `envs.base.Env` 抽象クラスに準拠して実装。
2. `envs/registry.py` に `register("<env_name>", <EnvClass>)` を追加。
3. `envs/<env_name>/spec.md` を書く(観測・行動・モデルの入出力契約)。
4. `submissions/<env_name>/` を作って `.gitkeep` を置く。
5. `tests/test_envs.py` の共通テストが通ることを確認(全環境共通テストは自動で適用される)。

## コーディング規約

- Python 3.10+
- 依存はランタイム側を最小化(`numpy + onnxruntime` のみで動くこと)。学習用フレームワーク(PyTorch / SB3)はランタイム必須にしない。
- 公開関数は型ヒントを付ける。
- 不要なコメントは付けない。WHY が非自明なものだけ。
