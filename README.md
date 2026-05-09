# rl-arena

強化学習コミュニティの **モデル対戦リポジトリ**。

メンバーは好きなゲームでモデルを学習し、リポジトリにプッシュ。プッシュされたモデル同士を対戦させたり、自分が指定したモデルと **人間 vs モデル** で遊んだりできる。

- **テーブル系モデル → CSV**
- **DNN 系モデル → ONNX**
- 任意で **PMML** にも対応(分類器系)
- 将来的に AlphaZero(MCTS + NN)を組み込めるように、`Env.clone()` などを最初から備えている

## ディレクトリ構成

```
rl-arena/
├── arena/                      # 共通ランタイム(コア)
│   ├── runner.py               # 対戦実行ロジック
│   ├── policy_loader.py        # モデル読み込み(csv/onnx/pmml対応)
│   ├── policies/               # base / csv / onnx / pmml / random / human
│   └── cli.py                  # コマンドライン入口
├── envs/                       # 対戦環境(運営が用意)
│   ├── base.py
│   ├── tictactoe/  (env.py, spec.md, examples/)
│   ├── othello/
│   └── connect4/
├── submissions/                # メンバーの提出物
│   ├── tictactoe/<author>_<name>/
│   ├── othello/...
│   └── connect4/...
├── training/                   # 学習サンプル(任意)
├── tests/
└── .github/workflows/          # PR時に自動検証
```

## クイックスタート

```bash
# 1. 依存をインストール(ランタイムだけなら numpy + onnxruntime で十分)
pip install -e .

# 2. 環境一覧を確認
python -m arena.cli list-envs

# 3. ランダム同士で1試合
python -m arena.cli play --env tictactoe --p1 random --p2 random --render

# 4. 例(完全解析した三目並べ)を作って戦う
python training/tictactoe_minimax.py
python -m arena.cli play --env tictactoe --p1 random \
    --p2 tictactoe:examples/minimax --games 10

# 5. 人間 vs モデル
python -m arena.cli human --env tictactoe \
    --opponent tictactoe:examples/minimax

# 6. 総当たり戦
python -m arena.cli tournament --env tictactoe \
    --dir submissions/tictactoe --games 4 --include-random
```

## モデルを提出するには

詳細は [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) と各環境の `envs/<env>/spec.md`。

最短手順:

1. 好きな環境で学習する(`training/` のサンプル参照)。
2. 学習結果を **CSV か ONNX** で書き出す。
3. `submissions/<env>/<your-handle>_<model-name>/` を作って、`model.csv` または `model.onnx` と `meta.json` を置く。
4. PR を出す。CI が自動で「対 random で違法手や例外を出さないか」を検証する。

`meta.json` の最小例:

```json
{
  "env": "tictactoe",
  "type": "csv",
  "file": "model.csv",
  "author": "yutaro",
  "name": "yutaro_qtable",
  "description": "Q-learning, 200k self-play"
}
```

## 用意されている環境

| 環境 | 観測 shape | 行動数 | 備考 |
|---|---|---|---|
| `tictactoe` | (2, 3, 3) | 9 | 三目並べ |
| `othello`   | (2, 8, 8) | 65 | 8x8、64 = パス |
| `connect4`  | (2, 6, 7) | 7 | 列を選んで落とす |

新環境を追加するときは `envs/<name>/env.py` を `Env` 抽象クラスに準拠して書き、`envs/registry.py` に登録 + `spec.md` を添える。

## 開発

```bash
pip install -e ".[dev]"
pytest -q
```

## ライセンス

MIT(予定)。
