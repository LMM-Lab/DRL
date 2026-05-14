# 概要

好きなゲームでモデルを学習し、リポジトリにプッシュ。プッシュされたモデル同士を対戦させたり、自分が指定したモデルと戦ったりできる。

## 用意されている環境

| 環境 | 観測 shape | 行動数 | 備考 |
|---|---|---|---|
| `tictactoe` | (2, 3, 3) | 9 | 三目並べ |
| `othello`   | (2, 8, 8) | 65 | 8x8、64 = パス |
| `connect4`  | (2, 6, 7) | 7 | 列を選んで落とす |

## ディレクトリ構成

```
rl-arena/
├── arena/                      # 共通ランタイム(コア)
│   ├── runner.py               # 対戦実行ロジック
│   ├── policy_loader.py        # モデル読み込み(csv/onnx/pmml対応)
│   ├── policies/               # base / csv / onnx / pmml / random / human
│   └── cli.py                  # コマンドライン入口
├── docs/                       # ドキュメント
│   ├── submission_guide.md     # 提出手順
│   └── usage.md                # 使い方
├── envs/                       # 対戦環境(運営が用意)
│   ├── base.py
│   ├── registry.py
│   ├── tictactoe/  (env.py, spec.md, examples/)
│   ├── othello/    (env.py, spec.md, examples/)
│   └── connect4/   (env.py, spec.md, examples/)
├── scripts/
│   └── validate_submission.py  # 提出物バリデーション
├── submissions/                # メンバーの提出物
│   ├── tictactoe/
│   │   ├── yutaro_dqn/         # (meta.json, model.onnx)
│   │   └── yutaro_qtable/      # (meta.json, model.csv)
│   ├── othello/
│   └── connect4/
├── training/                   # 学習サンプル(任意)
│   ├── README.md
│   ├── dqn_to_onnx.py
│   ├── tictactoe_minimax.py
│   └── tictactoe_qlearning.py
└── .github/workflows/          # PR時に自動検証
```

## クイックスタート

```bash
# 1. 依存をインストール(ランタイムだけなら numpy + onnxruntime で十分)
python -m venv vnev
source vnev/bin/activate
#仮想環境に入れたのを確認してからライブラリをインストール
pip install -r requirements.txt

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

# 7. 指定のモデルの対戦
python -m arena.cli play `
--env tictactoe `
--p1 submissions/path/to/folder `
--p2 submissions/path/to/folder `
--games 10 `
--swap `
--render

#サンプル学習

# 1. 学習 
python training/tictactoe_qlearning.py `
--episodes 200000 `
--out submissions/tictactoe/自分の名前_qtable `
--author 自分の名前
```

## モデルを提出するには

詳細は [docs/submission_guide.md](docs/submission_guide.md) と各環境の `envs/<env>/spec.md`。

最短手順:

1. 好きな環境で学習する(`training/` のサンプル参照)。
2. 学習結果を **CSV か ONNX** で書き出す。
3. `submissions/<env>/<your-handle>_<model-name>/` を作って、`model.csv` または `model.onnx` と `meta.json` を置く。
4. リポジトリにプッシュする。
