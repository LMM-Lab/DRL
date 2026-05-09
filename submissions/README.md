# submissions/

メンバーが提出した学習済みモデルが置かれる場所。

## 提出方法

1. このリポジトリをフォーク → 自分のブランチで作業。
2. `submissions/<env>/<author>_<name>/` ディレクトリを作成。
3. ディレクトリに以下を置く:
   - `meta.json`(必須)
   - `model.csv` または `model.onnx` または `model.pmml`(`type` に応じて)
4. PR を出す。CI で自動検証 → マージ。

詳細は [docs/CONTRIBUTING.md](../docs/CONTRIBUTING.md)、各環境の `envs/<env>/spec.md` を参照。

## ディレクトリ例

```
submissions/
├── tictactoe/
│   ├── alice_dqn/        ← 各人の作品
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
