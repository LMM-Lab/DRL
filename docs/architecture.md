# アーキテクチャ

## レイヤー

```
┌───────────────────────────────────────────────────┐
│  CLI  (arena/cli.py)                              │
├───────────────────────────────────────────────────┤
│  Runner (arena/runner.py)  - run_match / tournament │
├───────────────┬───────────────────────────────────┤
│  Policy 層    │  Env 層                          │
│  (arena/...)  │  (envs/...)                       │
│  Random       │  TicTacToe / Othello / Connect4   │
│  Csv          │                                   │
│  Onnx         │                                   │
│  Pmml         │                                   │
│  Human        │                                   │
└───────────────┴───────────────────────────────────┘
```

- **Env** は 2人ゼロ和・完全情報・交互手番の API に統一。`Observation` には ONNX 用の `array` と CSV/PMML 用の `state_key` を両方含める。
- **Policy** は `act(observation) -> int` だけが責務。Env を直接知らない(`HumanPolicy` のみ表示・パースのため Env を保持)。
- **Runner** は両方を組み合わせて 1 ゲーム進行する。違法行動 / Policy 例外 = **即敗北**。
- **policy_loader** が `meta.json` を読んで適切な Policy を返す。

## 状態キー(state_key)

テーブル系モデルが学習データを共有しやすいように、各環境で canonical な短い文字列で状態を表す:

- `X` = 自分(現プレイヤー)の石
- `O` = 相手の石
- `.` = 空

`current_player` の符号は state_key に含めない(canonical 化済み)。プレイヤー入れ替えで同じ state_key が再利用できるため、自己対戦による学習データが両プレイヤーで共有される。

## AlphaZero への拡張ポイント

すでに以下が用意されている:

- `Env.clone()` … MCTS の探索木展開で使う。
- `Observation.array` … policy network の入力として直接使える。
- `OnnxPolicy.policy_index` … policy/value の 2 出力モデルの policy 側を指定可能。

追加で必要なのは:

- MCTS 本体(`arena/policies/mcts_policy.py` などとして実装)。
- `meta.json` に `type: "alphazero"` を加え、`policy_loader` で MCTS + ONNX を組み立てる。
- 学習側(自己対戦 + NN 訓練ループ)は `training/` に追加。

## 拡張アイデア

- リーダーボード(`leaderboard.json` を CI で更新)
- 棋譜のリプレイ可視化(matplotlib / Web)
- タイムアウト負け(推論時間に制限)
- 観戦用 Web UI(Flask / Streamlit)
