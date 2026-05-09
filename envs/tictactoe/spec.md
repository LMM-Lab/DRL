# Tic-tac-toe (三目並べ) 仕様書

提出物が満たすべき入出力契約。

## ゲームルール

- 3×3 のマス目に交互に石を置き、縦・横・斜めいずれかに3つ並べた方の勝ち。
- 全マス埋まれば引き分け。
- 先手が `player 0`(`X`)、後手が `player 1`(`O`)。

## 行動空間 (action space)

- 整数 `0..8`(`row * 3 + col`)。
  ```
  0 1 2
  3 4 5
  6 7 8
  ```
- `legal_actions` に含まれない値を返した場合は **即敗北** とみなす(運営側でチェック)。

## 観測 (observation)

ONNX/コードベースで使う `array` と、CSVで使う `state_key` の両方が渡される。

- `array`: shape `(2, 3, 3)` の `float32`、canonical(現プレイヤー視点)。
  - チャンネル0: 自分の石 = 1.0、その他 = 0.0
  - チャンネル1: 相手の石 = 1.0、その他 = 0.0
- `state_key`: 9文字の文字列。`X` = 自分、`O` = 相手、`.` = 空。
  - 例: `X..O.X..O`

## モデル種別ごとの入出力

### CSV (Q-table)
- ヘッダ: `state_key,action,value`
- 同じ `state_key` の行から、`action ∈ legal_actions` の中で `value` が最大の行動を選ぶ。
- 未学習の `state_key` は legal な手からランダム fallback。

### ONNX
- 入力: `(1, 2, 3, 3)` float32(バッチ次元1必須)。
- 出力: `(1, 9)` float32 の Q値 / logits。`legal_actions` でマスク後 argmax。
- ランタイムは `onnxruntime` で読み込み可能であること(opset 13 以降推奨)。

### PMML (任意)
- `arena.policies.pmml_policy` 参照。9次元のフラット特徴量を入力に取る分類器を期待。

## meta.json
```json
{
  "env": "tictactoe",
  "type": "csv | onnx | pmml | random",
  "file": "model.csv",
  "author": "alice",
  "name": "alice_qtable",
  "description": "tabular Q-learning, 200k self-play episodes"
}
```
