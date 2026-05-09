# training/

提出物を作るためのサンプル学習スクリプト。任意のフレームワークで自由に学習し、最終的に CSV / ONNX として書き出すのが狙い。

## 含まれるもの

| script | 環境 | 出力 | 必要パッケージ |
|---|---|---|---|
| `tictactoe_minimax.py` | tictactoe | CSV | `numpy` |
| `tictactoe_qlearning.py` | tictactoe | CSV | `numpy` |
| `dqn_to_onnx.py` | tictactoe / connect4 | ONNX | `torch` |

## 使い方の例

```bash
# 三目並べを完全解析して examples/minimax/model.csv を生成
python training/tictactoe_minimax.py

# Q-learning を 100k 自己対戦で回して submissions/tictactoe/yutaro_qtable/ を作成
python training/tictactoe_qlearning.py --episodes 100000 --out submissions/tictactoe/yutaro_qtable

# DQN を学習して ONNX 出力(connect4)
python training/dqn_to_onnx.py --env connect4 --steps 200000 --out submissions/connect4/yutaro_dqn
```

## 自前で書く場合の最低条件

1. 環境を `envs.make("<env>")` で取得して学習(`envs/<env>/spec.md` の入出力契約を守る)。
2. 学習済みモデルを CSV か ONNX に書き出す。
3. `meta.json` を一緒に置く。
4. `python -m arena.cli play --env <env> --p1 random --p2 <your-dir>` がエラー無く動くことを確認。
