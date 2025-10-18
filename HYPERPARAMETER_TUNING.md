# ハイパーパラメータチューニングガイド

## 概要

GCNモデルのハイパーパラメータを最適化するためのツールが実装されています。
Grid SearchとRandom Searchの2つの手法をサポートしています。

## ファイル構成

```
src/gcn/tuning/
├── __init__.py
└── hyperparameter_tuner.py      # チューニング実装クラス

configs/gcn/
└── tuning_config.json           # チューニング設定

scripts/gcn/
└── tune_hyperparameters.py      # メイン実行スクリプト
```

## 基本的な使用方法

### 1. Grid Search実行

```bash
# 基本実行
python scripts/gcn/tune_hyperparameters.py --search grid

# 設定ファイル指定
python scripts/gcn/tune_hyperparameters.py --search grid --config configs/gcn/default.json
```

### 2. Random Search実行

```bash
# 基本実行（設定ファイルの試行回数で実行）
python scripts/gcn/tune_hyperparameters.py --search random

# 試行回数を指定
python scripts/gcn/tune_hyperparameters.py --search random --trials 50

# 詳細出力
python scripts/gcn/tune_hyperparameters.py --search random --trials 30
```

## チューニング設定のカスタマイズ

### tuning_config.json の編集

#### Grid Search設定例
```json
{
  "grid_search": {
    "parameters": {
      "learning_rate": [0.001, 0.01, 0.0001],
      "hidden_dim": [32, 64, 128],
      "num_layers": [5, 10, 15]
    }
  }
}
```

#### Random Search設定例
```json
{
  "random_search": {
    "n_trials": 20,
    "parameters": {
      "learning_rate": {
        "type": "float",
        "min": 0.0001,
        "max": 0.01,
        "log_scale": true
      },
      "hidden_dim": {
        "type": "choice",
        "choices": [32, 64, 96, 128, 160, 192, 256]
      }
    }
  }
}
```

## パラメータタイプ

### 1. 整数型 (int)
```json
{
  "type": "int",
  "min": 3,
  "max": 20
}
```

### 2. 浮動小数点型 (float)
```json
{
  "type": "float",
  "min": 0.1,
  "max": 0.5,
  "log_scale": false  # 対数スケールの有無
}
```

### 3. 選択型 (choice)
```json
{
  "type": "choice",
  "choices": ["mean", "sum"]
}
```

## チューニング対象パラメータ

### 高影響パラメータ（優先度高）
- `learning_rate`: 学習率
- `hidden_dim`: 隠れ層の次元
- `num_layers`: GCNレイヤー数

### 中影響パラメータ（優先度中）
- `batch_size`: バッチサイズ
- `dropout_rate`: ドロップアウト率
- `beam_size`: ビームサーチサイズ

### 低影響パラメータ（優先度低）
- `decay_rate`: 学習率減衰率
- `mlp_layers`: MLP層数
- `aggregation`: 集約方法

### RL特有のパラメータ（強化学習のみ）
- `rl_reward_type`: 報酬タイプ（"load_factor"など）
- `rl_baseline_momentum`: ベースライン更新の慣性（0.9推奨）
- `rl_entropy_weight`: エントロピー正則化の重み（0.01推奨）
- `rl_beam_search_type`: ビームサーチタイプ（"standard", "unconstrained"など）

## 結果の確認

### 結果ファイル
チューニング結果は `tuning_results/` ディレクトリに保存されます：

```
tuning_results/
├── grid_search_results_20241218_143022.json    # 詳細結果
├── grid_search_summary_20241218_143022.txt     # サマリーレポート
├── intermediate_results_20241218_143022.csv    # 中間結果CSV
└── best_config_20241218_143022.json           # 最適設定
```

### ベスト設定での実行
```bash
# 最適化された設定で実行
python scripts/gcn/train_gcn.py --config tuning_results/best_config_20241218_143022.json
```

## 評価指標

### Supervised Learning（教師あり学習）
- **主指標**: Test Approximation Rate（テスト近似率）
- **副指標**:
  - Val Approximation Rate（検証近似率）
  - Test/Val Infeasible Rate（実行不可能率）
  - Mean Load Factor（平均負荷率）

### Reinforcement Learning（強化学習）
- **主指標**: Test Mean Load Factor（テスト平均負荷率）- 最小化
- **副指標**:
  - Val Mean Load Factor（検証平均負荷率）
  - Test/Val Infeasible Rate（実行不可能率）

## 実行前の準備

1. **データ生成**（必要に応じて）
```bash
python scripts/common/generate_data.py
```

2. **設定ファイル確認**
- `configs/gcn/default.json` - ベース設定
- `configs/gcn/tuning_config.json` - チューニング設定

## 注意事項

- チューニングには時間がかかります（Grid Searchは特に）
- GPU使用時はメモリ使用量に注意
- `Ctrl+C`で中断可能（部分結果は保存されます）
- Random Searchの方が効率的な場合が多い

## 推奨ワークフロー

1. **小規模テスト**: 少数パラメータでGrid Search
2. **本格チューニング**: Random Searchで広範囲探索
3. **ファインチューニング**: ベスト近辺でGrid Search

## トラブルシューティング

### メモリ不足
- `batch_size`を小さくする
- `hidden_dim`を小さくする
- `num_layers`を少なくする

### 実行時間が長い
- Random Searchの`n_trials`を減らす
- Grid Searchのパラメータ数を減らす
- `max_epochs`を小さくする