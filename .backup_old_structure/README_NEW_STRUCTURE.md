# Graph Convolutional Network for UELB - 新構造ガイド

## 概要

このプロジェクトは **GCN (Graph Convolutional Network)** と **RL-KSP (Reinforcement Learning with K-Shortest Paths)** の2つの手法を実装しています。

リファクタリングにより、両手法が明確に分離され、それぞれ独立して実験できるようになりました。

---

## 新しいディレクトリ構造

```
graph-convnet-uelb/
│
├── src/
│   ├── common/              # 共通モジュール（両手法で共有）
│   │   ├── data_management/ # データ読み込み・生成
│   │   ├── graph/           # グラフ処理
│   │   ├── config/          # 設定管理
│   │   ├── visualization/   # 可視化
│   │   └── utils/           # 共通ユーティリティ
│   │
│   ├── gcn/                 # GCN手法専用
│   │   ├── models/          # GCNモデル
│   │   ├── algorithms/      # ビームサーチ
│   │   ├── train/           # トレーナー・評価器
│   │   └── tuning/          # ハイパーパラメータ調整
│   │
│   └── rl_ksp/              # RL-KSP手法専用
│       ├── environment/     # RL環境
│       ├── models/          # DQNモデル
│       ├── train/           # RLトレーナー
│       └── tuning/          # ハイパーパラメータ調整
│
├── configs/
│   ├── gcn/                 # GCN用設定ファイル
│   └── rl_ksp/              # RL-KSP用設定ファイル
│
├── scripts/
│   ├── gcn/                 # GCN実行スクリプト
│   ├── rl_ksp/              # RL-KSP実行スクリプト
│   ├── common/              # 共通スクリプト
│   └── comparison/          # 比較実験用（準備中）
│
├── data/                    # データ（共通）
├── saved_models/            # 保存モデル
├── results/                 # 実験結果
└── logs/                    # ログ
```

---

## 実行方法

### 1. データ生成（初回のみ）

```bash
python scripts/common/generate_data.py --config configs/gcn/default.json
```

### 2. GCN実験

#### 学習+テスト（一括実行）
```bash
python scripts/gcn/train_gcn.py --config configs/gcn/default.json
```

#### その他のGCN設定ファイル
```bash
# NSFNetトポロジー、10品種
python scripts/gcn/train_gcn.py --config configs/gcn/nsfnet_10_commodities.json

# NSFNetトポロジー、20品種
python scripts/gcn/train_gcn.py --config configs/gcn/nsfnet_20_commodities.json

# 保存済みモデルを読み込んで評価のみ
python scripts/gcn/train_gcn.py --config configs/gcn/load_saved_model.json
```

### 3. RL-KSP実験

#### 学習+テスト（一括実行）
```bash
python scripts/rl_ksp/train_rl_ksp.py --config configs/rl_ksp/rl_config.json
```

#### その他のRL-KSP設定ファイル
```bash
# NSFNetトポロジー、10品種
python scripts/rl_ksp/train_rl_ksp.py --config configs/rl_ksp/nsfnet_10_commodities_rl.json

# NSFNetトポロジー、20品種
python scripts/rl_ksp/train_rl_ksp.py --config configs/rl_ksp/nsfnet_20_commodities_rl.json
```

### 4. 結果の集計

```bash
python scripts/common/calculate_averages.py
```

---

## 旧コマンドとの対応

| 旧コマンド | 新コマンド |
|-----------|----------|
| `python main.py --mode gcn --config configs/default.json` | `python scripts/gcn/train_gcn.py --config configs/gcn/default.json` |
| `python main.py --mode rl --config configs/rl_config.json` | `python scripts/rl_ksp/train_rl_ksp.py --config configs/rl_ksp/rl_config.json` |
| `python generate_data.py --config configs/default.json` | `python scripts/common/generate_data.py --config configs/gcn/default.json` |
| `python tune_hyperparameters.py --config configs/tuning_config.json` | `python scripts/gcn/tune_gcn.py --config configs/gcn/tuning_config.json` |

---

## 各手法の実行フロー

### GCN
```
1. 設定ファイル読み込み
2. データセット読み込み
3. GCNモデル初期化
4. 学習実行（複数エポック）
   ├─ 各エポックで検証実行
   └─ ベストモデル保存
5. テスト実行（最終評価）
6. 結果保存（ログ、メトリクス）
```

### RL-KSP
```
1. 設定ファイル読み込み
2. データセット読み込み
3. RL環境・DQNモデル初期化
4. 学習実行（複数エピソード）
   └─ ε-greedy方策で経路探索
5. テスト実行（学習済みモデルで評価）
6. 結果保存（ログ、メトリクス）
```

**重要**: 両手法とも「学習 → テスト → 結果保存」を一括実行します。

---

## 設定ファイル

### GCN用設定ファイル（configs/gcn/）
- `default.json` - デフォルト設定
- `default2.json` - 代替デフォルト設定
- `nsfnet_10_commodities.json` - NSFNet 10品種
- `nsfnet_15_commodities.json` - NSFNet 15品種
- `nsfnet_20_commodities.json` - NSFNet 20品種
- `nsfnet_25_commodities.json` - NSFNet 25品種
- `tuning_config.json` - ハイパーパラメータ調整用
- `load_saved_model.json` - 保存済みモデル読み込み
- `with_model_saving.json` - モデル保存設定

### RL-KSP用設定ファイル（configs/rl_ksp/）
- `rl_config.json` - RLデフォルト設定
- `nsfnet_5_commodities_rl.json` - NSFNet 5品種（RL）
- `nsfnet_10_commodities_rl.json` - NSFNet 10品種（RL）
- `nsfnet_15_commodities_rl.json` - NSFNet 15品種（RL）
- `nsfnet_20_commodities_rl.json` - NSFNet 20品種（RL）
- `rl_load_saved_model.json` - RL保存済みモデル読み込み
- `rl_with_model_saving.json` - RLモデル保存設定

---

## 主要なモジュール

### GCN手法（src/gcn/）

#### モデル
- `models/gcn_model.py` - ResidualGatedGCNModel
- `models/gcn_layers.py` - GCNレイヤー
- `models/model_utils.py` - モデルユーティリティ

#### アルゴリズム
- `algorithms/beamsearch.py` - ビームサーチ
- `algorithms/beamsearch_uelb.py` - UELB用ビームサーチ
- `algorithms/beamsearch_comparator.py` - ビームサーチ比較

#### トレーニング
- `train/trainer.py` - GCNトレーナー
- `train/evaluator.py` - GCN評価器
- `train/metrics.py` - メトリクス記録

### RL-KSP手法（src/rl_ksp/）

#### 環境
- `environment/rl_environment.py` - MinMaxLoadKSPsEnv

#### モデル
- `models/dqn_model.py` - DQNModel

#### トレーニング
- `train/rl_trainer.py` - RLトレーナー

### 共通モジュール（src/common/）

#### データ管理
- `data_management/dataset_reader.py` - データセット読み込み
- `data_management/data_maker.py` - データ生成
- `data_management/exact_solution.py` - 厳密解計算

#### グラフ処理
- `graph/graph_making.py` - グラフ生成
- `graph/graph_utils.py` - グラフユーティリティ
- `graph/k_shortest_path.py` - K最短経路探索
- `graph/flow.py` - フロー計算

#### 設定
- `config/config.py` - 設定クラス
- `config/config_manager.py` - 設定管理

---

## トラブルシューティング

### ImportError が発生する場合

プロジェクトルートから実行していることを確認してください：

```bash
# 正しい
cd /path/to/graph-convnet-uelb
python scripts/gcn/train_gcn.py --config configs/gcn/default.json

# 間違い
cd /path/to/graph-convnet-uelb/scripts/gcn
python train_gcn.py --config ../../configs/gcn/default.json
```

### データが見つからない場合

データを生成してください：

```bash
python scripts/common/generate_data.py --config configs/gcn/default.json
```

### 旧ファイルについて

リファクタリング後も、`src/models/`, `src/train/`, `src/algorithms/` などの旧ディレクトリが残っています。

これらは**互換性のため残されていますが、新しいコードでは使用しないでください**。

---

## 開発者向け情報

### 新しいモジュールを追加する場合

#### GCN用モジュール
`src/gcn/` 以下に追加してください。

#### RL-KSP用モジュール
`src/rl_ksp/` 以下に追加してください。

#### 共通モジュール
両手法で使用する場合は `src/common/` に追加してください。

### インポートの書き方

```python
# GCNモジュール内
from src.gcn.models.gcn_model import ResidualGatedGCNModel
from src.common.data_management.dataset_reader import DatasetReader

# RL-KSPモジュール内
from src.rl_ksp.models.dqn_model import DQNModel
from src.common.graph.k_shortest_path import KShortestPathFinder

# 共通モジュール内
from src.common.config.config import get_config
from src.common.graph.graph_utils import *
```

**重要**: 相対インポート（`from ..module import`）は使用しないでください。
すべて絶対インポート（`from src.xxx import`）を使用してください。

---

## リファクタリング履歴

詳細は以下のドキュメントを参照してください：

- `RESTRUCTURE_PLAN.md` - リファクタリング計画
- `RESTRUCTURE_STATUS.md` - 進捗状況
- `RESTRUCTURE_LOG.md` - 実行ログ

---

## ライセンス

（既存のライセンス情報をここに記載）

---

## お問い合わせ

（既存の連絡先情報をここに記載）
