# Graph ConvNet UELB - Training Pipeline

このリポジトリは、**GCN (Graph Convolutional Network)** と **RL-KSP (Reinforcement Learning with K-Shortest Paths)** の2つの手法を用いたUELB問題の学習・評価パイプラインを提供します。

## 📋 目次
- [必要条件](#必要条件)
- [セットアップ](#セットアップ)
- [プロジェクト構造](#プロジェクト構造)
- [使い方](#使い方)
  - [データ生成](#1-データ生成)
  - [GCN実験](#2-gcn実験)
  - [RL-KSP実験](#3-rl-ksp実験)
- [設定ファイル](#設定ファイル)
- [結果の確認](#結果の確認)
- [トラブルシューティング](#トラブルシューティング)

---

## 必要条件

- **Python 3.9以降**
- **CUDA対応GPU**（推奨、CPUでも動作可能）
- **Conda**（MinicondaまたはAnaconda）

---

## セットアップ

### 1. Conda環境の作成とアクティベート

```bash
# 環境の作成
conda env create -f environment.yml

# 環境のアクティベート
conda activate gcn-uelb-env
```

### 2. 環境の確認

```bash
# 環境が正しく作成されたか確認
conda list
```

### 3. 設定ファイルの編集

設定ファイル（例: `configs/gcn/default2.json`）を編集して、学習条件やデータパスを調整します。

---

## プロジェクト構造

```
graph-convnet-uelb/
│
├── src/
│   ├── common/              # 共通モジュール（両手法で共有）
│   │   ├── data_management/ # データ読み込み・生成
│   │   ├── graph/           # グラフ処理
│   │   ├── config/          # 設定管理
│   │   └── visualization/   # 可視化
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
│   └── common/              # 共通スクリプト
│
├── data/                    # データセット（生成後に作成）
├── saved_models/            # 保存モデル
├── results/                 # 実験結果
└── logs/                    # ログファイル
```

---

## 使い方

### 1. データ生成

まず、学習・評価用のデータセットを生成します：

```bash
# GCN用データ生成
python scripts/common/generate_data.py --config configs/gcn/default2.json

# SeqFlowRL用データ生成
python scripts/common/generate_data.py --config configs/seqflowrl/seqflowrl_base.json

# 特定の設定ファイルでデータ生成
python scripts/common/generate_data.py --config configs/gcn/nsfnet_15_commodities.json

# 特定のモードのみ生成
python scripts/common/generate_data.py --config configs/gcn/nsfnet_15_commodities.json --modes train val

# 確認なしで強制実行
python scripts/common/generate_data.py --config configs/gcn/nsfnet_15_commodities.json --force

# データのみ削除（再生成しない）
python scripts/common/generate_data.py --clean-only
```

**実行後**、以下のディレクトリが作成されます：
- `data/train_data/` - 学習データ
- `data/val_data/` - 検証データ
- `data/test_data/` - テストデータ

---

### 2. GCN実験

#### 基本的な学習+テスト（一括実行）

```bash
# デフォルト設定で実行
python scripts/gcn/train_gcn.py --config configs/gcn/default2.json

# NSFNetトポロジー、10品種
python scripts/gcn/train_gcn.py --config configs/gcn/nsfnet_10_commodities.json

# NSFNetトポロジー、20品種
python scripts/gcn/train_gcn.py --config configs/gcn/nsfnet_20_commodities.json
```

#### 保存済みモデルを読み込んで評価のみ

```bash
python scripts/gcn/train_gcn.py --config configs/gcn/load_saved_model.json
```

#### 実行フロー
```
1. 設定ファイル読み込み
2. データセット読み込み
3. GCNモデル初期化
4. 学習実行（複数エポック）
   ├─ 各エポックで検証実行
   └─ ベストモデル保存
5. テスト実行（最終評価）
6. 結果保存（logs/, results/）
```

---

### 3. RL-KSP実験

#### 基本的な学習+テスト（一括実行）

```bash
# デフォルト設定で実行
python scripts/rl_ksp/train_rl_ksp.py --config configs/rl_ksp/rl_config.json

# NSFNetトポロジー、10品種
python scripts/rl_ksp/train_rl_ksp.py --config configs/rl_ksp/nsfnet_10_commodities_rl.json

# NSFNetトポロジー、20品種
python scripts/rl_ksp/train_rl_ksp.py --config configs/rl_ksp/nsfnet_20_commodities_rl.json
```

#### 実行フロー
```
1. 設定ファイル読み込み
2. データセット読み込み
3. RL環境・DQNモデル初期化
4. 学習実行（複数エピソード）
   └─ ε-greedy方策で経路探索
5. テスト実行（学習済みモデルで評価）
6. 結果保存（logs/, results/）
```

---

### 4. SeqFlowRL実験

#### 概要

**SeqFlowRL (Sequential Flow Reinforcement Learning)** は、GCNとTealの手法を統合した新しいアプローチです。

**主要な特徴**:
- **Actor-Critic (A2C)** アルゴリズム
- **Per-Commodity GNN更新** で動的な状態反映
- **Sequential Rollout** による逐次経路生成
- **2段階訓練** 対応（教師あり事前学習 + RL微調整）

**設計決定**:
- エンコーダ共有（Actor/Critic）
- グローバルValue関数
- ノードレベルアクション空間（デフォルト）
- Per-Commodity GNN更新頻度

詳細は `configs/seqflowrl/seqflowrl_base.json` 内のコメントを参照してください。

#### データ生成

```bash
# SeqFlowRL用データセット生成
python scripts/common/generate_data.py --config configs/seqflowrl/seqflowrl_base.json
```

#### 基本的な学習

```bash
# デフォルト設定で実行
python scripts/seq_flow_rl/train_seqflowrl.py --config configs/seqflowrl/seqflowrl_base.json
```

#### 2段階訓練（教師あり事前学習 + RL微調整）

```bash
# Phase 1: 教師あり学習で事前学習（既存のGCNスクリプト）
python scripts/gcn/train_gcn.py --config configs/gcn/supervised_pretraining.json

# Phase 2: 設定ファイルで load_pretrained_model を true に設定
# configs/seqflowrl/seqflowrl_base.json 内で:
#   "load_pretrained_model": true
#   "pretrained_model_path": "saved_models/supervised_pretrained.pt"
python scripts/seq_flow_rl/train_seqflowrl.py --config configs/seqflowrl/seqflowrl_base.json
```

#### 出力ファイル

学習完了後、以下のファイルが生成されます：

- **モデルチェックポイント**: `saved_models/seqflowrl/`
  - `best_model.pt` - 最良モデル
  - `latest_model.pt` - 最新モデル
  - `checkpoint_epoch_N.pt` - エポック毎のチェックポイント
- **学習ログ**: `logs/seqflowrl_training_YYYYMMDD_HHMMSS.txt`
  - 詳細な学習経過とメトリクス
  - エポック毎のLoss、Reward、Load Factor、Approximation Ratio

#### 実行フロー
```
1. 設定ファイル読み込み
2. データセット読み込み
3. SeqFlowRLモデル初期化
   ├─ HybridGNNEncoder (共有)
   ├─ PolicyHead (Actor)
   └─ ValueHead (Critic)
4. 事前学習モデルのロード（オプション）
5. A2C訓練実行
   ├─ Sequential Rollout（commodity毎）
   ├─ Per-Commodity GNN更新
   ├─ 報酬計算（Load Factor ベース）
   └─ Actor-Critic損失最適化
6. 検証・チェックポイント保存
7. 結果保存（saved_models/seqflowrl/、logs/）
```

---

## 設定ファイル

### GCN用設定ファイル（`configs/gcn/`）

| ファイル | 説明 |
|---------|------|
| `default.json` | デフォルト設定 |
| `default2.json` | 代替デフォルト設定 |
| `nsfnet_10_commodities.json` | NSFNet 10品種 |
| `nsfnet_15_commodities.json` | NSFNet 15品種 |
| `nsfnet_20_commodities.json` | NSFNet 20品種 |
| `nsfnet_25_commodities.json` | NSFNet 25品種 |
| `tuning_config.json` | ハイパーパラメータ調整用 |
| `load_saved_model.json` | 保存済みモデル読み込み |
| `with_model_saving.json` | モデル保存設定 |

### RL-KSP用設定ファイル（`configs/rl_ksp/`）

| ファイル | 説明 |
|---------|------|
| `rl_config.json` | RLデフォルト設定 |
| `nsfnet_5_commodities_rl.json` | NSFNet 5品種（RL） |
| `nsfnet_10_commodities_rl.json` | NSFNet 10品種（RL） |
| `nsfnet_15_commodities_rl.json` | NSFNet 15品種（RL） |
| `nsfnet_20_commodities_rl.json` | NSFNet 20品種（RL） |
| `rl_load_saved_model.json` | RL保存済みモデル読み込み |
| `rl_with_model_saving.json` | RLモデル保存設定 |

### SeqFlowRL用設定ファイル（`configs/seqflowrl/`）

| ファイル | 説明 |
|---------|------|
| `seqflowrl_base.json` | SeqFlowRLデフォルト設定 |

---

## 結果の確認

### 学習ログ

学習中の進捗は標準出力に表示されます：

```
GCN例:
epoch:01  execution time:12.34s  lr:1.00e-03  loss:0.1234
          mean_maximum_load_factor:0.567  gt_load_factor:0.456
          approximation_rate:80.45  infeasible_rate:5.2

RL-KSP例:
Episode 10/100, Total Reward: -0.0589, Loss: 0.0001,
                Epsilon: 0.2936, Time: 0.39s

SeqFlowRL例:
Epoch 1/30 | Time: 21.86s | LR: 0.000500
  Train - Loss: 1.2155 | Reward: 1.6512 | Load Factor: 0.1726 | Approx Ratio: 85.32%
  Val   - Load Factor: 0.1850 (min: 0.1234, max: 0.2567) | Approx Ratio: 82.10%
```

### 保存されるファイル

- **モデル**:
  - GCN: `saved_models/gcn/`
  - RL-KSP: `saved_models/rl_ksp/`
  - SeqFlowRL: `saved_models/seqflowrl/`
- **ログ**:
  - GCN: `logs/training_results_YYYYMMDD_HHMMSS.txt`
  - SeqFlowRL: `logs/seqflowrl_training_YYYYMMDD_HHMMSS.txt`
- **メトリクス**: `logs/training_results_YYYYMMDD_HHMMSS.pkl`
- **結果**: `results/` ディレクトリ

### 結果の集計

複数実験の平均値を計算：

```bash
python scripts/common/calculate_averages.py
```

---

## トラブルシューティング

### ImportError が発生する場合

プロジェクトルートから実行していることを確認してください：

```bash
# 正しい実行方法
cd /path/to/graph-convnet-uelb
python scripts/gcn/train_gcn.py --config configs/gcn/default2.json

# 間違った実行方法（これはエラーになります）
cd /path/to/graph-convnet-uelb/scripts/gcn
python train_gcn.py --config ../../configs/gcn/default2.json
```

### データが見つからない場合

データを生成してください：

```bash
python scripts/common/generate_data.py --config configs/gcn/default2.json
```

### GPU/CUDAエラーが発生する場合

設定ファイルで `use_gpu: false` を指定してCPUモードで実行：

```json
{
  "use_gpu": false,
  ...
}
```

### メモリ不足エラー

バッチサイズを小さくしてください：

```json
{
  "batch_size": 20,  // デフォルトより小さく
  ...
}
```

---

## ハイパーパラメータ調整

Grid SearchとRandom Searchの2手法をサポート。

### 実行方法

```bash
# Grid Search
python scripts/gcn/tune_hyperparameters.py --search grid

# Random Search（試行回数指定）
python scripts/gcn/tune_hyperparameters.py --search random --trials 50

# 設定ファイル指定
python scripts/gcn/tune_hyperparameters.py --search grid --config configs/gcn/default.json
```

### チューニング設定（`configs/gcn/tuning_config.json`）

```json
{
  "grid_search": {
    "parameters": {
      "learning_rate": [0.001, 0.01, 0.0001],
      "hidden_dim": [32, 64, 128],
      "num_layers": [5, 10, 15]
    }
  },
  "random_search": {
    "n_trials": 20,
    "parameters": {
      "learning_rate": { "type": "float", "min": 0.0001, "max": 0.01, "log_scale": true },
      "hidden_dim": { "type": "choice", "choices": [32, 64, 96, 128, 256] }
    }
  }
}
```

パラメータタイプ: `int`（min/max）、`float`（min/max/log_scale）、`choice`（choices配列）

### チューニング対象パラメータ

| 優先度 | パラメータ |
|--------|-----------|
| 高 | `learning_rate`, `hidden_dim`, `num_layers` |
| 中 | `batch_size`, `dropout_rate`, `beam_size` |
| 低 | `decay_rate`, `mlp_layers`, `aggregation` |
| RL特有 | `rl_baseline_momentum`, `rl_entropy_weight`, `rl_beam_search_type` |

### 結果の確認

結果は `tuning_results/` に保存されます。最適設定で再実行：

```bash
python scripts/gcn/train_gcn.py --config tuning_results/best_config_YYYYMMDD_HHMMSS.json
```

### 推奨ワークフロー

1. 少数パラメータでGrid Search → 2. Random Searchで広範囲探索 → 3. ベスト近辺でGrid Search

---

## 開発者向け情報

### 新しいモジュールを追加する場合

- **GCN用**: `src/gcn/` 以下に追加
- **RL-KSP用**: `src/rl_ksp/` 以下に追加
- **共通**: `src/common/` 以下に追加

### インポートの書き方

```python
# 絶対インポートを使用（相対インポートは使用しない）
from src.gcn.models.gcn_model import ResidualGatedGCNModel
from src.common.data_management.dataset_reader import DatasetReader
from src.rl_ksp.models.dqn_model import DQNModel
```

---

## リファクタリング履歴

2025-10-16にプロジェクト構造を大幅にリファクタリングしました。

詳細は以下のドキュメントを参照：
- `RESTRUCTURE_PLAN.md` - リファクタリング計画
- `RESTRUCTURE_COMPLETE.md` - 完了報告

### 旧ファイルについて

旧構造のファイルは `.backup_old_structure/` にバックアップされています。

---

## ライセンス

（ライセンス情報をここに記載）

---

## お問い合わせ

（連絡先情報をここに記載）

---

## 更新履歴

- **2025-10-16**: プロジェクト構造リファクタリング（GCN/RL-KSP分離）
- **2025-XX-XX**: 初版リリース
