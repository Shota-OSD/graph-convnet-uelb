[[[]()]()]()# CLAUDE.md

## プロジェクト概要

Graph ConvNet UELB — GCN・RL-KSP・SeqFlowRL の3手法で **Urban Edge Load Balancing (UELB)** 問題を解くMLリサーチプロジェクト。
グラフ上で複数コモディティのフローをルーティングし、全エッジの最大負荷率を最小化する。

## セットアップ

```bash
conda env create -f environment.yml
conda activate gcn-uelb-env
```

## よく使うコマンド

### データ生成
```bash
python scripts/common/generate_data.py --config configs/gcn/default2.json
```

### 学習
```bash
# GCN
python scripts/gcn/train_gcn.py --config configs/gcn/default2.json

# RL-KSP
python scripts/rl_ksp/train_rl_ksp.py --config configs/rl_ksp/rl_config.json

# SeqFlowRL
python scripts/seq_flow_rl/train_seqflowrl.py --config configs/seqflowrl/seqflowrl_base.json
```

### 評価
```bash
python scripts/gcn/train_gcn.py --config configs/gcn/load_saved_model.json
python scripts/rl_ksp/test_rl_ksp.py --config configs/rl_ksp/rl_load_saved_model.json
python scripts/seq_flow_rl/evaluate_seqflowrl.py --config configs/seqflowrl/seqflowrl_base.json
```

## プロジェクト構造

```
src/
├── common/          # 共有モジュール（config, data, graph, types, utils, visualization）
├── gcn/             # GCN実装（models, algorithms, train, training, tuning）
├── rl_ksp/          # RL-KSP実装（models, environment, train, tuning）
└── seq_flow_rl/     # SeqFlowRL実装（models, training, algorithms）
scripts/             # エントリーポイントスクリプト
configs/             # JSON設定ファイル（gcn/, rl_ksp/, seqflowrl/）
```

## コーディング規約

- **Python 3.9+** 対象
- **絶対インポート**を使用（例: `from src.gcn.models.gcn_model import ResidualGatedGCNModel`）。相対インポートは使わない
- PyTorch の `nn.Module` を継承してモデルを定義
- dtype は `dtypeFloat`（torch.float32）と `dtypeLong`（torch.long）を使用
- `nn.Embedding` には `.long()` テンソル、`nn.Linear` には `.float()` テンソルを渡す（混同注意）
- 型ヒントでテンソルの形状とdtypeをコメント明示（例: `x_nodes: Tensor  # [B, V, C] - torch.long`）
- 設定は全て JSON ファイルで管理し、`ConfigManager` 経由で読み込む
- 学習戦略は Strategy パターン（`BaseTrainingStrategy` → `SupervisedStrategy` / `ReinforcementStrategy`）

## 技術スタック

- **PyTorch** 2.5.1（CUDA対応）
- **NetworkX** 2.7.1（グラフ処理）
- **PuLP** / **MIP**（最適化）
- **TensorBoard**（ログ可視化）
- **Gymnasium**（RL環境）

## 手法別アーキテクチャ

### GCN（`src/gcn/`）

Residual Gated GCN でエッジスコアを予測し、ビームサーチまたはパスサンプリングでルーティング。

```
入力（隣接行列, 容量, コモディティ）
  → Embedding（node: nn.Embedding, commodity: nn.Linear, edge: nn.Linear）
  → N × ResidualGatedGCNLayer（Gating + Residual + BatchNorm）
  → MLP分類器
  → エッジ予測 [B, V, V, C]
  → 学習: Supervised（NLLLoss）or RL（REINFORCE + PathSampler）
  → 評価: BeamSearch → Load Factor → Approximation Rate
```

**学習戦略（Strategy パターン）:**
- `SupervisedStrategy` — 最適解ラベルとの交差エントロピー
- `ReinforcementStrategy` — REINFORCE + ベースライン + エントロピーボーナス。報酬: `2.0 - 2.0 * load_factor`

**ビームサーチバリアント（Factory パターン）:**
Standard / Deterministic / Greedy / Unconstrained

| クラス | ファイル | 役割 |
|---|---|---|
| `ResidualGatedGCNModel` | `models/gcn_model.py` | Gated GCN（Embedding → N層 → MLP） |
| `ResidualGatedGCNLayer` | `models/gcn_layers.py` | GCN層（NodeFeatures + EdgeFeatures + BN + Residual） |
| `PathSampler` | `algorithms/path_sampler.py` | 温度制御 + Top-p サンプリング + Reachability Mask |
| `BeamSearchFactory` | `algorithms/beamsearch_uelb.py` | ビームサーチ生成（4バリアント） |
| `Trainer` | `train/trainer.py` | 学習ループ（epoch → batch → strategy → optimizer） |
| `Evaluator` | `train/evaluator.py` | 検証・テスト評価 |
| `ModelConverter` | `utils/model_converter.py` | 教師あり→RL モデル変換（logit差分法） |

---

### RL-KSP（`src/rl_ksp/`）

DQN で K最短パスの中から最適なパス割当を学習。Gymnasium環境ベース。

```
KShortestPathFinder（Yen's Algorithm）→ 各コモディティにK本の候補パス生成
  → MinMaxLoadKSPsEnv（Gym環境）
    状態: 上位20件のパス交換候補の報酬ポテンシャル [20]
    行動: 候補の1つを選択（Discrete(20)）
    報酬: -(新max_load - 旧max_load)（改善ベース）
  → DQNModel（3層MLP: 20→32→32→32→20）で Q値を近似
  → epsilon-greedy 探索（ε: 0.8→0.01 に減衰）
  → Q-learning 更新: target = reward + γ * max Q(s', a')
```

| クラス | ファイル | 役割 |
|---|---|---|
| `DQNModel` | `models/dqn_model.py` | 3層MLP の Deep Q-Network |
| `MinMaxLoadKSPsEnv` | `environment/rl_environment.py` | Gym環境（グラフ読込・パス交換・負荷計算） |
| `RLTrainer` | `train/rl_trainer.py` | DQN学習ループ（episode → step → Q更新） |
| `KShortestPathFinder` | `common/graph/k_shortest_path.py` | Yen's Algorithm で K最短パス生成 |

---

### SeqFlowRL（`src/seq_flow_rl/`）

GCNエンコーダ + Actor-Critic（A2C）で、コモディティを1つずつ逐次的にルーティング。

```
For each commodity c:
  → HybridGNNEncoder（容量 + 動的使用量を入力、Per-Commodity再エンコード）
    → node_features [B,V,C,H], edge_features [B,V,V,C,H], graph_embedding [B,H]
  → PolicyHead（Actor）: current_node + dst_node → MLP → action_probs [B,V]
    → MaskGenerator で制約適用（容量・自己ループ・訪問済み・到達可能性）
    → 温度制御 + Top-p サンプリングで次ノード選択
  → ValueHead（Critic）: node_features → MLP → V(s) [B]（グローバル価値）
  → エッジ使用量を更新 → 次のコモディティへ

A2C損失 = L_actor + 0.5 * L_critic + 0.01 * L_entropy
報酬: 2.0 - 2.0 * load_factor
```

**設計決定:**
- エンコーダ共有（Actor/Critic）
- グローバルValue関数（per-commodityではない）
- ノードレベルアクション空間（デフォルト）
- Per-Commodity GNN更新（コモディティ毎に再エンコード）
- 2フェーズ学習対応（教師あり事前学習 → RL微調整）

| クラス | ファイル | 役割 |
|---|---|---|
| `SeqFlowRLModel` | `models/seqflowrl_model.py` | Actor-Critic統合モデル |
| `HybridGNNEncoder` | `models/hybrid_gnn_encoder.py` | GNNエンコーダ（容量+使用量入力、GCN層再利用） |
| `PolicyHead` | `models/policy_head.py` | Actor（ノード/エッジレベル行動、Top-pサンプリング） |
| `ValueHead` | `models/value_head.py` | Critic（グローバルV(s)、MLP: V\*C\*H→512→256→128→1） |
| `SequentialRolloutEngine` | `algorithms/sequential_rollout.py` | 逐次パス生成（commodity毎にGNN再エンコード） |
| `A2CStrategy` | `training/a2c_strategy.py` | A2C損失計算（advantage正規化 + 勾配クリップ） |
| `SeqFlowRLTrainer` | `training/trainer.py` | 学習ループ（epoch → batch → rollout → A2C更新） |
| `MaskGenerator` | `utils/mask_utils.py` | 行動マスク（容量・ループ・訪問済み・到達可能性） |

**評価メトリクス:**

| メトリクス | 定義 | 備考 |
|---|---|---|
| Load Factor | `max_edge(usage / capacity)` | 全エッジの最大負荷率。小さいほど良い |
| Complete Rate | `(dst到達コモディティ数 / 全コモディティ数) × 100` | 全コモディティが目的地に到達した割合。100% が理想 |
| Approximation Ratio | `mean(gt_load_factor_i / model_load_factor_i) × 100` | サンプル単位で最適解との比を取り平均。100% に近いほど良い。**全コモディティが dst に到達したサンプルのみ** で計算（不完全解は除外） |
| Reward | `2.0 - 2.0 * load_factor` | 連続報酬。容量超過時は追加ペナルティあり |

---

### 共通モジュール（`src/common/`）

| クラス | ファイル | 役割 |
|---|---|---|
| `ConfigManager` | `config/config_manager.py` | JSON設定の読込・パース |
| `DatasetReader` | `data_management/dataset_reader.py` | バッチイテレータ |
| `MetricsLogger` | `train/metrics.py` | 学習メトリクス記録 |

## データ形式

- グラフ: GML形式（NetworkX互換）
- データディレクトリ: `train_data/`, `val_data/`, `test_data/`（コモディティ別サブディレクトリ）
- モデル保存先: `saved_models/{gcn,rl_ksp,seqflowrl}/best_model.pt`
- ログ: `logs/training_results_YYYYMMDD_HHMMSS.{txt,pkl}`
