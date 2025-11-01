# RL-GCN + Teal ハイブリッド手法の実装計画

**作成日**: 2025-10-22
**目的**: 既存のRL-GCNとTealの強みを組み合わせた高性能なUELB解法の開発

---

## 📊 エグゼクティブサマリー

### 現状分析

**あなたのRL-GCN**:
- ✅ 洗練されたResidual Gated GCN アーキテクチャ
- ✅ REINFORCE + PathSampler による離散パス選択
- ✅ Beam search での効率的な経路探索
- ✅ バッチ処理対応
- ⚠️ 小〜中規模ネットワーク向け（スケーラビリティに課題）
- ⚠️ 密な隣接行列（O(B×V²×C×H)のメモリ）

**Teal (SIGCOMM '23)**:
- ✅ 疎グラフ表現（O(E×H)のメモリ、スケーラブル）
- ✅ 連続フロー配分による最適化
- ✅ ADMM制約処理（容量・需要制約の自動調整）
- ✅ COMA式報酬推定（分散削減）
- ✅ 大規模WAN対応（6,474ノードで実証）
- ⚠️ シンプルなGNN（FlowGNN = 線形層 + スパース行列積）

### 提案手法

**ハイブリッドアプローチ**: 既存のResidualGatedGCNを維持し、Tealのフロー最適化テクニックを統合

```
既存 RL-GCN (維持)          Teal技術 (追加)
┌─────────────────┐        ┌──────────────────┐
│ ResidualGatedGCN│   →    │ Flow Allocation  │
│ (Batch処理)      │        │ ADMM Constraints │
│ 密な隣接行列      │        │ Flow Rounding    │
└─────────────────┘        │ COMA Rewards     │
                           └──────────────────┘
```

**期待される効果**:
- 🎯 制約満足率の向上（ADMMによる自動調整）
- 🎯 学習の安定化（COMA報酬推定）
- 🎯 最適解の品質向上（フロー配分の洗練化）
- 🎯 将来的なスケール拡張の基盤

---

## 🔍 技術的背景

### ResidualGatedGCN vs Teal FlowGNN

| 項目 | ResidualGatedGCN | Teal FlowGNN |
|------|------------------|--------------|
| **アーキテクチャ** | Gating機構 + Residual | 単純な線形層 |
| **グラフ表現** | 密な隣接行列 (B×V×V×C×H) | 疎なCOO形式 (E×H) |
| **メモリ効率** | O(B×V²×C×H) | O(E×H) |
| **バッチ処理** | ✅ あり | ❌ なし |
| **コモディティ処理** | 明示的に分離 | パスノードとして統合 |
| **エッジ更新** | ✅ 双方向更新 | ⚠️ 単方向更新 |
| **正規化** | BatchNorm | なし |
| **表現力** | 高い | 低い |
| **スケール** | 小〜中規模 | 大規模 |

### メモリ使用量の比較

**小規模 (10ノード、5コモディティ、バッチ32)**:
- ResidualGatedGCN: 約 9 MB
- FlowGNN: 約 76 KB
- **削減率: 120倍**

**中規模 (50ノード、20コモディティ、バッチ32)**:
- ResidualGatedGCN: 約 1.6 GB
- FlowGNN: 約 500 KB
- **削減率: 3,200倍**

**大規模 (100ノード、50コモディティ、バッチ32)**:
- ResidualGatedGCN: 約 32 GB（GPU1台に載らない）
- FlowGNN: 約 1 MB
- **削減率: 32,000倍**

### Tealの主要技術

#### 1. ADMM (Alternating Direction Method of Multipliers)

**目的**: 容量・需要制約違反を反復的に修正

```python
# Tealのコード（参考）
for iteration in range(num_admm_steps):
    # 1. 容量制約違反を計算
    edge_flow = aggregate_path_flow_to_edges(path_flow)
    util = edge_flow / capacity

    # 2. 違反度に応じてフロー調整
    violation_factor = relu(util - 1.0) + 1.0
    path_adjustment = compute_adjustment(violation_factor)

    # 3. フローを更新
    path_flow = path_flow - learning_rate * path_adjustment
```

**効果**:
- 制約違反を自動的に修正
- 実現可能解の生成率が向上
- 最適化の収束性が改善

#### 2. COMA (Counterfactual Multi-Agent) 報酬推定

**問題**: Policy Gradientは分散が大きい

**解決策**: 反事実的ベースラインで分散削減

```python
# 現在の行動の報酬
current_reward = env.step(current_action)

# 代替行動をサンプリングして平均報酬を計算
baseline_reward = 0
for _ in range(num_samples):
    alt_action = sample_alternative_action()
    baseline_reward += env.step(alt_action)
baseline_reward /= num_samples

# Advantage = 現在の報酬 - ベースライン
advantage = current_reward - baseline_reward
loss = -log_prob * advantage
```

**効果**:
- 勾配の分散を大幅削減
- 学習の安定化
- 収束速度の向上

#### 3. Flow Rounding

**目的**: 連続フロー配分を実現可能解に変換

```python
def round_flow(path_flow, capacity, demand):
    # Step 1: 需要制約を満たすよう正規化
    for commodity in commodities:
        total_flow = sum(path_flow[commodity])
        if total_flow > demand[commodity]:
            path_flow[commodity] *= demand[commodity] / total_flow

    # Step 2: 容量制約を満たすよう反復削減
    for iteration in range(num_iterations):
        edge_flow = aggregate_to_edges(path_flow)
        violation_ratio = edge_flow / capacity

        for path in paths:
            max_violation = max(violation_ratio[edges_in_path])
            path_flow[path] /= max_violation

    return path_flow
```

**効果**:
- 必ず実現可能な解を生成
- 最適解に近い品質を維持

---

## 🎯 ハイブリッド手法の設計

### アプローチ選択: ResidualGatedGCN + Teal Flow Framework（推奨）

**理由**:
1. 既存のGCNアーキテクチャを維持（再実装不要）
2. バッチ処理を維持（訓練効率）
3. Tealの実証済みテクニックのみ導入（リスク低減）
4. 段階的な実装・検証が可能

### システムアーキテクチャ

```
┌─────────────────────────────────────────────────────────┐
│ Input: Graph topology, Commodities, Capacities         │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│ Phase 1: Feature Extraction (既存)                      │
│ ┌───────────────────────────────────────────────────┐   │
│ │ ResidualGatedGCNModel                             │   │
│ │ - Embedding layers                                │   │
│ │ - 3 x ResidualGatedGCNLayer                       │   │
│ │   - Gating mechanism                              │   │
│ │   - Residual connections                          │   │
│ │   - Batch normalization                           │   │
│ └───────────────────────────────────────────────────┘   │
│ Output: Edge scores (B, V, V, C)                        │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│ Phase 2: Flow Allocation (新規)                         │
│ ┌───────────────────────────────────────────────────┐   │
│ │ FlowAllocationLayer                               │   │
│ │ - Convert edge scores → path flows                │   │
│ │ - Apply softmax per commodity                     │   │
│ │ - Multiply by demands                             │   │
│ └───────────────────────────────────────────────────┘   │
│ Output: Path flows (B, C, K) K=num_paths               │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│ Phase 3: Constraint Handling (新規)                     │
│ ┌───────────────────────────────────────────────────┐   │
│ │ SimpleFlowRounder (Phase 1A)                      │   │
│ │ - Round demand constraints                        │   │
│ │ - Round capacity constraints                      │   │
│ │                                                    │   │
│ │ ADMMConstraintHandler (Phase 1B)                  │   │
│ │ - Iterative ADMM refinement                       │   │
│ │ - Dual variable updates                           │   │
│ └───────────────────────────────────────────────────┘   │
│ Output: Feasible path flows (B, C, K)                  │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│ Phase 4: Reward Computation & Learning (改良)           │
│ ┌───────────────────────────────────────────────────┐   │
│ │ FlowRewardComputer                                │   │
│ │ - Load factor (existing)                          │   │
│ │ - Total flow (Teal)                               │   │
│ │ - Constraint violation penalty                    │   │
│ │                                                    │   │
│ │ COMARewardEstimator (Phase 1B)                    │   │
│ │ - Counterfactual baseline                         │   │
│ │ - Advantage computation                           │   │
│ │                                                    │   │
│ │ Policy Gradient Loss                              │   │
│ │ loss = -log_prob * advantage                      │   │
│ └───────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

---

## 📝 実装計画

### Phase 1A: 最小実装（2週間）

**目標**: 基本的なフロー配分機能を実装し、既存RL-GCNと同等性能を確認

#### 1. FlowAllocationLayer

**ファイル**: `src/gcn_flow/models/flow_allocation_layer.py`

```python
class FlowAllocationLayer(nn.Module):
    """
    GCNのエッジスコアをパスごとのフロー配分に変換

    Input:
        edge_scores: (B, V, V, C) - GCNの出力
        paths: List[(src, dst, [path_nodes])] - 候補パス
        demands: (B, C) - コモディティ需要

    Output:
        path_flows: (B, num_paths) - 各パスのフロー
        log_probs: (B,) - ログ確率（Policy Gradient用）
    """
```

**実装内容**:
- エッジスコア → パススコア集約
- Softmax per commodity
- 需要との乗算
- ログ確率の計算

#### 2. SimpleFlowRounder

**ファイル**: `src/gcn_flow/algorithms/flow_rounder.py`

```python
class SimpleFlowRounder:
    """
    基本的なフロー丸め処理

    Tealのround_flow()を簡略化した版
    """

    def round(self, path_flows, capacity, demands):
        # 1. 需要制約の強制
        # 2. 容量制約の強制（2回反復）
        return feasible_flows
```

#### 3. FlowRewardComputer

**ファイル**: `src/gcn_flow/utils/metrics.py`

```python
class FlowRewardComputer:
    """
    複合報酬関数の計算

    reward = α * load_factor_reward
           + β * flow_utilization_reward
           - γ * constraint_violation_penalty
    """

    def compute_reward(self, flows, capacities, demands):
        # Load factor (from RL-GCN)
        load_reward = -compute_max_load_factor(flows, capacities)

        # Total flow (from Teal)
        flow_reward = flows.sum() / demands.sum()

        # Violation penalty
        violation = self._compute_violation(flows, capacities, demands)
        penalty = -violation

        return self.alpha * load_reward + \
               self.beta * flow_reward + \
               self.gamma * penalty
```

#### 4. HybridRLStrategy (基本版)

**ファイル**: `src/gcn_flow/training/hybrid_rl_strategy.py`

```python
class HybridRLStrategy(BaseTrainingStrategy):
    """
    RL-GCN + Teal フロー最適化のハイブリッド戦略

    Phase 1A: REINFORCE + Simple Rounding
    Phase 1B: COMA + ADMM
    """

    def __init__(self, config):
        super().__init__(config)

        # Flow allocation
        self.flow_layer = FlowAllocationLayer(...)

        # Constraint handling
        self.flow_rounder = SimpleFlowRounder()

        # Reward computation
        self.reward_computer = FlowRewardComputer(
            alpha=config.get('reward_alpha', 1.0),
            beta=config.get('reward_beta', 0.1),
            gamma=config.get('reward_gamma', 5.0)
        )

    def compute_loss(self, model, batch_data, device):
        # 1. GCN forward
        edge_scores, _ = model.forward(...)

        # 2. Flow allocation
        path_flows, log_probs = self.flow_layer(
            edge_scores, paths, demands
        )

        # 3. Round to feasible solution
        feasible_flows = self.flow_rounder.round(
            path_flows, capacities, demands
        )

        # 4. Compute reward
        reward = self.reward_computer.compute_reward(
            feasible_flows, capacities, demands
        )

        # 5. Policy gradient loss (REINFORCE)
        advantage = reward - self.baseline
        loss = -(log_probs * advantage.detach()).mean()

        # 6. Update baseline
        self.baseline = 0.9 * self.baseline + 0.1 * reward.mean()

        metrics = {
            'loss': loss.item(),
            'reward': reward.mean().item(),
            'advantage': advantage.mean().item(),
            'max_load': compute_max_load_factor(feasible_flows, capacities),
            'total_flow': feasible_flows.sum().item(),
        }

        return loss, metrics
```

#### タスクリスト

- [ ] FlowAllocationLayer の実装
- [ ] SimpleFlowRounder の実装
- [ ] FlowRewardComputer の実装
- [ ] HybridRLStrategy の実装
- [ ] PathGenerator ユーティリティ（k-shortest paths）
- [ ] 単体テストの作成
- [ ] 既存RL-GCNとの性能比較実験

**期待される成果**:
- 既存RL-GCNと同等の性能
- フロー配分の可視化
- ベースライン確立

---

### Phase 1B: Teal統合（2週間）

**目標**: ADMMとCOMAを導入し、性能向上を確認

#### 5. ADMMConstraintHandler

**ファイル**: `src/gcn_flow/algorithms/admm_handler.py`

```python
class ADMMConstraintHandler:
    """
    Teal-style ADMM制約処理

    Augmented Lagrangian:
    L(x, λ, ρ) = f(x) + λᵀ(Ax-b) + (ρ/2)||Ax-b||²
    """

    def __init__(self, rho=1.0, num_iterations=5):
        self.rho = rho  # ペナルティパラメータ
        self.num_iterations = num_iterations
        self.dual_vars = None  # ラグランジュ乗数

    def refine(self, path_flows, p2e_mapping, capacities, demands):
        """
        ADMM反復によるフロー調整

        Args:
            path_flows: (B, num_paths) - 初期フロー
            p2e_mapping: (2, num_path_edge_pairs) - パス→エッジマッピング
            capacities: (B, num_edges) - エッジ容量
            demands: (B, num_commodities) - 需要

        Returns:
            refined_flows: (B, num_paths) - 調整後フロー
        """
        x = path_flows.clone()

        # Initialize dual variables
        if self.dual_vars is None:
            num_edges = capacities.shape[1]
            self.dual_vars = torch.zeros(capacities.shape)

        for iteration in range(self.num_iterations):
            # 1. Compute edge flows
            edge_flows = self._aggregate_to_edges(x, p2e_mapping)

            # 2. Compute constraint violations
            capacity_violation = F.relu(edge_flows - capacities)

            # 3. Update dual variables
            self.dual_vars += self.rho * capacity_violation

            # 4. Update primal variables (path flows)
            # Gradient of augmented Lagrangian
            gradient = self._compute_gradient(
                x, edge_flows, capacities, self.dual_vars, p2e_mapping
            )
            x = x - (1.0 / self.rho) * gradient

            # 5. Project to feasible set
            x = self._project_to_demands(x, demands)
            x = F.relu(x)  # Non-negative flows

        return x
```

#### 6. COMARewardEstimator

**ファイル**: `src/gcn_flow/algorithms/coma_reward.py`

```python
class COMARewardEstimator:
    """
    Counterfactual Multi-Agent 報酬推定

    分散削減のため、代替行動をサンプリングして
    ベースラインを計算
    """

    def __init__(self, num_samples=5):
        self.num_samples = num_samples

    def estimate_advantage(self, model, batch_data, current_action, current_reward):
        """
        Args:
            model: GCNモデル
            batch_data: バッチデータ
            current_action: 現在の行動（パスフロー）
            current_reward: 現在の報酬

        Returns:
            advantage: current_reward - baseline_reward
        """
        baseline_rewards = []

        with torch.no_grad():
            for _ in range(self.num_samples):
                # Alternative action:
                # Option 1: Sample from uniform distribution
                alt_action = torch.rand_like(current_action)
                alt_action = self._normalize_to_demands(alt_action, batch_data)

                # Option 2: Sample from model with noise
                # noise = torch.randn_like(current_action) * 0.1
                # alt_action = current_action + noise

                # Compute reward for alternative action
                alt_reward = self._compute_reward(alt_action, batch_data)
                baseline_rewards.append(alt_reward)

        baseline = torch.stack(baseline_rewards).mean(dim=0)
        advantage = current_reward - baseline

        return advantage
```

#### 7. HybridRLStrategy の拡張

```python
class HybridRLStrategy(BaseTrainingStrategy):
    def __init__(self, config):
        # ... (Phase 1Aの内容)

        # Phase 1B additions
        if config.get('use_admm', True):
            self.admm_handler = ADMMConstraintHandler(
                rho=config.get('admm_rho', 1.0),
                num_iterations=config.get('admm_iterations', 5)
            )

        if config.get('use_coma', True):
            self.coma_estimator = COMARewardEstimator(
                num_samples=config.get('coma_samples', 5)
            )

    def compute_loss(self, model, batch_data, device):
        # 1-2. Same as Phase 1A
        edge_scores, _ = model.forward(...)
        path_flows, log_probs = self.flow_layer(...)

        # 3. ADMM refinement (instead of simple rounding)
        if hasattr(self, 'admm_handler'):
            feasible_flows = self.admm_handler.refine(
                path_flows, p2e_mapping, capacities, demands
            )
        else:
            feasible_flows = self.flow_rounder.round(...)

        # 4. Compute reward
        reward = self.reward_computer.compute_reward(...)

        # 5. COMA advantage estimation
        if hasattr(self, 'coma_estimator'):
            advantage = self.coma_estimator.estimate_advantage(
                model, batch_data, feasible_flows, reward
            )
        else:
            advantage = reward - self.baseline
            self.baseline = 0.9 * self.baseline + 0.1 * reward.mean()

        # 6. Policy gradient loss
        loss = -(log_probs * advantage.detach()).mean()

        return loss, metrics
```

#### タスクリスト

- [ ] ADMMConstraintHandler の実装
- [ ] COMARewardEstimator の実装
- [ ] HybridRLStrategy の拡張
- [ ] ADMM収束性のテスト
- [ ] COMA分散削減効果の検証
- [ ] Phase 1Aとの性能比較実験

**期待される成果**:
- 制約満足率の向上（90% → 98%+）
- 学習の安定化（勾配分散の削減）
- 最適解への近似度向上

---

### Phase 2: スケール対応（2週間）

**目標**: より大規模な問題インスタンスへの対応

#### 8. GraphConverter

**ファイル**: `src/gcn_flow/utils/graph_converter.py`

```python
class GraphConverter:
    """
    密な隣接行列 ⇔ 疎なCOO形式の変換

    大規模問題では疎表現に変換してメモリ削減
    """

    def dense_to_sparse(self, adjacency_matrix, capacity_matrix):
        """
        Args:
            adjacency_matrix: (B, V, V)
            capacity_matrix: (B, V, V)

        Returns:
            edge_index: (2, E) - COO format
            edge_attr: (E, features)
        """
        pass

    def sparse_to_dense(self, edge_index, edge_attr, num_nodes):
        """Reverse conversion"""
        pass
```

#### 9. EfficientPathGenerator

**ファイル**: `src/gcn_flow/utils/path_generator.py`

```python
class EfficientPathGenerator:
    """
    効率的なk-shortest path生成

    Yen's algorithmまたはSuurballeのアルゴリズムを使用
    """

    def generate_paths(self, graph, commodities, k=4, edge_disjoint=False):
        """
        Args:
            graph: NetworkX graph
            commodities: List[(src, dst, demand)]
            k: Number of paths per commodity
            edge_disjoint: Whether to find edge-disjoint paths

        Returns:
            path_dict: {(src, dst): [path1, path2, ..., pathk]}
        """
        pass
```

#### タスクリスト

- [ ] GraphConverter の実装
- [ ] EfficientPathGenerator の実装
- [ ] メモリプロファイリング
- [ ] 大規模インスタンスでのテスト（50+ ノード）
- [ ] バッチサイズ vs メモリ使用量のトレードオフ分析

**期待される成果**:
- 50ノード以上の問題に対応
- メモリ使用量の大幅削減
- より現実的な規模での評価

---

## 📊 評価計画

### ベースラインとの比較

| 手法 | Phase 1A | Phase 1B | Phase 2 |
|------|----------|----------|---------|
| **既存RL-GCN** | ✓ | ✓ | ✓ |
| **Supervised GCN** | ✓ | ✓ | ✓ |
| **Gurobi (最適解)** | ✓ | ✓ | ✓ |
| **Teal (if applicable)** | - | - | ✓ |

### 評価指標

**主要指標**:
1. **最大負荷率 (Max Load Factor)**: 小さいほど良い
2. **実行時間**: 速いほど良い
3. **制約満足率**: 高いほど良い（目標: 95%+）

**副次指標**:
4. **総フロー**: 大きいほど良い
5. **学習曲線**: 安定しているほど良い
6. **収束エポック数**: 少ないほど良い

### 実験セットアップ

**小規模問題** (Phase 1A/1B):
- ノード数: 10-20
- コモディティ数: 5-10
- バッチサイズ: 32
- エポック数: 100

**中規模問題** (Phase 1B/2):
- ノード数: 20-50
- コモディティ数: 10-30
- バッチサイズ: 16
- エポック数: 150

**大規模問題** (Phase 2):
- ノード数: 50-100
- コモディティ数: 30-100
- バッチサイズ: 4-8
- エポック数: 200

---

## 📂 ディレクトリ構造

```
src/gcn_flow/
├── models/
│   ├── __init__.py
│   ├── flow_gnn.py                 # ✅ 実装済み（参考用）
│   ├── flow_actor.py               # ✅ 実装済み（参考用）
│   ├── flow_allocation_layer.py    # 🆕 Phase 1A - パスフロー配分
│   └── hybrid_gcn_model.py         # 🆕 Phase 1A - 統合モデル
│
├── algorithms/
│   ├── __init__.py
│   ├── flow_env.py                 # ✅ 実装済み（参考用）
│   ├── flow_rounder.py             # 🆕 Phase 1A - Simple rounding
│   ├── admm_handler.py             # 🆕 Phase 1B - ADMM制約処理
│   └── coma_reward.py              # 🆕 Phase 1B - COMA報酬推定
│
├── training/
│   ├── __init__.py
│   ├── hybrid_rl_strategy.py       # 🆕 Phase 1A - ハイブリッド戦略
│   └── flow_trainer.py             # ✅ 実装済み（要改良）
│
├── utils/
│   ├── __init__.py
│   ├── metrics.py                  # 🆕 Phase 1A - 報酬計算
│   ├── path_generator.py           # 🆕 Phase 1A/2 - パス生成
│   ├── graph_converter.py          # 🆕 Phase 2 - グラフ変換
│   └── visualization.py            # 🆕 Optional - 可視化
│
├── configs/
│   ├── hybrid_rl_phase1a.json      # 🆕 Phase 1A設定
│   ├── hybrid_rl_phase1b.json      # 🆕 Phase 1B設定
│   └── hybrid_rl_phase2.json       # 🆕 Phase 2設定
│
├── tests/
│   ├── test_flow_allocation.py     # 🆕 単体テスト
│   ├── test_admm.py                # 🆕 単体テスト
│   └── test_coma.py                # 🆕 単体テスト
│
└── README.md                        # ✅ 実装済み
```

---

## 🚀 実装スケジュール

### Week 1-2: Phase 1A (最小実装)

**Week 1**:
- Day 1-2: FlowAllocationLayer 実装
- Day 3-4: SimpleFlowRounder 実装
- Day 5: FlowRewardComputer 実装

**Week 2**:
- Day 1-2: HybridRLStrategy 実装
- Day 3-4: PathGenerator 実装
- Day 5: 統合テスト & デバッグ

**Milestone**: 既存RL-GCNと同等性能を達成

### Week 3-4: Phase 1B (Teal統合)

**Week 3**:
- Day 1-3: ADMMConstraintHandler 実装
- Day 4-5: ADMM収束性テスト

**Week 4**:
- Day 1-2: COMARewardEstimator 実装
- Day 3-4: HybridRLStrategy 拡張
- Day 5: 性能比較実験

**Milestone**: Phase 1Aより10%以上の性能向上

### Week 5-6: Phase 2 (スケール対応)

**Week 5**:
- Day 1-2: GraphConverter 実装
- Day 3-4: EfficientPathGenerator 実装
- Day 5: メモリプロファイリング

**Week 6**:
- Day 1-3: 大規模実験
- Day 4-5: 結果分析 & ドキュメント

**Milestone**: 50ノード以上の問題で実用的な性能

### Week 7-8: 最終調整 & 論文準備

- ハイパーパラメータチューニング
- 追加実験
- 可視化・グラフ作成
- ドキュメント整備

---

## ⚙️ 設定例

### Phase 1A設定

```json
{
  "expt_name": "hybrid_rl_phase1a",
  "training_strategy": "hybrid_rl",

  "model": {
    "num_layers": 3,
    "hidden_dim": 128,
    "dropout_rate": 0.3
  },

  "hybrid_rl": {
    "use_flow_allocation": true,
    "use_admm": false,
    "use_coma": false,

    "flow_rounder": {
      "type": "simple",
      "num_iterations": 2
    },

    "reward": {
      "alpha": 1.0,
      "beta": 0.1,
      "gamma": 5.0
    },

    "baseline_momentum": 0.9
  },

  "path_generator": {
    "k_paths": 4,
    "edge_disjoint": false
  },

  "training": {
    "learning_rate": 0.0001,
    "batch_size": 32,
    "max_epochs": 100
  }
}
```

### Phase 1B設定

```json
{
  "expt_name": "hybrid_rl_phase1b",
  "training_strategy": "hybrid_rl",

  "hybrid_rl": {
    "use_flow_allocation": true,
    "use_admm": true,
    "use_coma": true,

    "admm": {
      "rho": 1.0,
      "num_iterations": 5
    },

    "coma": {
      "num_samples": 5
    },

    "reward": {
      "alpha": 1.0,
      "beta": 0.2,
      "gamma": 5.0
    }
  },

  "training": {
    "learning_rate": 0.0001,
    "batch_size": 32,
    "max_epochs": 150
  }
}
```

---

## 🔧 トラブルシューティング

### Issue 1: ADMMが収束しない

**症状**: ADMM反復後も制約違反が残る

**解決策**:
1. `rho` パラメータを調整（0.1 → 10.0の範囲）
2. 反復回数を増やす（5 → 10）
3. 初期フローの品質を改善（GCNの事前訓練）

### Issue 2: COMA報酬推定の分散が大きい

**症状**: 学習が不安定

**解決策**:
1. サンプル数を増やす（5 → 10）
2. 代替行動の生成方法を変更（uniform → Gaussian noise）
3. Advantage normalizationを追加

### Issue 3: メモリ不足

**症状**: OOM エラー

**解決策**:
1. バッチサイズを減らす
2. Gradient accumulationを使用
3. Phase 2のGraphConverterで疎表現に変換

### Issue 4: 既存RL-GCNより性能が悪い

**症状**: Phase 1Aで期待される性能が出ない

**診断**:
1. FlowAllocationLayerのスコア集約が正しいか確認
2. 報酬関数の各項の重み (α, β, γ) を調整
3. PathGeneratorが適切なパスを生成しているか確認
4. SimpleFlowRounderが過度に制約を満たそうとしていないか

---

## 📚 参考文献

1. **Teal**: Xu et al., "Teal: Learning-Accelerated Optimization of WAN Traffic Engineering", SIGCOMM 2023
2. **ADMM**: Boyd et al., "Distributed Optimization and Statistical Learning via ADMM", 2011
3. **COMA**: Foerster et al., "Counterfactual Multi-Agent Policy Gradients", AAAI 2018
4. **Residual Gated GCN**: Bresson & Laurent, "Residual Gated Graph ConvNets", 2017
5. **REINFORCE**: Williams, "Simple Statistical Gradient-Following Algorithms", 1992

---

## ✅ 次のステップ

**即座に開始**:
1. ✅ 計画レポート作成完了
2. 🚀 Phase 1A実装開始
   - FlowAllocationLayer から実装
   - 段階的なテスト・検証

**短期目標（2週間）**:
- Phase 1A完成
- 既存RL-GCNとの性能比較

**中期目標（1ヶ月）**:
- Phase 1B完成
- ADMM/COMA効果の実証

**長期目標（2ヶ月）**:
- Phase 2完成
- 大規模問題での評価
- 論文執筆準備

---

**作成者**: Claude (Anthropic)
**バージョン**: 1.0
**最終更新**: 2025-10-22
