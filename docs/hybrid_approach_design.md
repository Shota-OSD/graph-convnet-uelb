# ハイブリッドアプローチ設計書
## GCN + Teal 統合手法の詳細設計

**作成日**: 2025-11-01
**対象**: グラフ最大負荷率最小化問題
**目的**: GCNとTealの長所を組み合わせた新しいRL手法の開発

---

## 目次

1. [設計方針](#設計方針)
2. [アーキテクチャ概要](#アーキテクチャ概要)
3. [詳細設計](#詳細設計)
4. [意思決定ポイント（判断が必要な箇所）](#意思決定ポイント判断が必要な箇所)
5. [実装ロードマップ](#実装ロードマップ)
6. [期待される効果](#期待される効果)

---

## 設計方針

### **統合する長所**

#### **GCNから採用**
- ✅ **End-to-End学習**: グラフ構造から直接経路を学習
- ✅ **柔軟な経路生成**: 事前計算不要、任意の経路を探索可能
- ✅ **教師あり事前学習**: 最適解データの活用
- ✅ **Residual Gated GCN**: 強力なグラフ表現学習

#### **Tealから採用**
- ✅ **Policy Head分離**: モジュール性と再利用性
- ✅ **逐次決定（環境更新）**: 動的な状態反映
- ✅ **小次元アクション空間**: 計算効率とサンプリング効率
- ✅ **GNN+DNN交互層**: 容量・需要制約の明示的キャプチャ

### **統合の基本戦略**

```
Residual Gated GCN (エンコーダ)
    ↓
Policy Head (アクション選択)
    ↓
Sequential Rollout (環境との相互作用)
    ↓
Actor-Critic RL (安定学習)
```

---

## アーキテクチャ概要

### **システム全体図**

```
┌─────────────────────────────────────────────────────────────┐
│                    Input Layer                               │
│  - Graph Structure (Adjacency, Capacity)                     │
│  - Commodities (src, dst, demand)                           │
│  - Current Network State (dynamic edge usage)               │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                 Embedding Layer                              │
│  - Node Embedding (commodity info)                          │
│  - Edge Embedding (capacity + current usage)                │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│            Graph Neural Network Encoder                      │
│  ┌───────────────────────────────────────────────┐          │
│  │  Residual Gated GCN Layers (x N)              │          │
│  │  - Node Features: [B, V, C, H]                │          │
│  │  - Edge Features: [B, V, V, C, H]             │          │
│  └───────────────────────────────────────────────┘          │
│                  [決定ポイント #1]                            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              Policy & Value Heads                            │
│  ┌─────────────────────┐   ┌──────────────────────┐        │
│  │   Actor (Policy)    │   │   Critic (Value)     │        │
│  │   Next-hop probs    │   │   State value V(s)   │        │
│  │   π(a|s,u,dst)      │   │                      │        │
│  └─────────────────────┘   └──────────────────────┘        │
│                  [決定ポイント #2]                            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│           Sequential Rollout Engine                          │
│  For each commodity:                                         │
│    While current_node ≠ destination:                        │
│      1. Get GNN state encoding                              │
│      2. Policy Head → next hop distribution                 │
│      3. Sample next_node                                    │
│      4. Update network state (edge usage)                   │
│      5. Re-encode with GNN [Optional]                       │
│                  [決定ポイント #3]                            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                Reward Computation                            │
│  - Load Factor calculation                                   │
│  - Feasibility check                                        │
│  - Continuous reward function                               │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              RL Loss Computation                             │
│  Actor Loss:  -log π(a|s) × (R - V(s))                     │
│  Critic Loss: MSE(V(s), R)                                  │
│  Entropy:     H(π) for exploration                          │
│                  [決定ポイント #4]                            │
└─────────────────────────────────────────────────────────────┘
```

---

## 詳細設計

### **1. モデルアーキテクチャ**

#### **1.1 HybridGNNEncoder**

```python
class HybridGNNEncoder(nn.Module):
    """
    GCNのResidual Gated GCNをベースにしたエンコーダ
    動的な状態（edge usage）を考慮��た設計
    """
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers

        # Embedding layers
        self.node_embedding = nn.Embedding(config.voc_nodes_in, hidden_dim // 2)
        self.commodity_embedding = nn.Linear(1, hidden_dim // 2)

        # [NEW] 動的エッジ状態の埋め込み
        self.edge_capacity_embedding = nn.Linear(1, hidden_dim // 2)
        self.edge_usage_embedding = nn.Linear(1, hidden_dim // 2)

        # Residual Gated GCN layers
        self.gcn_layers = nn.ModuleList([
            ResidualGatedGCNLayer(hidden_dim, config.aggregation)
            for _ in range(num_layers)
        ])

    def forward(self, x_nodes, x_commodities, x_edges_capacity,
                x_edges_usage=None):
        """
        Args:
            x_nodes: [B, V, C] - ノードのcommodity情報
            x_commodities: [B, C, 3] - (src, dst, demand)
            x_edges_capacity: [B, V, V] - エッジ容量（静的）
            x_edges_usage: [B, V, V] - 現在のエッジ使用量（動的）

        Returns:
            node_features: [B, V, C, H] - ノード特徴
            edge_features: [B, V, V, C, H] - エッジ特徴
        """
        # Embedding
        x = self.node_embedding(x_nodes)
        c = self.commodity_embedding(x_commodities[:, :, 2:3])  # demand

        # [NEW] Dynamic edge features
        if x_edges_usage is None:
            x_edges_usage = torch.zeros_like(x_edges_capacity)

        e_capacity = self.edge_capacity_embedding(
            x_edges_capacity.unsqueeze(-1).unsqueeze(-1)
        )  # [B, V, V, 1, H/2]
        e_usage = self.edge_usage_embedding(
            x_edges_usage.unsqueeze(-1).unsqueeze(-1)
        )  # [B, V, V, 1, H/2]
        e = torch.cat([e_capacity, e_usage], dim=-1)  # [B, V, V, 1, H]
        e = e.expand(-1, -1, -1, x_commodities.size(1), -1)

        # Aggregate node + commodity features
        x_agg = x.mean(dim=2).unsqueeze(2).expand(-1, -1, c.size(1), -1)
        x = torch.cat([x_agg, c.unsqueeze(1).expand(-1, x.size(1), -1, -1)], -1)

        # GCN layers
        for layer in self.gcn_layers:
            x, e = layer(x, e)

        return x, e
```

**設計のポイント**:
- ✅ GCNのResidual Gated GCN構造を継承
- ✅ **動的エッジ使用量** (`x_edges_usage`) を入力に追加
- ✅ 容量と使用量を別々に埋め込み、結合

---

#### **1.2 PolicyHead（Actor）** - 柔軟なアクション空間設計

```python
class PolicyHead(nn.Module):
    """
    Teal風の分離されたPolicy Head
    現在ノードから次ホップを選択する確率分布を出力

    【実装】両方のアクション空間タイプをサポート:
    - 'node': ノードレベルアクション（デフォルト）
    - 'edge': エッジレベルアクション

    重要: このHeadは「次のホップを選択する確率分布」を出力するのみ。
    目的地到達判定やループ制御は呼び出し側（SequentialRolloutEngine）が担当。
    """
    def __init__(self, hidden_dim, num_nodes, action_type='node', mlp_layers=2):
        """
        Args:
            hidden_dim: GNNエンコーダの隠れ層次元
            num_nodes: グラフのノード数
            action_type: 'node' (デフォルト) または 'edge'
            mlp_layers: MLPの層数
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.action_type = action_type

        if action_type == 'node':
            # [Option 1] Node-level action: 次ノードを直接選択
            # 入力: current_node特徴 + dst_node特徴
            # 出力: V個のノードの確率分布
            self.action_mlp = MLP(hidden_dim * 2, num_nodes, num_layers=mlp_layers)

        elif action_type == 'edge':
            # [Option 2] Edge-level action: エッジスコアを計算
            # 入力: エッジ特徴
            # 出力: 各エッジのスコア（1次元）
            self.action_mlp = MLP(hidden_dim, 1, num_layers=mlp_layers)

        else:
            raise ValueError(f"Unknown action_type: {action_type}. "
                           f"Must be 'node' or 'edge'.")

    def forward(self, node_features, edge_features, current_node, dst_node,
                commodity_idx, valid_edges_mask, reached_destination=False):
        """
        Args:
            node_features: [B, V, C, H] from encoder
            edge_features: [B, V, V, C, H] from encoder
            current_node: int - 現在のノード
            dst_node: int - 目的地ノード
            commodity_idx: int - commodity index
            valid_edges_mask: [B, V] - 有効エッジのマスク (node), or [B, V, V] (edge)
            reached_destination: bool - 目的地到達済みフラグ

        Returns:
            action_probs: [B, V] - 次ホップの確率分布

        Note:
            両方のaction_typeで出力形式は [B, V] に統一。
            edge typeの場合、current_nodeからの出力エッジのみを計算。
        """
        batch_size = node_features.size(0)

        # [SAFETY CHECK] 目的地到達済みの場合は警告
        if reached_destination:
            import warnings
            warnings.warn(
                f"PolicyHead called for commodity {commodity_idx} that already "
                f"reached destination. This should be handled by the caller."
            )
            return torch.ones(batch_size, self.num_nodes,
                            device=node_features.device) / self.num_nodes

        if self.action_type == 'node':
            # [Node-level action]
            # 現在ノードと目的地ノードの特徴を結合
            current_feat = node_features[:, current_node, commodity_idx, :]  # [B, H]
            dst_feat = node_features[:, dst_node, commodity_idx, :]  # [B, H]
            context = torch.cat([current_feat, dst_feat], dim=-1)  # [B, 2H]

            # 次ホップのロジット
            action_logits = self.action_mlp(context)  # [B, V]

            # 無効エッジのマスキング (valid_edges_mask: [B, V])
            action_logits = action_logits.masked_fill(~valid_edges_mask, -1e9)

            # Softmax for probabilities
            action_probs = F.softmax(action_logits, dim=-1)

        elif self.action_type == 'edge':
            # [Edge-level action]
            # 現在ノードから出る全エッジの特徴を取得
            # edge_features: [B, V, V, C, H]
            outgoing_edges = edge_features[:, current_node, :, commodity_idx, :]  # [B, V, H]

            # 各エッジのスコアを計算
            edge_scores = self.action_mlp(outgoing_edges).squeeze(-1)  # [B, V]

            # 無効エッジのマスキング
            # valid_edges_mask: [B, V] または [B, V, V]
            if len(valid_edges_mask.shape) == 3:
                # [B, V, V] -> [B, V] (current_nodeからの出力エッジのみ)
                valid_mask = valid_edges_mask[:, current_node, :]
            else:
                # [B, V]
                valid_mask = valid_edges_mask

            edge_scores = edge_scores.masked_fill(~valid_mask, -1e9)

            # Softmax for probabilities
            action_probs = F.softmax(edge_scores, dim=-1)

        return action_probs
```

**設計のポイント**:
- ✅ **両方のアクション空間を実装**
  - `action_type='node'`: ノードレベル（**デフォルト**）
  - `action_type='edge'`: エッジレベル
- ✅ 設定ファイルで切り替え可能
- ✅ 出力形式は両方とも `[B, V]` に統一
- ✅ Tealの分離設計を採用
- ✅ 現在ノード + 目的地のコンテキストを考慮（node type）
- ✅ エッジ特徴を直接使用（edge type）
- ✅ 無効エッジのマスキング（両タイプ）
- ✅ **目的地到達フラグの受け取り**（安全性チェック用）
- ⚠️ **重要**: 目的地到達判定は呼び出し側（SequentialRolloutEngine）が責任を持つ

---

#### **1.3 ValueHead（Critic）** ✅ **確定: グローバルValue**

```python
class ValueHead(nn.Module):
    """
    状態価値V(s)を推定するCritic
    Actor-Critic学習で分散削減

    【確定】決定ポイント #2-B: Option 2-B-1（グローバルValue）を採用
    - 状態全体を単一のスカラー値で評価
    - 最終目標（load factor最小化）と直接整合
    """
    def __init__(self, hidden_dim, num_nodes, num_commodities, mlp_layers=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.num_commodities = num_commodities

        # [確定] Global value: 全ノード×全commodityの特徴を入力
        # 入力次元: hidden_dim * num_nodes * num_commodities
        # 出力: 単一スカラー値 V(s)
        self.value_mlp = MLP(
            hidden_dim * num_nodes * num_commodities,
            1,
            num_layers=mlp_layers
        )

    def forward(self, node_features):
        """
        Args:
            node_features: [B, V, C, H] from encoder
                B: batch size
                V: num_nodes
                C: num_commodities
                H: hidden_dim

        Returns:
            state_value: [B, 1] - グローバル状態価値
        """
        # 全ノード・commodity特徴をフラット化
        batch_size = node_features.size(0)
        global_feat = node_features.view(batch_size, -1)  # [B, V*C*H]

        # MLPで状態価値を推定
        state_value = self.value_mlp(global_feat)  # [B, 1]

        return state_value
```

**設計のポイント**:
- ✅ **確定**: グローバル状態価値（単一スカラー）
- ✅ GCNにはないCriticネットワーク（新規追加）
- ✅ 最終目標（最大load factor最小化）と直接対応
- ✅ 標準的なA2C/PPOアーキテクチャ
- ✅ シンプルで解釈しやすい

---

#### **1.4 統合モデル: HybridActorCritic**

```python
class HybridActorCritic(nn.Module):
    """
    GCN + Tealのハイブリッドモデル
    Actor-Critic構造

    【確定】決定ポイント #1: Option 1-A（共有エンコーダ）を採用
    - ActorとCriticで単一のGNNエンコーダを共有
    - メリット: パラメータ効率、表現学習の一貫性、訓練安定性
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # [確定] Shared encoder for both Actor and Critic
        self.encoder = HybridGNNEncoder(config)

        # Policy Head (Actor) with flexible action type
        action_type = config.get('action_type', 'node')  # デフォルト: 'node'
        policy_mlp_layers = config.get('policy_head_mlp_layers', 2)
        self.actor = PolicyHead(
            config.hidden_dim,
            config.num_nodes,
            action_type=action_type,
            mlp_layers=policy_mlp_layers
        )

        # Value Head (Critic)
        value_mlp_layers = config.get('value_head_mlp_layers', 3)
        self.critic = ValueHead(
            config.hidden_dim,
            config.num_nodes,
            config.num_commodities,
            mlp_layers=value_mlp_layers
        )

    def forward(self, state, mode='train'):
        """
        Args:
            state: dict with keys
                - x_nodes: [B, V, C]
                - x_commodities: [B, C, 3]
                - x_edges_capacity: [B, V, V]
                - x_edges_usage: [B, V, V] (dynamic)
            mode: 'train' or 'eval'

        Returns:
            If training: node_features, edge_features, state_value
            If eval: node_features, edge_features
        """
        # 共有エンコーダで特徴抽出
        node_feat, edge_feat = self.encoder(
            state['x_nodes'],
            state['x_commodities'],
            state['x_edges_capacity'],
            state.get('x_edges_usage', None)
        )

        if mode == 'train':
            # 訓練時: ActorとCriticの両方の出力を返す
            state_value = self.critic(node_feat)
            return node_feat, edge_feat, state_value
        else:
            # 評価時: Actorの特徴量のみ
            return node_feat, edge_feat
```

**設計のポイント**:
- ✅ **確定**: ActorとCriticで単一エンコーダを共有
- ✅ パラメータ数削減（メモリ効率）
- ✅ 表現学習の一貫性
- ✅ 標準的なA2C/PPOアーキテクチャ

---

### **2. Sequential Rollout Engine** ✅ **確定: Per-Commodity GNN更新**

```python
class SequentialRolloutEngine:
    """
    逐次的な経路生成エンジン
    Commodity毎にGNN状態を更新（動的な相互作用を反映）

    【確定】決定ポイント #3: Option 3-B（Per-Commodity）を採用
    - Commodity毎にエッジ使用量を反映してGNNを再計算
    - 計算コストと動的性のバランスを取る中間案
    """
    def __init__(self, model, config):
        self.model = model
        self.config = config

        # [確定] Per-commodity GNN更新
        self.gnn_update_frequency = config.get('gnn_update_frequency', 'per_commodity')
        # 'per_commodity': commodity毎にGNN更新（デフォルト、確定）
        # 'never': 初回のみGNN計算（デバッグ・比較用）
        # 'per_step': 各ホップ毎にGNN更新（実験用、計算コスト高）

        self.temperature = config.get('sampling_temperature', 1.0)
        self.top_p = config.get('sampling_top_p', 0.9)

    def rollout(self, batch_data, mode='train'):
        """
        バッチデータに対して逐次的に経路を生成

        Args:
            batch_data: dict with graph and commodity info
            mode: 'train' or 'eval'

        Returns:
            paths: [B, C, max_path_len] - 生成された経路
            log_probs: [B] - 総log確率
            state_values: [B] - 状態価値（trainingのみ）
            entropy: [B] - エントロピー（trainingのみ）
        """
        batch_size = batch_data['x_nodes'].size(0)
        num_commodities = batch_data['x_commodities'].size(1)
        device = batch_data['x_nodes'].device

        # 初期状態のエンコード
        state = {
            'x_nodes': batch_data['x_nodes'],
            'x_commodities': batch_data['x_commodities'],
            'x_edges_capacity': batch_data['x_edges_capacity'],
            'x_edges_usage': torch.zeros_like(batch_data['x_edges_capacity'])
        }

        node_feat, edge_feat, state_value = self.model(state, mode='train')

        # 各バッチの経路とログ確率を記録
        batch_paths = []
        batch_log_probs = []
        batch_entropies = []

        for b in range(batch_size):
            commodity_paths = []
            total_log_prob = torch.zeros(1, device=device)
            total_entropy = torch.zeros(1, device=device)

            for c in range(num_commodities):
                src = int(batch_data['x_commodities'][b, c, 0].item())
                dst = int(batch_data['x_commodities'][b, c, 1].item())
                demand = batch_data['x_commodities'][b, c, 2].item()

                # [確定] Per-commodity GNN更新
                # 前のcommodityのエッジ使用量を反映してGNNを再計算
                if c > 0 or self.gnn_update_frequency == 'per_commodity':
                    # 更新された状態でGNNを再エンコード
                    # state['x_edges_usage']には前commodityまでの使用量が蓄積
                    node_feat, edge_feat, _ = self.model(state, mode='train')

                # 経路サンプリング
                path, log_prob, entropy, reached_dst = self._sample_path(
                    node_feat[b], edge_feat[b], src, dst, c,
                    state['x_edges_capacity'][b],
                    state['x_edges_usage'][b]
                )

                commodity_paths.append(path)
                total_log_prob += log_prob
                total_entropy += entropy

                # [重要] 目的地到達フラグの記録（報酬計算で使用）
                # 不完全パス（reached_dst=False）は報酬でペナルティ
                # この情報は後で報酬計算時に使える
                # （現在の設計では、報酬計算側で path[-1] != dst をチェック）

                # エッジ使用量を更新（次のcommodityのGNN更新で反映）
                self._update_edge_usage(
                    state['x_edges_usage'][b], path, demand
                )

            batch_paths.append(commodity_paths)
            batch_log_probs.append(total_log_prob)
            batch_entropies.append(total_entropy)

        log_probs_tensor = torch.stack(batch_log_probs).squeeze(-1)
        entropy_tensor = torch.stack(batch_entropies).squeeze(-1)

        if mode == 'train':
            return batch_paths, log_probs_tensor, state_value, entropy_tensor
        else:
            return batch_paths, log_probs_tensor

    def _sample_path(self, node_feat, edge_feat, src, dst, commodity_idx,
                     edges_capacity, edges_usage):
        """
        単一commodity用の経路サンプリング（1 hopずつ）

        重要な処理:
        1. 目的地到達判定（if current == dst: break）
        2. 訪問済みノードのマスキング（ループ防止）
        3. 到達可能性マスク（デッドエンド回避）
        4. 最大ステップ数制限（無限ループ防止）

        Returns:
            path: List[int] - ノードのシーケンス
            log_prob: Tensor - 経路の総log確率
            entropy: Tensor - 経路生成時のエントロピー
            reached_dst: bool - 目的地到達フラグ
        """
        path = [src]
        log_prob = torch.zeros(1, device=node_feat.device)
        entropy = torch.zeros(1, device=node_feat.device)
        current = src
        max_steps = self.config.num_nodes * 2
        reached_dst = False

        for step in range(max_steps):
            # [重要] 目的地到達チェック
            if current == dst:
                reached_dst = True
                break

            # [オプション] Per-step GNN更新（実験用、デフォルトでは無効）
            # 確定: Per-commodity更新を使用するため、通常はこの分岐に入らない
            if self.gnn_update_frequency == 'per_step':
                # 各ステップで再エンコード（計算コスト非常に高い）
                # 実験・比較用にのみ使用
                state_updated = self._get_current_state()
                node_feat, edge_feat, _ = self.model(state_updated, mode='eval')

            # 有効エッジのマスク（容量あり、self-loop禁止、訪問済み除外）
            valid_mask = (edges_capacity[current] > 0) & \
                         (torch.arange(edges_capacity.size(0),
                                      device=edges_capacity.device) != current)

            # [追加] 訪問済みノードを除外（ループ防止）
            for visited_node in path:
                valid_mask[visited_node] = False

            # [オプション] 到達可能性マスク（GCN PathSamplerと同様）
            # reachability_mask = self.reachability[batch_idx, :, dst]
            # valid_mask = valid_mask & reachability_mask

            # 有効な次ホップが存在しない場合
            if not valid_mask.any():
                # 目的地に到達できない（デッドエンド）
                # 不完全パスとして返す
                break

            # Policy Headで次ホップ確率取得
            action_probs = self.model.actor(
                node_feat.unsqueeze(0),
                edge_feat.unsqueeze(0),
                current, dst, commodity_idx,
                valid_mask.unsqueeze(0),
                reached_destination=False  # まだ到達していない
            ).squeeze(0)  # [V]

            # Top-p sampling
            next_node = self._top_p_sample(action_probs)

            # Log prob & entropy
            log_prob += torch.log(action_probs[next_node] + 1e-8)
            entropy += -(action_probs * torch.log(action_probs + 1e-8)).sum()

            path.append(int(next_node))
            current = int(next_node)

        return path, log_prob, entropy, reached_dst

    def _update_edge_usage(self, edges_usage, path, demand):
        """経路に沿ってエッジ使用量を更新"""
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            edges_usage[u, v] += demand

    def _top_p_sample(self, probs):
        """Nucleus (top-p) sampling"""
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumsum_probs = torch.cumsum(sorted_probs, dim=0)
        mask = cumsum_probs <= self.top_p
        mask[0] = True  # 最低1つは含める
        filtered_probs = sorted_probs * mask.float()
        filtered_probs /= filtered_probs.sum()
        sampled_idx = torch.multinomial(filtered_probs, 1).item()
        return sorted_indices[sampled_idx].item()
```

**設計のポイント**:
- ✅ GCNの逐次サンプリング + Tealの環境更新を統合
- ✅ エッジ使用量を動的に追跡
- ✅ **目的地到達判定を明示的に処理**
- ✅ **訪問済みノードマスク**でループ防止
- ✅ **確定: Per-commodity GNN更新**（決定ポイント #3-B）
  - Commodity毎にエッジ使用量を反映してGNNを再計算
  - 計算コスト: O(C) × GNN順伝播（C: commodity数）
  - Commodity間の相互作用を動的に反映
- ✅ **確定: A2Cアルゴリズム**（決定ポイント #4-A）
  - Actor-Criticアーキテクチャでadvantage計算
  - シンプルで安定した学習

---

### **2.1 目的地到達マスクの処理詳細**

#### **問題点**

バッチ内で複数のcommodityを処理する際、以下の状況が発生します：

1. **すでに目的地に到達したcommodity**
   - `current_node == dst_node` の場合
   - これ以上の経路サンプリングは不要

2. **まだ経路探索中のcommodity**
   - `current_node != dst_node` の場合
   - Policy Headで次ホップを選択

#### **解決方針**

**責任分離アーキテクチャ**:

```
┌──────────────────────────────────────┐
│ SequentialRolloutEngine              │
│  - 目的地到達判定（if current == dst）│ ← 主責任
│  - ループ制御（for step in max_steps）│
│  - 訪問済みノードマスク              │
│  - 到達可能性チェック                │
└──────────────────────────────────────┘
              ↓ 呼び出し
┌──────────────────────────────────────┐
│ PolicyHead                           │
│  - 次ホップ確率分布の計算のみ        │ ← 副次的な安全チェック
│  - reached_destination フラグ受取    │   （本来は呼ばれないはず）
│  - 無効エッジマスキング              │
└──────────────────────────────────────┘
```

#### **実装の詳細**

**1. SequentialRolloutEngine側の処理**:

```python
for step in range(max_steps):
    # [1] 目的地到達判定（最優先）
    if current == dst:
        reached_dst = True
        break  # ループ終了、PolicyHeadは呼ばれない

    # [2] 訪問済みノードマスク（ループ防止）
    valid_mask = ...
    for visited_node in path:
        valid_mask[visited_node] = False

    # [3] 到達可能性マスク（オプション、デッドエンド回避）
    # reachability_mask = self.reachability[batch_idx, :, dst]
    # valid_mask = valid_mask & reachability_mask

    # [4] デッドエンドチェック
    if not valid_mask.any():
        # 有効な次ホップなし → 不完全パス
        break

    # [5] Policy Headで次ホップ選択
    action_probs = self.model.actor(
        ...,
        reached_destination=False  # まだ到達していない
    )

    # [6] サンプリングと経路更新
    next_node = self._top_p_sample(action_probs)
    path.append(next_node)
    current = next_node
```

**2. PolicyHead側の処理**:

```python
def forward(self, ..., reached_destination=False):
    # 安全性チェック（デバッグ用）
    if reached_destination:
        warnings.warn("PolicyHead called after reaching destination")
        return dummy_distribution

    # 通常の処理
    action_logits = self.action_mlp(context)
    action_logits = action_logits.masked_fill(~valid_edges_mask, -1e9)
    return F.softmax(action_logits, dim=-1)
```

#### **マスクの種類と優先順位**

| マスク種類 | 目的 | 適用箇所 | 必須度 |
|-----------|------|---------|--------|
| 目的地到達マスク | 到達済みcommodityの処理停止 | RolloutEngine | **必須** |
| 無効エッジマスク | 容量0エッジの除外 | PolicyHead | **必須** |
| Self-loopマスク | 自己ループ禁止 | RolloutEngine | **必須** |
| 訪問済みマスク | ループ防止 | RolloutEngine | **必須** |
| 到達可能性マスク | デッドエンド回避 | RolloutEngine | 推奨 |

#### **報酬計算での不完全パス処理**

```python
def _compute_rewards(self, paths, batch_data):
    rewards = []
    for b in range(batch_size):
        for c_idx, path in enumerate(paths[b]):
            dst = batch_data['x_commodities'][b, c_idx, 1]

            # 目的地到達チェック
            if len(path) == 0 or path[-1] != dst:
                # 不完全パス → ペナルティ
                reward = -0.5  # 軽めのペナルティ
            else:
                # 完全パス → load factor計算
                load_factor = ...
                reward = 2.0 - 2.0 * load_factor
            rewards.append(reward)
    return torch.tensor(rewards)
```

#### **バッティング箇所**

**なし** - 目的地到達処理は既存GCNと同様の設計

- GCN: `path_sampler.py:295` で `if current == dst: break`
- Hybrid: `SequentialRolloutEngine._sample_path:506` で同様の処理

---

### **3. RL訓練戦略**

```python
class HybridRLStrategy:
    """
    Actor-Critic強化学習戦略
    A2CまたはPPOベース
    """
    def __init__(self, config):
        self.config = config

        # [決定ポイント #4] RLアルゴリズムの選択
        self.rl_algorithm = config.get('rl_algorithm', 'a2c')
        # Options:
        # - 'a2c': Advantage Actor-Critic
        # - 'ppo': Proximal Policy Optimization
        # - 'reinforce_baseline': 既存GCN方式（移動平均baseline）

        self.entropy_weight = config.get('entropy_weight', 0.01)
        self.value_loss_weight = config.get('value_loss_weight', 0.5)
        self.gamma = config.get('gamma', 0.99)  # 割引率（将来報酬）

        # PPO-specific
        if self.rl_algorithm == 'ppo':
            self.ppo_epsilon = config.get('ppo_epsilon', 0.2)
            self.ppo_epochs = config.get('ppo_epochs', 4)

    def compute_loss(self, model, batch_data, rollout_engine):
        """
        バッチデータに対してRL損失を計算

        Returns:
            loss: Total loss
            metrics: dict with detailed metrics
        """
        # Rollout paths
        paths, log_probs, state_values, entropy = rollout_engine.rollout(
            batch_data, mode='train'
        )

        # 報酬計算（GCNの連続報酬関数を使用）
        rewards = self._compute_rewards(paths, batch_data)

        # Advantage計算
        if self.rl_algorithm == 'reinforce_baseline':
            # GCN方式: 移動平均baseline
            advantages = rewards - self.baseline
            self.baseline = 0.95 * self.baseline + 0.05 * rewards.mean()
        else:
            # A2C/PPO: Critic推定値をbaseline
            advantages = rewards - state_values.squeeze(-1).detach()

        # Advantage正規化
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Actor loss (Policy Gradient)
        if self.rl_algorithm == 'ppo':
            # PPO clipped objective
            ratio = torch.exp(log_probs - log_probs.detach())
            clipped_ratio = torch.clamp(ratio, 1-self.ppo_epsilon, 1+self.ppo_epsilon)
            actor_loss = -torch.min(
                ratio * advantages,
                clipped_ratio * advantages
            ).mean()
        else:
            # A2C/REINFORCE
            actor_loss = -(log_probs * advantages).mean()

        # Critic loss (TD error)
        critic_loss = F.mse_loss(state_values.squeeze(-1), rewards)

        # Entropy bonus (exploration)
        entropy_loss = -entropy.mean()

        # Total loss
        total_loss = (
            actor_loss +
            self.value_loss_weight * critic_loss +
            self.entropy_weight * entropy_loss
        )

        metrics = {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy': -entropy_loss.item(),
            'mean_reward': rewards.mean().item(),
            'mean_advantage': advantages.mean().item(),
            'mean_state_value': state_values.mean().item()
        }

        return total_loss, metrics

    def _compute_rewards(self, paths, batch_data):
        """
        GCNの連続報酬関数を使用
        reward = 2.0 - 2.0 * load_factor
        """
        batch_size = len(paths)
        rewards = []

        for b in range(batch_size):
            # Load factor計算
            load_factor = self._calculate_load_factor(
                paths[b],
                batch_data['x_commodities'][b],
                batch_data['x_edges_capacity'][b]
            )

            # 連続報酬関数
            if load_factor == float('inf'):
                reward = -1.0  # 無効エッジ使用
            else:
                reward = 2.0 - 2.0 * load_factor
                reward = max(-2.0, min(2.0, reward))

            rewards.append(reward)

        return torch.tensor(rewards, dtype=torch.float32)
```

**設計のポイント**:
- ✅ A2C/PPOでCriticを活用
- ✅ GCNの連続報酬関数を継承
- ⚠️ **RLアルゴリズムの選択が必要**（決定ポイント #4）

---

## 意思決定ポイント（判断が必要な箇所）

### **決定ポイント #1: エンコーダの共有** ✅ **確定**

**問題**: ActorとCriticでGNNエンコーダを共有するか？

**【確定】採用案: Option 1-A（共有エンコーダ）**

```python
# Single shared encoder
self.encoder = HybridGNNEncoder(config)
self.actor = PolicyHead(...)
self.critic = ValueHead(...)
```

**採用理由**:
- ✅ パラメータ数削減（メモリ効率）
- ✅ 表現学習の一貫性
- ✅ 訓練安定性向上
- ✅ A2C/PPOでは共有が一般的で安定している

**実装への影響**:
- `HybridActorCritic.__init__()` で単一の `self.encoder` を定義
- ActorとCriticは同じ特徴量を共有
- 勾配はbackward時に両方のHeadから伝播

**代替案（不採用）**: ~~Option 1-B（分離エンコーダ）~~
- パラメータ数2倍、過学習リスクのため不採用

---

### **決定ポイント #2: アクション・Value表現**

#### **決定ポイント #2-A: アクション空間の設計** ✅ **両方実装**

**問題**: 次ホップ選択をどう表現するか？

**【実装方針】両方のオプションを実装し、設定ファイルで切り替え可能にする**

##### **Option 2-A-1: ノードレベルアクション（デフォルト）**

```python
# 設定: "action_type": "node"
# Output: [B, V] - 次ノードの確率分布
action_probs = policy_head(context)  # V個のノードから選択
```

**メリット**:
- ✅ 直感的（次ノードを直接選択）
- ✅ サンプリング効率が高い
- ✅ GCNの現在実装に近い
- ✅ 計算コスト小（入力: hidden_dim × 2）

**デメリット**:
- ❌ 有効エッジのマスキングが必須
- ❌ エッジ特徴を間接的にしか使わない

**使用シーン**:
- 初期実装、ベースライン
- 高速推論が必要な場合
- グラフが大きい場合（V > 50）

##### **Option 2-A-2: エッジレベルアクション**

```python
# 設定: "action_type": "edge"
# Output: [B, V] - current_nodeからの出力エッジの確率分布
edge_scores = policy_head(edge_features[current_node, :])  # Vつの出力エッジ
```

**メリット**:
- ✅ エッジ属性を直接考慮
- ✅ より豊富な情報（容量、使用量など）
- ✅ エッジ特徴を直接活用

**デメリット**:
- ❌ 計算コストやや高（入力: hidden_dim）
- ❌ 実装がやや複雑

**使用シーン**:
- エッジ属性が重要な場合
- より高精度が必要な場合
- グラフが小さい場合（V < 30）

##### **実装**: 両方を `PolicyHead` に統合
```python
# 初期化時に選択
policy_head = PolicyHead(..., action_type='node')  # または 'edge'

# 設定ファイル
{
  "action_type": "node",  # デフォルト
  "_action_type_options": ["node", "edge"]
}
```

**切り替え方法**:
- 設定ファイルの `action_type` を変更するのみ
- モデルの再訓練が必要（アーキテクチャが異なるため）

---

#### **決定ポイント #2-B: Value表現の粒度** ✅ **確定**

**問題**: 状態価値V(s)の粒度は？

**【確定】採用案: Option 2-B-1（グローバルValue）**

```python
# Output: [B, 1] - 状態全体の価値
# 入力: 全ノード×全commodity特徴 [B, V*C*H]
state_value = critic(global_features)
```

**採用理由**:
- ✅ **状態全体を単一スカラーで評価**（最終目標に直結）
- ✅ 標準的なA2C/PPO設計
- ✅ 目的関数（最大load factor最小化）と直接対応
- ✅ シンプルで解釈しやすい
- ✅ 実装・デバッグが容易

**実装への影響**:
- `ValueHead.__init__()`: MLP入力次元 = `hidden_dim * num_nodes * num_commodities`
- `ValueHead.forward()`: 出力 = `[B, 1]`（バッチごとに1つのスカラー値）
- RL損失計算: `critic_loss = MSE(state_value, reward)`

**代替案（不採用）**: ~~Option 2-B-2（Per-commodity Value）~~
- commodity毎の評価は複雑で、最終目標（グローバルload factor）との乖離があるため不採用

**設定ファイル**:
```json
{
  "_comment_value": "=== Value Network ===",
  "value_head_mlp_layers": 3,
  "value_type": "global"
}
```

---

### **決定ポイント #3: GNN再計算の頻度** ✅ **確定**

**問題**: 逐次決定中、いつGNNを再計算するか？

**【確定】採用案: Option 3-B（Per-Commodity）**

#### **Option 3-A: Never（GCN方式）** - 比較用

```python
# 初回のみGNN計算、以降は固定
node_feat, edge_feat = encoder(initial_state)
for commodity in commodities:
    for step in range(max_steps):
        action = actor(node_feat, ...)  # 固定feature使用
```

**メリット**:
- ✅ **計算コスト最小**（GNN順伝播1回のみ）
- ✅ 実装が簡単
- ✅ 既存GCNと同等の効率

**デメリット**:
- ❌ 動的状態（エッジ使用量）が反映されない
- ❌ 後続commodityが先行の影響を学習できない

**計算量**: O(1) × GNN計算

---

#### **Option 3-B: Per-Commodity（中間案）** ✅ **採用**

```python
# [確定] Commodity毎にGNN更新
for commodity in commodities:
    state['x_edges_usage'] = current_usage  # 前commodityまでの使用量を反映
    node_feat, edge_feat = encoder(state)  # 再計算
    for step in range(max_steps):
        action = actor(node_feat, ...)  # 同一commodity内は固定feature
```

**採用理由**:
- ✅ **Commodity間の相互作用を反映**（動的性の確保）
- ✅ 計算コストとのバランスが良い
- ✅ 現実的な動的更新（先行commodityの影響を考慮）
- ✅ 実装の複雑性が適度
- ✅ デバッグ・解釈が容易

**トレードオフ**:
- ⚠️ GNN計算がC倍（commodity数）
  - 例: C=5の場合、GNN順伝播が5回/バッチ
  - 許容可能な計算コスト（GPUで並列化可能）
- ⚠️ 同一commodity内のステップ間は固定
  - Per-Stepに比べて動的性は劣るが、実用上問題なし

**計算量**: O(C) × GNN計算

**設定ファイル**:
```json
{
  "gnn_update_frequency": "per_commodity"  // デフォルト、確定
}
```

---

#### **Option 3-C: Per-Step（Teal方式、最も動的）** - 実験用

```python
# 各ホップ毎にGNN更新
for commodity in commodities:
    for step in range(max_steps):
        state['x_edges_usage'] = current_usage  # 毎回更新
        node_feat, edge_feat = encoder(state)  # 毎回再計算
        action = actor(node_feat, ...)
```

**メリット**:
- ✅ **最も動的で現実的**
- ✅ 各ステップで最新状態を反映
- ✅ 高い表現力

**デメリット**:
- ❌ **計算コスト爆発**（GNN計算 × C × 平均経路長）
- ❌ 訓練時間大幅増加
- ❌ メモリ使用量増加

**計算量**: O(C × avg_path_length) × GNN計算

---

#### **比較表**

| Option | GNN計算回数 | 計算コスト | 動的性 | 推奨度 | 確定状況 |
|--------|------------|----------|--------|--------|---------|
| 3-A: Never | 1回/バッチ | **最小** | 低 | ⭐⭐⭐ 初期実装 | 実験用 |
| **3-B: Per-Commodity** | **C回/バッチ** | **中** | **中** | **⭐⭐⭐⭐ デフォルト** | **✅ 採用** |
| 3-C: Per-Step | C×L回/バッチ | **最大** | 高 | ⭐ 性能優先 | 実験用 |

(C: commodity数、L: 平均経路長)

#### **✅ 確定採用**: Option 3-B (Per-Commodity)

**デフォルト設定として確定**:
- Commodity毎にGNNを再計算
- 各commodityの経路決定後、エッジ使用量を更新して次のcommodityのGNN計算に反映
- 計算コストと動的性のバランスが最適

#### **実装方針**:

1. **デフォルト**: Option 3-B（Per-Commodity）
   - `gnn_update_frequency = 'per_commodity'`（デフォルト）
   - 推奨設定として実装

2. **実験用オプション**:
   - Option 3-A (Never): `gnn_update_frequency = 'never'`
   - Option 3-C (Per-Step): `gnn_update_frequency = 'per_step'`

#### **設定例**:
```json
{
  "gnn_update_frequency": "per_commodity",  // デフォルト（確定）
  // "gnn_update_frequency": "never",      // 実験用
  // "gnn_update_frequency": "per_step"    // 実験用
}
```

---

### **決定ポイント #4: RLアルゴリズムの選択** ✅ **確定**

**問題**: どのRLアルゴリズムを採用するか？

#### **Option 4-A: A2C (Advantage Actor-Critic)** ✅ **採用**

```python
# On-policy, single-step update
advantage = reward - state_value.detach()
actor_loss = -(log_prob * advantage).mean()
critic_loss = F.mse_loss(state_value, reward)
```

**メリット**:
- ✅ **シンプルで安定**
- ✅ Criticがbaseline（分散削減）
- ✅ 既存GCNからの自然な拡張
- ✅ 実装・デバッグが容易

**デメリット**:
- ❌ サンプル効率やや低い
- ❌ On-policyのみ

**採用理由**:
- 初期実装として最適なバランス
- デバッグ・検証が容易
- 安定性を確保しながらCriticの効果を確認可能
- 将来的にPPOへの移行も容易

---

#### **Option 4-B: PPO (Proximal Policy Optimization)** - 実験用

```python
# Clipped objective for stability
ratio = exp(log_prob - old_log_prob)
clipped_ratio = clip(ratio, 1-epsilon, 1+epsilon)
actor_loss = -min(ratio * advantage, clipped_ratio * advantage).mean()
```

**メリット**:
- ✅ **最も安定**（現代RL標準）
- ✅ 大きなポリシー更新を防ぐ
- ✅ サンプル効率向上
- ✅ ハイパーパラメータに頑健

**デメリット**:
- ❌ 実装がやや複雑（複数epoch更新）
- ❌ 計算コストやや高い

**位置づけ**:
- 将来的な拡張オプション
- A2Cで性能限界を感じた場合の代替案

---

#### **Option 4-C: REINFORCE + Baseline（既存GCN方式）** - 非推奨

```python
# Simple policy gradient
advantage = reward - baseline  # baselineは移動平均
actor_loss = -(log_prob * advantage).mean()
baseline = 0.95 * baseline + 0.05 * reward.mean()
```

**メリット**:
- ✅ **最もシンプル**
- ✅ Criticネットワーク不要
- ✅ 既存GCNと同じ

**デメリット**:
- ❌ **分散大**（学習不安定）
- ❌ サンプル効率低い
- ❌ 移動平均baselineは粗い

**位置づけ**:
- Hybrid設計ではCriticを活用するため採用しない
- 既存GCNとの比較実験用のみ

---

#### **比較表**

| アルゴリズム | 安定性 | サンプル効率 | 実装難易度 | Critic要否 | 確定状況 |
|------------|--------|------------|----------|-----------|---------|
| **A2C** | **高** | **中** | **低** | **必須** | **✅ 採用** |
| PPO | **最高** | 高 | 中 | 必須 | 実験用 |
| REINFORCE+Baseline | 低 | 低 | **最低** | 不要 | 非推奨 |

#### **✅ 確定採用**: Option 4-A (A2C)

**デフォルト設定として確定**:
- Advantage Actor-Critic (A2C) をベースアルゴリズムとして実装
- シンプルで安定した学習が可能
- デバッグ・検証が容易

#### **実装方針**:

1. **デフォルト**: Option 4-A (A2C)
   - Actor-Critic両方を学習
   - Advantageで分散削減

2. **実験用オプション**:
   - Option 4-B (PPO): 将来的な拡張用
   - Option 4-C (REINFORCE): 既存GCN比較用のみ

---

### **決定ポイント #5: 教師あり事前学習の扱い** ✅ **確定**

**問題**: GCNの2段階訓練（教師あり→RL）を継承するか？

#### **Option 5-A: 2段階訓練** ✅ **採用**

```python
# Phase 1: Supervised pre-training
# - 最適解データでActor初期化
# - voc_edges_out=2 (binary classification)

# Phase 2: RL fine-tuning
# - Phase 1のActorを初期値としてRL学習
# - Criticはランダム初期化
```

**メリット**:
- ✅ **強力な初期化**（最適解の知識）
- ✅ RL学習の高速化・安定化
- ✅ 既存GCNの資産活用

**デメリット**:
- ❌ 最適解データが必要
- ❌ 訓練時間増加

**採用理由**:
- 最適解データを活用することで学習を大幅に加速
- Actor部分のみ事前学習、Criticはランダム初期化でRL学習
- 既存GCN資産を最大限活用

---

#### **Option 5-B: RLのみ（Teal方式）** - 実験用

```python
# ランダム初期化からRL学習のみ
```

**メリット**:
- ✅ 最適解データ不要
- ✅ End-to-end RL

**デメリット**:
- ❌ 学習時間長い
- ❌ 収束不安定

**位置づけ**:
- 最適解データがない場合の代替案
- 事前学習の効果を検証する比較実験用

---

### **決定ポイント #6: モデル構造の詳細** ✅ **確定**

#### **決定ポイント #6-A: GNN層数とhidden_dim** ✅ **確定**

**問題**: Residual Gated GCNのハイパーパラメータ

**既存設定**:
- GCN supervised: `num_layers=10, hidden_dim=128`
- GCN RL: `num_layers=5, hidden_dim=64`

**✅ 確定設定**:
```json
{
  "num_layers": 8,
  "hidden_dim": 128,
  "mlp_layers": 3,
  "aggregation": "mean",
  "dropout_rate": 0.3
}
```

**採用理由**:
- 動的状態を扱うため表現力必要（hidden_dim=128）
- 計算コストとのバランス（num_layers=8）
- Supervisedより少し軽量、RL単体より強力

---

#### **決定ポイント #6-B: Policy/Value Head構造** ✅ **確定**

**問題**: MLP層数と活性化関数

**✅ 確定設定**:
```python
# Policy Head
PolicyHead:
  policy_head_mlp_layers: 2
  Input: hidden_dim * 2 (context)
  MLP: [256, 128, num_nodes]
  Activation: ReLU
  Output: Softmax

# Value Head
ValueHead:
  value_head_mlp_layers: 3
  Input: hidden_dim * num_nodes * num_commodities
  MLP: [512, 256, 64, 1]
  Activation: ReLU
  Output: Linear (unbounded)
```

**採用理由**:
- Policy Head: 2層MLP（シンプルで効率的）
- Value Head: 3層MLP（複雑な価値関数の表現力確保）
- ReLU活性化（標準的で安定）

---

## 実装ロードマップ

### **Phase 1: 基本実装（2-3週間）**

#### **Week 1: モデル構築**
- [ ] `HybridGNNEncoder` 実装
  - 動的エッジ使用量の埋め込み追加
  - Residual Gated GCN層の再利用
- [ ] `PolicyHead` 実装
  - **両方のアクション空間を実装** (`action_type='node'` / `'edge'`)
  - デフォルト: `action_type='node'`
- [ ] `ValueHead` 実装（Option 2-B-1）
- [ ] `HybridActorCritic` 統合
  - 設定ファイルからの`action_type`読み込み

**確定事項**:
- ✅ **決定ポイント #1**: エンコーダ共有 → **Option 1-A（共有）確定**
- ✅ **決定ポイント #2-A**: アクション空間 → **両方実装、デフォルト='node'**
- ✅ **決定ポイント #2-B**: Value粒度 → **Option 2-B-1（グローバル）確定**
- ✅ **決定ポイント #3**: GNN更新頻度 → **Option 3-B（Per-Commodity）確定**
- ✅ **決定ポイント #4**: RLアルゴリズム → **Option 4-A（A2C）確定**
- ✅ **決定ポイント #5**: 事前学習 → **Option 5-A（2段階訓練）確定**
- ✅ **決定ポイント #6**: モデル構造 → **確定（詳細は上記参照）**

---

#### **Week 2: Rolloutエンジン**
- [ ] `SequentialRolloutEngine` 実装
  - ✅ **Per-Commodity GNN更新（Option 3-B）**
  - Top-p sampling
  - エッジ使用量トラッキング
- [ ] 単体テスト（経路生成確認）

---

#### **Week 3: RL訓練**
- [ ] `HybridRLStrategy` 実装
  - ✅ **A2Cアルゴリズム（Option 4-A）**
- [ ] 報酬関数統合（GCN連続関数）
- [ ] 訓練ループ実装
- [ ] メトリクス記録（`MetricsLogger`拡張）

---

### **Phase 2: 教師あり事前学習統合（1-2週間）**

#### **Week 4-5: 2段階訓練パイプライン**
- [ ] 既存教師あり訓練コードの適応
  - ✅ **Actor部分のみ事前学習（Option 5-A）**
  - Criticはランダム初期化
- [ ] モデル変換ユーティリティ
  - `supervised → RL (voc_edges_out: 2→1)`拡張
- [ ] `two_phase_training.py` 更新

**確定事項**:
- ✅ **決定ポイント #5**: 事前学習 → **Option 5-A（2段階訓練）確定**

---

### **Phase 3: 実験的拡張（オプション）**

#### **オプション: GNN更新頻度の比較実験**
- [ ] Option 3-A (Never) の性能評価
  - 計算時間比較
  - Load factor比較
- [ ] Option 3-C (Per-Step) の性能評価
  - GPU最適化
  - コスト対効果分析

#### **オプション: PPO移行実験**
- [ ] PPO損失関���実装
- [ ] 複数epoch更新
- [ ] Importance sampling
- [ ] ハイパーパラメータ調整
  - `ppo_epsilon`, `ppo_epochs`

---

### **Phase 4: 最適化と評価（2週間）**

#### **Week 6: 性能最適化**
- [ ] ハイパーパラメータチューニング
  - Grid search / Bayesian optimization
- [ ] データ拡張
- [ ] モデルアンサンブル（オプション）

#### **Week 7: 総合評価**
- [ ] ベンチマーク実験
  - GCN単体 vs Teal vs ハイブリッド
- [ ] 論文執筆用結果収集
- [ ] ドキュメント整備

---

## 期待される効果

### **定量的改善目標**

| 指標 | GCN単体 | Teal | ハイブリッド目標 |
|------|---------|------|-----------------|
| 最大負荷率 (Load Factor) | 0.85 | 0.80 | **0.75** |
| 訓練時間 (epoch) | 50 | 100 | **60** |
| 推論時間 (per instance) | 0.5s | 2.0s | **0.8s** |
| 実行可能解率 | 92% | 98% | **95%** |
| GPU メモリ使用量 | 4GB | 2GB | **5GB** |

### **定性的改善期待**

1. **学習安定性**: Criticによる分散削減
2. **探索効率**: 動的GNN更新による適応的学習
3. **汎化性能**: 教師あり事前学習 + RL微調整
4. **モジュール性**: Policy/Value Head分離

---

## 設定ファイル例

```json
{
  "expt_name": "hybrid_gcn_teal",
  "gpu_id": "0",
  "use_gpu": true,

  "_comment_model": "=== Model Architecture ===",
  "num_nodes": 14,
  "num_commodities": 5,
  "hidden_dim": 128,
  "num_layers": 8,
  "mlp_layers": 3,
  "aggregation": "mean",
  "dropout_rate": 0.3,

  "_comment_action": "=== Action Space ===",
  "action_type": "node",
  "_action_type_options": ["node", "edge"],
  "_action_type_note": "node: ノードレベルアクション (デフォルト), edge: エッジレベルアクション",
  "policy_head_mlp_layers": 2,

  "_comment_value": "=== Value Network ===",
  "value_head_mlp_layers": 3,
  "value_type": "global",

  "_comment_rollout": "=== Sequential Rollout ===",
  "gnn_update_frequency": "per_commodity",
  "_gnn_update_options": ["never", "per_commodity", "per_step"],
  "_gnn_update_note": "per_commodity: デフォルト（確定）",
  "sampling_temperature": 1.0,
  "sampling_top_p": 0.9,

  "_comment_rl": "=== RL Algorithm ===",
  "rl_algorithm": "a2c",
  "_rl_options": ["a2c", "ppo", "reinforce_baseline"],
  "entropy_weight": 0.01,
  "value_loss_weight": 0.5,
  "gamma": 0.99,

  "_comment_ppo": "=== PPO Specific ===",
  "ppo_epsilon": 0.2,
  "ppo_epochs": 4,

  "_comment_reward": "=== Reward Function ===",
  "reward_function_version": "continuous_v1",
  "_reward_formula": "2.0 - 2.0 * load_factor",

  "_comment_training": "=== Training ===",
  "two_phase_training": true,
  "supervised_config": "configs/gcn/supervised_pretraining.json",
  "max_epochs": 50,
  "batch_size": 32,
  "learning_rate": 0.0005,
  "decay_rate": 1.2,

  "_comment_data": "=== Data ===",
  "num_train_data": 1000,
  "num_val_data": 200,
  "num_test_data": 200
}
```

---

## 決定事項サマリー

### **✅ 全決定ポイント確定**

| # | 決定ポイント | 確定選択 | 理由 |
|---|------------|---------|------|
| **#1** | エンコーダ共有 | **Option 1-A: 共有** | パラメータ効率、A2C標準 |
| **#2-A** | アクション空間 | **両方実装、デフォルト='node'** | 柔軟性確保、ノードレベル優先 |
| **#2-B** | Value粒度 | **Option 2-B-1: グローバル** | Load factor最小化と整合 |
| **#3** | GNN更新頻度 | **Option 3-B: Per-Commodity** | 動的性と計算コストのバランス |
| **#4** | RLアルゴリズム | **Option 4-A: A2C** | シンプルで安定、デバッグ容易 |
| **#5** | 事前学習 | **Option 5-A: 2段階訓練** | 最適解データ活用、学習加速 |
| **#6-A** | GNN構造 | **layers=8, hidden=128** | 表現力と効率のバランス |
| **#6-B** | Head構造 | **Policy=2層, Value=3層** | 適切な複雑度 |

### **実験用オプション（性能向上時）**

以下は基本実装で性能が不十分な場合に検討：

| オプション | 目的 | 追加コスト | 詳細 |
|-----------|------|----------|------|
| Per-Step GNN | 最大動的性 | +200% | [改善提案](hybrid_improvement_proposals.md#提案1) |
| PPO | 学習安定化 | +20% | [改善提案](hybrid_improvement_proposals.md#提案2) |
| Entropy調整 | 探索最適化 | 0% | [改善提案](hybrid_improvement_proposals.md#提案3) |

---

## 次のステップ

### **実装開始の準備**

全決定ポイントが確定したため、実装を開始できます：

1. **ディレクトリ構成**
   ```
   src/hybrid/
   ├── models/
   │   ├── hybrid_gnn_encoder.py
   │   ├── policy_head.py
   │   ├── value_head.py
   │   └── hybrid_actor_critic.py
   ├── training/
   │   ├── hybrid_rl_strategy.py
   │   └── sequential_rollout.py
   └── utils/
       └── model_converter.py

   configs/hybrid/
   └── hybrid_base.json
   ```

2. **既存コードの再利用**
   - `src/gcn/models/gcn_layers.py` → そのまま利用
   - `src/gcn/training/reinforcement_strategy.py` → A2Cベースを拡張
   - `src/gcn/algorithms/path_sampler.py` → 参考実装
   - `src/gcn/utils/model_converter.py` → 拡張

3. **実装順序**
   - **Phase 1 (Week 1-3)**: 基本実装
     - HybridGNNEncoder, PolicyHead, ValueHead
     - SequentialRolloutEngine (Per-Commodity)
     - HybridRLStrategy (A2C)
   - **Phase 2 (Week 4-5)**: 教師あり事前学習統合
     - 2段階訓練パイプライン
     - モデル変換ユーティリティ

4. **性能評価後の対応**
   - 性能不足 → [改善提案ドキュメント](hybrid_improvement_proposals.md)参照
   - 目標達成 → 実験・論文執筆

---

**最終更新**: 2025-11-01
**ステータス**: ✅ 設計完了、全決定ポイント確定、実装開始可能
