# ハイブリッドアプローチ改善提案
## 性能向上のための段階的改善戦略

**作成日**: 2025-11-01
**対象**: ハイブリッドGCN+Teal手法の性能が期待に届かなかった場合の改善案
**前提**: 基本実装（Per-Commodity GNN更新 + A2C）が完了済み

---

## 目次

1. [改善提案の全体像](#改善提案の全体像)
2. [提案1: GNN更新頻度の動的化](#提案1-gnn更新頻度の動的化)
3. [提案2: RLアルゴリズムの段階的強化](#提案2-rlアルゴリズムの段階的強化)
4. [提案3: 探索制御の最適化](#提案3-探索制御の最適化)
5. [実装優先順位](#実装優先順位)

---

## 改善提案の全体像

### **改善の方向性**

基本実装で以下の問題が発生した場合の対処法：

| 問題 | 症状 | 推奨改善案 |
|------|------|-----------|
| **性能不足** | Load factorが目標に届かない | 提案1: Per-Step GNN更新 |
| **学習不安定** | 報酬が収束しない、振動する | 提案2: PPO導入 |
| **探索不足** | 局所解に収束、多様性不足 | 提案3: Entropy調整 |
| **過学習** | 訓練データに過適合 | 提案3: Entropy強化 + Dropout |

### **段階的改善フロー**

```
基本実装 (Per-Commodity + A2C)
    ↓
【問題診断】
    ↓
┌─────────────┬─────────────┬─────────────┐
│  性能不足    │  学習不安定  │  探索不足    │
│  ↓          │  ↓          │  ↓          │
│ 提案1       │ 提案2       │ 提案3       │
│ Per-Step    │ PPO        │ Entropy調整 │
└─────────────┴─────────────┴─────────────┘
```

---

## 提案1: GNN更新頻度の動的化

### **Option 3-C: Per-Step（Teal方式、最も動的）の有効性**

#### **概要**

現在の実装（Per-Commodity更新）から、**各ステップ毎にGNNを再計算**する方式に変更します。

```python
# 現在: Per-Commodity更新
for c in range(num_commodities):
    node_feat, edge_feat, _ = self.model.encoder(state)  # C回のGNN計算
    path = self._sample_path(node_feat, edge_feat, c)
    self._update_edge_usage(state, path, demand[c])

# 提案: Per-Step更新
for c in range(num_commodities):
    path = []
    current = src[c]
    while current != dst[c]:
        node_feat, edge_feat, _ = self.model.encoder(state)  # 各ステップでGNN計算
        next_node = self._sample_next_hop(node_feat, edge_feat, current, dst[c])
        path.append(next_node)
        self._update_edge_usage(state, current, next_node, demand[c])  # 即座に反映
        current = next_node
```

#### **有効性の根拠**

| 観点 | Per-Commodity (現在) | Per-Step (提案) |
|------|---------------------|----------------|
| **GNN計算回数** | O(C) | O(C × L) |
| **状態の動的性** | Commodity単位 | **ステップ単位（最も細かい）** |
| **エッジ使用量反映** | Commodity完了後 | **各ホップ直後** |
| **表現力** | 中 | **高（最も現実的）** |
| **計算コスト** | 中 | 高（L: 平均経路長） |

**期待される効果**:
- ✅ **リアルタイム状態反映**: 各決定が即座にネットワーク状態に反映される
- ✅ **より正確な負荷予測**: エッジ使用量が常に最新の状態で考慮される
- ✅ **Commodity間の相互作用**: 先行commodityの影響を後続commodityがより細かく認識
- ✅ **複雑な制約への対応**: 容量制約違反の早期検出・回避

#### **実装上の注意点**

**メリット**:
- Tealの元設計に最も近い（実績あり）
- 最も動的で表現力が高い
- デバッグがステップ単位で可能

**デメリット**:
- 計算コスト大（GNN順伝播が10-20倍に増加）
- GPU最適化が必須
- メモリ使用量増加

#### **導入判断基準**

以下の条件を**すべて満たす場合**に導入を検討：

1. ✅ Per-Commodityで性能が頭打ち（Load factor改善 < 5%）
2. ✅ GPU性能が十分（V100以上推奨）
3. ✅ 訓練時間の増加を許容できる（2-3倍）
4. ✅ 推論速度よりも性能を優先

#### **実装ステップ**

**難易度**: ★★★☆☆（中）

```python
# ステップ1: SequentialRolloutEngineの修正
class SequentialRolloutEngine:
    def __init__(self, model, config):
        self.gnn_update_frequency = config.get('gnn_update_frequency', 'per_commodity')

    def rollout(self, batch_data, mode='train'):
        if self.gnn_update_frequency == 'per_step':
            return self._rollout_per_step(batch_data, mode)
        else:
            return self._rollout_per_commodity(batch_data, mode)

    def _rollout_per_step(self, batch_data, mode):
        """各ステップでGNNを再計算"""
        state = self._initialize_state(batch_data)
        all_paths = []

        for c in range(num_commodities):
            path = []
            current = src[c]

            while current != dst[c]:
                # 【重要】各ステップでGNN再エンコード
                node_feat, edge_feat, _ = self.model.encoder(
                    state['x_nodes'],
                    state['x_commodities'],
                    state['x_edges_capacity'],
                    state['x_edges_usage']  # 最新の使用量
                )

                # 次ホップをサンプリング
                next_node, log_prob, entropy = self._sample_next_hop(
                    node_feat, edge_feat, current, dst[c], c
                )
                path.append(next_node)

                # 【即座に状態更新】
                self._update_edge_usage_step(
                    state['x_edges_usage'],
                    current,
                    next_node,
                    demand[c]
                )

                current = next_node

            all_paths.append(path)

        return all_paths
```

**設定ファイル変更**:
```json
{
  "gnn_update_frequency": "per_step",
  "_note": "高性能GPU必須、計算時間2-3倍増加"
}
```

**GPU最適化**:
```python
# バッチ並列化でコスト削減
# 同一バッチ内の複数commodityを並列処理
node_feat = self.model.encoder(state)  # [B, V, C, H]
# → 複数commodityの現在ノードを同時に処理
```

---

## 提案2: RLアルゴリズムの段階的強化

### **段階的改善戦略**

基本実装のA2Cから、より安定したPPOへ段階的に移行します。

| フェーズ | 手法 | 目的 | 難易度 |
|---------|------|------|--------|
| 🥇 **ステップ1** | **A2C** | 安定学習を確立。ハイパラや損失動作を確認 | ★★☆☆☆ |
| 🥈 **ステップ2** | **PPO** | 学習効率を上げる。報酬が安定してから導入 | ★★★☆☆ |
| 🥉 **ステップ3** | **PPO + Entropy調整** | 探索を制御して性能向上 | ★★★★☆ |

---

### **🥇 ステップ1: A2C（基本実装）**

#### **目的**
- 安定した学習の確立
- ハイパーパラメータの調整
- 損失関数の動作確認

#### **実装内容**

```python
class HybridRLStrategy:
    def compute_loss(self, batch_paths, batch_rewards):
        """A2C損失関数"""
        # Actor損失: Policy Gradient with Advantage
        state_values = self.model.critic(state_features)  # [B]
        advantages = batch_rewards - state_values.detach()  # [B]

        actor_loss = -(log_probs * advantages).mean()

        # Critic損失: MSE
        critic_loss = F.mse_loss(state_values, batch_rewards)

        # Entropy正則化（探索促進）
        entropy_loss = -entropy.mean()

        total_loss = (
            actor_loss +
            self.value_loss_weight * critic_loss +
            self.entropy_weight * entropy_loss
        )

        return total_loss
```

#### **チューニングポイント**

| パラメータ | 推奨範囲 | 効果 |
|-----------|---------|------|
| `learning_rate` | 0.0001 - 0.001 | 学習速度 |
| `value_loss_weight` | 0.5 - 1.0 | Critic重視度 |
| `entropy_weight` | 0.001 - 0.01 | 探索度合い |
| `gamma` | 0.95 - 0.99 | 将来報酬割引 |

#### **成功の指標**

- ✅ 訓練損失が単調減少
- ✅ 報酬が収束（振動幅 < 10%）
- ✅ Load factorが改善傾向

---

### **🥈 ステップ2: PPO（学習効率向上）**

#### **目的**
- 学習効率の向上
- 大きなポリシー更新の防止
- より安定した収束

#### **導入タイミング**

以下の条件を満たした場合に導入：

1. ✅ A2Cで基本性能が確認できた
2. ✅ 学習が不安定（報酬が振動）
3. ✅ サンプル効率を上げたい

#### **実装内容**

```python
class HybridRLStrategy:
    def __init__(self, config):
        self.rl_algorithm = config.get('rl_algorithm', 'a2c')
        self.ppo_epsilon = config.get('ppo_epsilon', 0.2)
        self.ppo_epochs = config.get('ppo_epochs', 4)

    def compute_loss_ppo(self, batch_paths, batch_rewards, old_log_probs):
        """PPO損失関数（Clipped Objective）"""
        # 新しいポリシーで再評価
        new_log_probs, entropy, state_values = self.evaluate_actions(batch_paths)

        # Importance sampling ratio
        ratio = torch.exp(new_log_probs - old_log_probs)  # π_new / π_old

        # Advantage計算
        advantages = batch_rewards - state_values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Clipped surrogate objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.ppo_epsilon, 1.0 + self.ppo_epsilon) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()

        # Critic損失（clippingオプション）
        critic_loss = F.mse_loss(state_values, batch_rewards)

        # Entropy bonus
        entropy_loss = -entropy.mean()

        total_loss = (
            actor_loss +
            self.value_loss_weight * critic_loss +
            self.entropy_weight * entropy_loss
        )

        return total_loss

    def train_step(self, batch_data):
        """PPO: 複数epoch更新"""
        # データ収集（old policy）
        with torch.no_grad():
            batch_paths, old_log_probs, rewards = self.rollout(batch_data)

        # 複数回更新
        for epoch in range(self.ppo_epochs):
            loss = self.compute_loss_ppo(batch_paths, rewards, old_log_probs)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
```

#### **PPOハイパーパラメータ**

| パラメータ | 推奨値 | 説明 |
|-----------|--------|------|
| `ppo_epsilon` | 0.1 - 0.3 | クリッピング範囲（小さいほど安定） |
| `ppo_epochs` | 3 - 10 | 更新回数（多いほど効率的） |
| `batch_size` | 64 - 128 | PPOは大きめが推奨 |

#### **A2C vs PPO 比較**

| 観点 | A2C | PPO |
|------|-----|-----|
| **安定性** | 中 | **高** |
| **サンプル効率** | 低 | **高** |
| **実装複雑度** | 低 | 中 |
| **計算コスト** | 低 | 中（複数epoch） |
| **推奨用途** | 初期実装 | 本番実装 |

#### **難易度**: ★★★☆☆（中）

**実装時間**: 1-2日

---

### **🥉 ステップ3: PPO + Entropy調整（探索最適化）**

#### **目的**
- 探索と活用のバランス最適化
- 局所解からの脱出
- 多様な経路の発見

#### **導入タイミング**

以下の問題が発生した場合：

1. ✅ PPOで性能が頭打ち
2. ✅ 同じような経路ばかり選択される
3. ✅ 検証データで性能が低い（過学習）

#### **実装内容**

```python
class AdaptiveEntropyScheduler:
    """Entropy係数の動的調整"""
    def __init__(self, initial_entropy=0.01, min_entropy=0.001, decay_rate=0.995):
        self.entropy_weight = initial_entropy
        self.min_entropy = min_entropy
        self.decay_rate = decay_rate
        self.diversity_threshold = 0.5  # 経路多様性の閾値

    def update(self, epoch, diversity_score):
        """
        diversity_score: バッチ内の経路の多様性指標
        - 高い → 探索十分 → Entropy減少
        - 低い → 探索不足 → Entropy増加
        """
        if diversity_score < self.diversity_threshold:
            # 探索不足 → Entropy増加
            self.entropy_weight = min(self.entropy_weight * 1.1, 0.1)
        else:
            # 探索十分 → Entropy減衰
            self.entropy_weight = max(
                self.entropy_weight * self.decay_rate,
                self.min_entropy
            )

        return self.entropy_weight

class HybridRLStrategy:
    def __init__(self, config):
        self.entropy_scheduler = AdaptiveEntropyScheduler(
            initial_entropy=config.get('initial_entropy', 0.01),
            min_entropy=config.get('min_entropy', 0.001),
            decay_rate=config.get('entropy_decay_rate', 0.995)
        )

    def compute_diversity_score(self, batch_paths):
        """バッチ内の経路多様性を計算"""
        # 例: ユニークな経路の割合
        unique_paths = len(set(map(tuple, batch_paths)))
        total_paths = len(batch_paths)
        return unique_paths / total_paths

    def train_step(self, batch_data, epoch):
        # 通常のPPO更新
        loss = self.compute_loss_ppo(batch_data)

        # 多様性評価
        diversity = self.compute_diversity_score(batch_data['paths'])

        # Entropy係数を動的調整
        self.entropy_weight = self.entropy_scheduler.update(epoch, diversity)

        return loss
```

#### **Entropy調整戦略**

| 段階 | Entropy係数 | 目的 |
|------|------------|------|
| **序盤（0-20 epoch）** | 0.01 - 0.05 | 広範な探索 |
| **中盤（20-60 epoch）** | 0.005 - 0.01 | 有望な領域の探索 |
| **終盤（60+ epoch）** | 0.001 - 0.005 | 最良解への収束 |

#### **多様性指標**

```python
def compute_diversity_metrics(batch_paths):
    """複数の多様性指標"""
    metrics = {}

    # 1. ユニーク経路率
    metrics['unique_ratio'] = len(set(map(tuple, batch_paths))) / len(batch_paths)

    # 2. 平均経路長の分散
    path_lengths = [len(p) for p in batch_paths]
    metrics['length_variance'] = np.var(path_lengths)

    # 3. エッジ使用の多様性（Gini係数）
    edge_usage = compute_edge_usage_distribution(batch_paths)
    metrics['edge_diversity'] = compute_gini(edge_usage)

    return metrics
```

#### **難易度**: ★★★★☆（高）

**実装時間**: 2-3日

---

## 提案3: 探索制御の最適化

### **その他の探索促進手法**

PPOとEntropyに加えて、以下の手法も有効：

#### **3.1 Curiosity-driven Exploration**

```python
class CuriosityModule(nn.Module):
    """珍しい状態を訪れることに報酬"""
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.predictor = MLP(state_dim, state_dim, num_layers=2, hidden_dim=hidden_dim)
        self.target = MLP(state_dim, state_dim, num_layers=2, hidden_dim=hidden_dim)

    def compute_intrinsic_reward(self, state):
        """予測誤差 = 新規性"""
        predicted = self.predictor(state)
        with torch.no_grad():
            target = self.target(state)
        novelty = F.mse_loss(predicted, target, reduction='none').mean(dim=-1)
        return novelty

# 報酬に追加
total_reward = extrinsic_reward + curiosity_weight * intrinsic_reward
```

#### **3.2 ε-greedy Sampling**

```python
def sample_action_with_epsilon(self, probs, epsilon=0.1):
    """ε確率でランダム探索"""
    if random.random() < epsilon:
        # ランダムサンプリング
        return torch.multinomial(torch.ones_like(probs), 1).item()
    else:
        # ポリシーに従う
        return torch.multinomial(probs, 1).item()
```

#### **3.3 Temperature Scaling**

```python
def sample_with_temperature(self, logits, temperature=1.0):
    """
    temperature > 1.0: より探索的
    temperature < 1.0: より活用的
    """
    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=-1)
    return torch.multinomial(probs, 1)
```

---

## 実装優先順位

### **推奨実装順序**

```
Phase 1: 基本実装
  ✅ Per-Commodity GNN更新
  ✅ A2C

Phase 2: 学習安定化（問題が出たら）
  → ステップ1完了（A2Cチューニング）
  → ステップ2: PPO導入

Phase 3: 性能向上（頭打ちになったら）
  → ステップ3: Entropy調整
  → 提案1: Per-Step GNN更新（最終手段）

Phase 4: 高度な探索（必要に応じて）
  → Curiosity / ε-greedy
```

### **判断フローチャート**

```
基本実装完了
    ↓
【性能評価】
    ↓
Load factor < 0.80 かつ 学習が安定？
    ├─ No（学習不安定）→ ステップ2: PPO
    └─ Yes（安定だが性能不足）
        ↓
    探索不足が原因？
        ├─ Yes → ステップ3: Entropy調整
        └─ No（表現力不足）→ 提案1: Per-Step GNN
```

### **各手法の期待改善度**

| 手法 | 期待Load Factor改善 | 計算コスト増 | リスク |
|------|-------------------|-------------|--------|
| A2C調整 | +5% | 0% | 低 |
| PPO | +10% | +20% | 低 |
| Entropy調整 | +5-10% | 0% | 中 |
| Per-Step GNN | +15-20% | +200% | 高 |

---

## まとめ

### **基本方針**

1. **まず基本実装で限界を見極める**
   - Per-Commodity + A2C で十分な場合も多い

2. **問題に応じて段階的に改善**
   - 学習不安定 → PPO
   - 探索不足 → Entropy調整
   - 性能限界 → Per-Step GNN

3. **計算コストとのトレードオフを常に意識**
   - Per-Step GNNは最終手段

### **設定ファイル例（全機能有効）**

```json
{
  "expt_name": "hybrid_advanced",

  "_comment": "=== 改善提案の設定 ===",
  "gnn_update_frequency": "per_step",
  "rl_algorithm": "ppo",
  "ppo_epsilon": 0.2,
  "ppo_epochs": 4,

  "_comment": "=== Entropy調整 ===",
  "use_adaptive_entropy": true,
  "initial_entropy": 0.01,
  "min_entropy": 0.001,
  "entropy_decay_rate": 0.995,

  "_comment": "=== 探索促進 ===",
  "use_curiosity": false,
  "curiosity_weight": 0.1,
  "epsilon_greedy": 0.0,
  "temperature": 1.0
}
```

---

**最終更新**: 2025-11-01
**ステータス**: 改善提案完成、実装は基本実装の性能評価後に判断
