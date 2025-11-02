# GCN vs Teal: アプローチの比較分析

## 概要

本ドキュメントでは、グラフ最大負荷率最小化問題に対する2つのアプローチを比較します：
- **既存GCNアプローチ** (`src/gcn/`)
- **Teal (インポート済み)** (`teal/`)

両者とも強化学習（RL）を用いてネットワークルーティング問題を解決しますが、アーキテクチャと学習戦略に重要な違いがあります。

---

## 問題設定

**目的**: ネットワークグラフにおいて、複数のcommodity（需要）をルーティングし、**最大負荷率（Maximum Load Factor）を最小化**する経路を見つける。

- **入力**: グラフ構造、エッジ容量、commodity情報（src, dst, demand）
- **出力**: 各commodityに対する経路
- **制約**: エッジ容量制約（実行可能性）

---

## アーキテクチャの比較

### 1. モデル構造

#### **既存GCN (`src/gcn/models/gcn_model.py`)**

```
入力 → Embedding → ResidualGatedGCN層 (複数) → MLP → 出力
```

- **GNNエンコーダ**: Residual Gated GCN（5-10層）
  - ノード埋め込み + commodity埋め込み + エッジ容量埋め込み
  - 各層で残差接続とゲート機構を使用

- **出力形式**: `[Batch, Nodes, Nodes, Commodities]`
  - **全ノード対の確率行列を一度に生成**
  - `voc_edges_out=1`: エッジスコア（RL用）
  - `voc_edges_out=2`: バイナリ分類（教師あり学習用）

#### **Teal (`teal/lib/FlowGNN.py`, `teal/lib/teal_actor.py`)**

```
入力 → FlowGNN → Policy Head (mean/std) → 行動サンプリング
```

- **GNNエンコーダ**: FlowGNN
  - GNN層（容量制約をキャプチャ）とDNN層（需要制約をキャプチャ）を交互に配置
  - スキップ接続により各層の出力を結合

- **Policy Head**: 独立した線形層
  - 入力: FlowGNNの埋め込み `[num_path*(num_layer+1)]`
  - 出力: `mean` と `std`（正規分布のパラメータ）
  - **各commodityに対して独立した確率分布を生成**

**重要な違い**:
- GCN: 全ノード対の巨大な確率行列を生成（`[V×V×C]`次元）
- Teal: 事前計算された経路候補に対する確率分布のみ（`[num_path]`次元）

---

### 2. 経路生成メカニズム

#### **既存GCN: Batch Path Generation**

**方式**: ビームサーチまたはTop-pサンプリング

```python
# src/gcn/algorithms/path_sampler.py
# または src/gcn/algorithms/beamsearch_uelb.py

1. GCNが全エッジの確率を一度に出力
2. ビームサーチまたはサンプリングで全commodityの経路を並列生成
3. 経路完成後に実行可能性をチェック
```

**特徴**:
- **一括決定**: 全commodityの経路を並列に構築
- **静的**: GNN出力は1回の順伝播で固定
- **探索空間**: ビームサイズで制御（`beam_size=1280`）

**実装** (`src/gcn/algorithms/path_sampler.py:51-212`):
```python
def sample(self):
    for b in range(self.batch_size):
        commodity_paths = []
        for commodity in commodities:
            src, dst = commodity[0], commodity[1]
            path = self._sample_single_path(src, dst, ...)
            commodity_paths.append(path)
        batch_paths.append(commodity_paths)
```

#### **Teal: Sequential Rollout (逐次決定型)**

**方式**: 環境とのインタラクション

```python
# teal/lib/teal_env.py:147-186

1. 環境から観測（capacity + traffic matrix）を取得
2. FlowGNN → Policy Head でアクションサンプリング
3. 環境にアクション送信 → 報酬受領
4. 次の観測を取得（状態更新）
5. 繰り返し
```

**特徴**:
- **逐次決定**: 環境からの観測 → アクション → 報酬のループ
- **動的**: 各ステップで観測が更新される可能性
- **インタラクティブ**: 環境の状態変化を反映

**実装** (`teal/lib/teal_model.py:64-71`):
```python
# 訓練ループ
obs = self.env.get_obs()  # 観測取得
raw_action, log_probability = self.actor.evaluate(obs)  # アクション
reward, info = self.env.step(raw_action, num_sample=num_sample)  # 報酬
loss += -(log_probability*reward).mean()
```

**注**: Tealは事前計算された経路候補に対する分割比（split ratio）を出力し、ADMMで最適化するアプローチです。

---

### 3. 強化学習の実装

#### **既存GCN: REINFORCE with Baseline**

**アルゴリズム**: Policy Gradient (REINFORCE)

```python
# src/gcn/training/reinforcement_strategy.py:410-440

Policy Gradient Loss = -Σ log_prob(path) × advantage
advantage = reward - baseline
baseline = EMA(rewards)  # 移動平均ベースライン
```

**特徴**:
- **Baseline**: 単純な移動平均（スカラー）
  - `baseline = momentum × baseline + (1-momentum) × reward`
  - デフォルト: `rl_baseline_momentum=0.95`

- **Advantage正規化**: オプションで有効化
  - `normalized_advantage = (advantage - mean) / (std + eps)`

- **エントロピー正則化**: 探索促進
  - `loss = policy_loss - entropy_weight × entropy`

**実装** (`src/gcn/training/reinforcement_strategy.py:378-413`):
```python
# Baseline更新
if self.reward_baseline is None:
    self.reward_baseline = rewards.mean().item()

advantages = rewards - self.reward_baseline
self.reward_baseline = (
    self.baseline_momentum * self.reward_baseline +
    (1 - self.baseline_momentum) * rewards.mean().item()
)

# Advantage正規化
if self.normalize_advantages:
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
```

#### **Teal: 環境ベースの報酬計算**

**アルゴリズム**: Policy Gradient + COMA風の報酬

```python
# teal/lib/teal_model.py:71

loss = -(log_probability*reward).mean()
```

**特徴**:
- **環境からの報酬**: `TealEnv.step()` が報酬を計算
  - 目的関数: `total_flow` または `max_link_utilization`
  - ADMM最適化後の結果を報酬として使用

- **Value Network (Critic)**: **コード内に未実装**
  - TealActorにはPolicy（Actor）のみ存在
  - Criticネットワークは見当たらない

**実装** (`teal/lib/teal_env.py:147-186`):
```python
def step(self, raw_action, num_sample=0, num_admm_step=0):
    # Split ratioをアクションとして受け取る
    # ADMMで最適化
    # 報酬計算（目的関数値）
    return reward, info
```

---

## 指摘された3つの違いの検証

### **① Policy Headの明確化**

**検証結果**: ✅ **違いとして適切**

- **GCN**:
  - MLPが直接全ノード対のスコアを出力 (`[B, V, V, C]`)
  - Policy Headという明確な分離はない
  - GNN出力 → MLP → エッジ確率（一体化）

- **Teal**:
  - FlowGNN（エンコーダ）と Policy Head（線形層）が明確に分離
  - Policy Headは `mean_linear` と `log_std_linear` で正規分布パラメータを出力
  - `teal/lib/teal_actor.py:49-56`

**影響**:
- **GCN**: 出力次元が大きい（V×V×C）→ パラメータ数増加
- **Teal**: 出力次元が小さい（num_path）→ 計算効率的

**GCNへの統合方法**:
```python
# 現在
x, e = ResidualGatedGCN(...)
y_pred_edges = MLP(e)  # [B, V, V, C]

# Policy Head追加案
embeddings = ResidualGatedGCN(...)  # [B, V, H]
policy_mean = PolicyHead(embeddings)  # [B, V, num_actions]
```

---

### **② 逐次決定型Rollout (Sequential Decision)**

**検証結果**: ⚠️ **部分的に適切だが、実装の詳細が異なる**

**GCN**:
- `PathSampler` は1 hopずつサンプリングしている（`_sample_single_path`）
- しかし、**GNN出力は固定**（ステップごとに再計算しな���）
- `src/gcn/algorithms/path_sampler.py:272-396`

```python
def _sample_single_path(self, src, dst, ...):
    current = src
    while current != dst:
        # 現在ノードからの出力エッジ確率を取得（固定されたGNN出力から）
        outgoing_probs = edge_probs[current, :].clone()
        next_node = self._top_p_sample(outgoing_probs)
        path.append(next_node)
        current = next_node
```

**Teal**:
- 環境とのインタラクション（`env.step()`）で**観測を更新**
- 各ステップで新しい観測に基づいてアクションを決定
- ただし、**経路を1 hopずつ構築するわけではない**
  - 事前計算された経路候補の分割比を決定
  - `teal/lib/teal_env.py:147`

**結論**:
- 「逐次決定」の意味が異なる:
  - **GCN**: 固定GNN出力から1 hopずつ経路サンプリング
  - **Teal**: 環境状態を更新しながらアクション決定（経路レベル）

- **GNN再計算の有無**が本質的な違い:
  - GCN: GNNは1回のみ（静的）
  - Teal: 観測更新あり（動的）

**違いとしての適切性**: ✅ ただし、「1 hopずつ」という表現は誤解を招く可能性あり

---

### **③ Value Network (Critic)の追加**

**検証結果**: ❌ **Tealには実装されていない**

**調査結果**:
- `TealActor` には **Policyネットワーク（Actor）のみ**
- Criticネットワーク（Value Function）は**存在しない**
- `teal/lib/teal_actor.py` 全体を確認

**Tealの実装**:
```python
# teal/lib/teal_actor.py:95-145
class TealActor(nn.Module):
    def forward(self, feature):
        mean = self.mean_linear(x)
        std = self.log_std_linear(x)  # オプション
        return mean, std  # Policy出力のみ
```

**GCNの実装**:
- GCNも **Baselineのみ**（移動平均スカラー）
- Criticネットワークは**未実装**

**結論**:
- この指摘は**不正確**
- 両者ともCriticネットワークを持たない
- GCN: 移動平均baseline
- Teal: Baselineなし（環境からの生報酬を使用）

**違いとしての適切性**: ❌ **両者ともValue Networkを持たない**

---

## 実際の主要な違い

### **A. 出力表現の次元**

| アプローチ | 出力次元 | 説明 |
|-----------|---------|------|
| GCN | `[B, V, V, C]` | 全ノード対×全commodity |
| Teal | `[num_path]` | 事前計算経路の分割比 |

### **B. 経路候補の生成方法**

| アプローチ | 方法 | タイミング |
|-----------|------|-----------|
| GCN | GNN出力からサンプリング/ビームサーチ | 推論時（動的） |
| Teal | 最短経路を事前計算 | 事前処理（静的） |

**Tealの事前計算** (`teal/lib/path_utils.py`):
- `find_paths()`: Dijkstraで最短k経路を計算
- `edge_disjoint`: エッジ素経路オプション
- これらの経路候補に対する分割比を学習

### **C. 報酬関数の設計**

#### **GCN** (`src/gcn/training/reinforcement_strategy.py:344-376`)

```python
# 連続報酬関数 (2025-10-22更新)
reward = 2.0 - 2.0 * load_factor

# 不完全パス: -0.5 ~ -2.0
# 無効エッジ: -1.0
# 実行可能: load_factor=0.0 → +2.0
#          load_factor=1.0 → 0.0
#          load_factor=2.0 → -2.0
```

**特徴**:
- 連続的（C0連続）
- load_factor=1.0で実行可能/不可能の境界
- 勾配が明確

#### **Teal** (`teal/lib/teal_env.py`)

```python
# 環境が計算する報酬
if obj == 'total_flow':
    reward = satisfied_demand
elif obj == 'max_link_utilization':
    reward = -max_utilization
```

**特徴**:
- ADMM最適化後の目的関数値を使用
- 問題依存の報酬設計

### **D. 訓練パイプライン**

#### **GCN**
```
データ生成 → GCN推論 → サンプリング/ビームサーチ →
報酬計算 → Policy Gradient → 勾配更新
```

- **2段階訓練**: 教師あり事前学習 → RL微調整
- `scripts/gcn/two_phase_training.py`

#### **Teal**
```
環境リセット → 観測取得 → Policy → アクション →
環境ステップ → 報酬 → 勾配更新
```

- **RL訓練のみ**
- 環境中心のループ

---

## アプローチの長所・短所

### **既存GCN**

**長所**:
✅ **柔軟な経路生成**: 事前計算不要、任意のグラフに対応
✅ **教師あり学習の活用**: 最適解から事前学習可能
✅ **End-to-End学習**: グラフ → 経路の直接学習
✅ **探索空間の広さ**: ビームサーチで多様な経路探索

**短所**:
❌ **出力次元の大きさ**: `V×V×C` の確率行列
❌ **計算コスト**: GNN順伝播 + ビームサーチ
❌ **サンプリングの非効率**: 無効な経路を生成する可能性
❌ **静的なGNN出力**: ステップごとに再計算しない

### **Teal**

**長所**:
✅ **計算効率**: 事前計算経路 + 小次元出力
✅ **ADMM最適化**: 制約満足を後処理で保証
✅ **環境とのインタラクション**: 動的な状態更新
✅ **メモリ効率**: 小さいアクション空間

**短所**:
❌ **事前計算への依存**: グラフ変更時に再計算必要
❌ **経路候補の制限**: k本の最短経路のみ
❌ **ADMMの計算コスト**: 推論時のオーバーヘッド
❌ **汎化性の制約**: トポロジ固有の学習

---

## 統合の可能性

### **GCNにTealの要素を導入する場合**

#### **1. Policy Headの明確化**
```python
class GCNWithPolicyHead(nn.Module):
    def __init__(self, ...):
        self.encoder = ResidualGatedGCN(...)
        self.policy_head = PolicyHead(hidden_dim, num_actions)

    def forward(self, x):
        embeddings = self.encoder(x)  # [B, V, H]
        action_probs = self.policy_head(embeddings)  # [B, V, A]
        return action_probs
```

**利点**: モジュール分離、再利用性向上

#### **2. 動的GNN更新（逐次決定）**
```python
# 各ステップでGNN再計算
for commodity in commodities:
    # 現在のネットワーク状態でGNN更新
    embeddings = self.encoder(current_state)
    next_hop_probs = self.policy_head(embeddings, current_node)
    next_node = sample(next_hop_probs)
    current_state = update_capacity(current_state, edge_usage)
```

**課題**: 計算コスト増大（GNN順伝播 × ステップ数）

#### **3. Value Networkの追加**
```python
class ActorCriticGCN(nn.Module):
    def __init__(self, ...):
        self.actor = GCNWithPolicyHead(...)
        self.critic = ValueNetwork(...)  # V(s)推定

    def forward(self, x):
        action_probs = self.actor(x)
        state_value = self.critic(x)
        return action_probs, state_value
```

**利点**: 分散削減、学習安定化（A2C/PPO）

---

## 結論

### **3つの指摘の検証結果**

| 指摘 | 適切性 | 詳細 |
|-----|-------|------|
| ① Policy Headの明確化 | ✅ 適切 | Tealは分離設計、GCNは一体化 |
| ② 逐次決定型Rollout | ⚠️ 部分的 | 両者とも逐次だが、GNN再計算の有無が違い |
| ③ Value Network | ❌ 不適切 | 両者ともCriticネットワークなし |

### **実際の主要な違い**

1. **出力表現**: GCN（全ノード対）vs Teal（事前計算経路）
2. **経路生成**: GCN（動的サンプリング）vs Teal（静的経路+分割比）
3. **報酬設計**: GCN（連続load_factor）vs Teal（ADMM最適化後）
4. **訓練戦略**: GCN（教師あり→RL）vs Teal（RL のみ）

### **推奨事項**

既存GCNの改善方向:
- ✅ **Policy Head分離**: モジュール性向上
- ⚠️ **動的GNN更新**: コスト対効果を検証
- ✅ **Critic追加**: A2C/PPOアーキテクチャへの移行検討

Tealの活用方法:
- 事前計算経路の補助情報として使用
- ハイブリッドアプローチ（GCN経路 + ADMM最適化）

---

**最終更新**: 2025-11-01
**対応バージョン**: GCN (feature/supervised_pretrained), Teal (imported)
