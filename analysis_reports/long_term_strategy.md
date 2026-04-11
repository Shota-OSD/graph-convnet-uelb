# RL-GCN + Teal ハイブリッド手法の実装計画

**作成日**: 2025-10-22
**ステータス**: 未実装（長期戦略）

---

## エグゼクティブサマリー

既存のResidualGatedGCNを維持し、Teal (SIGCOMM '23) のフロー最適化テクニックを統合するハイブリッドアプローチ。

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
- 制約満足率の向上（ADMMによる自動調整）
- 学習の安定化（COMA報酬推定による分散削減）
- 最適解の品質向上（フロー配分の洗練化）
- 将来的なスケール拡張の基盤

---

## ResidualGatedGCN vs Teal FlowGNN

| 項目 | ResidualGatedGCN | Teal FlowGNN |
|------|------------------|--------------|
| アーキテクチャ | Gating機構 + Residual | 単純な線形層 |
| グラフ表現 | 密な隣接行列 (B*V^2*C*H) | 疎なCOO形式 (E*H) |
| バッチ処理 | あり | なし |
| 表現力 | 高い | 低い |
| スケール | 小~中規模 | 大規模 |

### メモリ使用量比較

- 小規模 (10ノード): ResidualGatedGCN ~9MB / FlowGNN ~76KB（120倍差）
- 中規模 (50ノード): ~1.6GB / ~500KB（3,200倍差）
- 大規模 (100ノード): ~32GB / ~1MB（32,000倍差）

---

## Tealの主要技術

### 1. ADMM (Alternating Direction Method of Multipliers)
容量・需要制約違反を反復的に修正。実現可能解の生成率と最適化の収束性が向上。

### 2. COMA (Counterfactual Multi-Agent) 報酬推定
代替行動をサンプリングして反事実的ベースラインを計算し、Policy Gradientの分散を削減。

### 3. Flow Rounding
連続フロー配分を実現可能解に変換。需要制約の正規化 → 容量制約の反復削減。

---

## 実装計画

### Phase 1A: 最小実装（2週間）

既存RL-GCNと同等性能を確認するための基本的なフロー配分機能。

- `FlowAllocationLayer` — エッジスコア → パスフロー変換
- `SimpleFlowRounder` — 基本的なフロー丸め処理
- `FlowRewardComputer` — 複合報酬関数（load_factor + flow_utilization - violation_penalty）
- `HybridRLStrategy` — REINFORCE + Simple Rounding

### Phase 1B: Teal統合（2週間）

- `ADMMConstraintHandler` — ADMM反復による制約処理（rho=1.0, 5 iterations）
- `COMARewardEstimator` — 反事実的ベースラインによる分散削減

期待: 制約満足率 90% → 98%+

### Phase 2: スケール対応（2週間）

- `GraphConverter` — 密な隣接行列 ⇔ 疎なCOO形式の変換
- `EfficientPathGenerator` — Yen's algorithm によるk-shortest path生成

期待: 50ノード以上の問題に対応

---

## 評価計画

### ベースライン
既存RL-GCN、Supervised GCN、Gurobi（最適解）

### 主要指標
1. 最大負荷率（小さいほど良い）
2. 実行時間（速いほど良い）
3. 制約満足率（95%+目標）

---

## 参考文献

1. Xu et al., "Teal: Learning-Accelerated Optimization of WAN Traffic Engineering", SIGCOMM 2023
2. Boyd et al., "Distributed Optimization and Statistical Learning via ADMM", 2011
3. Foerster et al., "Counterfactual Multi-Agent Policy Gradients", AAAI 2018
