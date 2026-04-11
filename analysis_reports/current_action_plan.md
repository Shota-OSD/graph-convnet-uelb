# 強化学習 改善アクションプラン

**最終更新**: 2025-10-22
**統合元**: improvement_plan_20251021.md, reachability_mask_proposal.md, solution_20251020.md

---

## 現状の診断

### 訓練結果サマリー（50エポック時点）

| 指標 | 値 | 評価 |
|------|-----|------|
| 平均報酬 | 1.41（理想: 7~9） | 要改善 |
| Complete Paths Rate | 81.6%（理想: 95%+） | 要改善 |
| Load Factor | 0.34 | 良好 |
| Finite Solution Rate | 100% | 完璧 |
| Capacity Violation | 0.4% | ほぼゼロ |

### 特定された問題

1. **報酬が低い（平均1.41）** — 約20%のコモディティが目的地に到達できず、ペナルティが支配的
2. **報酬の変動が大きい（0.29~2.85）** — 探索と活用のバランスが悪く、ベースラインが不安定
3. **Complete Rateが停滞（79-86%）** — 不完全パスへのペナルティが弱く、モデルがリスクを取る戦略を学習
4. **エントロピー改善済みだが活用されていない** — 0.0005 → 0.05~0.15に改善したが探索過多で収束遅れ

### Reachability Maskの実験結果

静的Reachability Mask（事前計算BFS）を実装済みだが、**効果は限定的**：

| メトリクス | 実装前 | 実装後 | 変化 |
|---|---|---|---|
| Complete Rate | 82% | 83% | +1% |
| Avg Reward | 1.48 | 1.97 | +0.49 |
| Capacity Violation | 2% | 0% | 解消 |

**根本原因の再分析**: 17%の未到達は「構造的な到達不可能性」ではなく、動的な容量制約・学習不安定性・探索不足が原因。

---

## 改善策

### Phase 1: 基本改善（即実施推奨）

#### 1. エントロピー重みの削減（最優先）

```json
"rl_entropy_weight": 0.05  // 0.5 → 0.05（90%削減）
```

現在の値0.5は非常に高すぎるため、探索過多で収束が遅れている。
期待効果: Complete Rate 81% → 88%+、報酬分散の減少

#### 2. 温度スケジュールの加速

```json
"rl_sampling_temperature": 1.5,    // 2.0 → 1.5
"rl_min_temperature": 0.8,         // 1.0 → 0.8
"rl_temperature_decay": 0.035      // 0.02 → 0.035
```

初期は広く探索（温度1.5）→ 中期で収束（温度1.0、epoch 10-15）→ 後期はほぼ決定的（温度0.8、epoch 20+）

#### 3. ベースラインの安定化

```json
"rl_baseline_momentum": 0.95  // 0.85 → 0.95
```

ベースラインの急激な変動を防止し、Advantageの計算を安定化。

#### 4. エポック数の削減

```json
"max_epochs": 25,      // 50 → 25
"val_every": 5,        // 3 → 5
"test_every": 5        // 3 → 5
```

Load Factorはepoch 10で安定済み。後半40エポックの改善はわずか5%。訓練時間を50%削減。

### Phase 2: ペナルティ調整（Phase 1の結果次第）

#### 5. 不完全パスへのペナルティ強化

```python
# reinforcement_strategy.py:310-315
if num_incomplete <= 1:
    reward_i = -5.0   # -2.0 → -5.0
elif num_incomplete <= 2:
    reward_i = -10.0  # -5.0 → -10.0
else:
    reward_i = -15.0  # -10.0 → -15.0
```

期待効果: Complete Rate 81% → 90%+、平均報酬 1.4 → 4.0+

#### 6. 報酬設計の微調整

```python
# 実行可能解（LF ≤ 1.0）
reward = 15.0 - LF * 5.0  # 範囲: 10.0~15.0（現在: 7.0~10.0）

# 容量超過（LF > 1.0）
reward = 8.0 - LF * 3.0   # （現在: 5.0 - LF * 2.0）
```

### Phase 3: 高度な最適化（オプション）

#### 7. Early Stopping の実装

```python
early_stopping_patience = 10  # 10エポック改善なしで停止
```

#### 8. Curriculum Learning

小規模問題（commodities=3）→ 標準問題（commodities=5）→ 大規模問題（commodities=7）と段階的に難易度を上げる。

#### 9. 動的Reachability Mask（残容量考慮）

静的Maskでは効果が限定的だったため、サンプリング中の残容量を追跡して容量不足の経路をマスクする。期待効果: Complete Rate 83% → 90-95%。実装工数は中。

---

## エントロピー計算の改善手法

### 手法A: 行動可能ノードのみでエントロピー平均

選択肢が1つしかないノード（自明な行動）をエントロピー平均から除外。

```python
multi_choice = (probs > 1e-8).sum(dim=2) >= 2  # 2つ以上の選択肢がある箇所
entropy = row_entropy[multi_choice].mean() if multi_choice.any() else row_entropy.mean()
```

### 手法B: サンプリング軌跡上でエントロピー算出

全ノードの分布ではなく、実際に行動したノード列（軌跡）に基づいてエントロピーを計算。学習に効く行動にだけボーナスが入る。

### 手法C: epsilon-greedy的な有効近傍の一様混合

```python
eps = 0.05
probs = (1 - eps) * probs + eps * uni  # 少量の一様分布を混合
```

「候補ゼロで詰む」や「ほぼ1点集中」を回避。

---

## 推奨設定ファイル

```json
{
    "expt_name": "gcn_rl_unconstrained_improved",
    "max_epochs": 25,
    "val_every": 5,
    "test_every": 5,
    "batch_size": 20,
    "learning_rate": 0.001,
    "training_strategy": "reinforcement",
    "rl_reward_type": "load_factor",
    "rl_use_baseline": true,
    "rl_baseline_momentum": 0.95,
    "rl_entropy_weight": 0.05,
    "rl_use_trajectory_entropy": true,
    "rl_entropy_epsilon": 0.05,
    "rl_use_sampling": true,
    "rl_sampling_temperature": 1.5,
    "rl_min_temperature": 0.8,
    "rl_temperature_decay": 0.035,
    "rl_sampling_top_p": 0.95,
    "rl_normalize_advantages": true,
    "rl_mask_invalid_edges": true,
    "rl_beam_search_type": "unconstrained"
}
```

---

## 期待される改善効果

| 指標 | 現在 | 目標 | 改善幅 |
|------|------|------|--------|
| 平均報酬 | 1.41 | 5.0~7.0 | +250%~400% |
| Complete Rate | 81.6% | 90%+ | +8%+ |
| Load Factor | 0.344 | 0.30~0.35 | -10%~維持 |
| 報酬の標準偏差 | 大 | < 1.5 | 安定化 |
| 訓練時間 | 331秒 | 166秒 | -50% |

---

## チェックリスト

- [ ] Phase 1設定ファイル作成・実行（25 epochs）
- [ ] 結果の分析（報酬、Complete Rate、時間）
- [ ] Phase 2の必要性判断
- [ ] （必要なら）ペナルティ調整の実装
- [ ] 最終的な設定の決定

---

## 学んだ教訓

1. **エントロピーボーナスは諸刃の剣** — 探索を促進するが収束を遅らせる。初期は高め、後期は低めが理想
2. **温度スケジューリングが重要** — 適切な減衰率が学習効率を左右
3. **報酬設計のバランス** — ペナルティが弱すぎると悪い行動を学習、強すぎると探索が停止。現在は「弱すぎる」側
4. **エポック数の最適化** — 多ければ良いわけではない。Epoch 10-15で主要な学習は完了
5. **Reachability Maskの限界** — 静的な構造的マスクでは動的な容量制約の問題は解決できない
