# 強化学習改善プラン - 2025年10月21日

## 📊 現状の診断

### 訓練結果サマリー（50エポック）
- **平均報酬**: 1.41（理想: 7~9）⚠️
- **Complete Paths Rate**: 81.6%（理想: 95%+）⚠️
- **Load Factor**: 0.34（良好）✅
- **Finite Solution Rate**: 100%（完璧）✅
- **Capacity Violation**: 0.4%（ほぼゼロ）✅

### 問題点の特定

#### 1. **報酬が低い（平均1.41）**
理由：
- 約20%のコモディティが目的地に到達できず
- 1パス失敗で-2.0、2パス失敗で-5.0のペナルティ
- 完全な解でもLF最適化が不十分（報酬7~9に到達せず）

#### 2. **報酬の変動が大きい（0.29~2.85）**
理由：
- 探索（高温度、高エントロピー重み）と活用のバランスが悪い
- ベースラインが不安定（0.11~3.16）
- 温度低下が遅すぎる

#### 3. **Complete Rateが停滞（79-86%）**
理由：
- 不完全パスへのペナルティが弱い（-2.0は軽すぎる）
- エントロピーボーナスが探索を促しすぎる
- モデルが「リスクを取る」戦略を学習

#### 4. **エントロピーの改善は成功したが活用されていない**
- エントロピー: 0.0005 → 0.05~0.15（100倍改善）✅
- しかし探索過多で収束が遅れている

---

## 🎯 改善策

### 【優先度: 高】即効性のある改善

#### 1. エントロピー重みの削減（最優先）
**目的**: 探索を減らし、より決定的な方策を学習

```json
"rl_entropy_weight": 0.05,  // 0.5 → 0.05（90%削減）
```

**重要**: 現在の値0.5は**非常に高すぎる**ため、探索過多で収束が遅れています。

**効果**:
- より一貫した行動選択
- 報酬の分散が減少
- Complete Rateの向上（81% → 88%+）
- 学習の収束速度が大幅に改善

---

#### 2. 温度スケジュールの加速
**目的**: より早く収束モードに移行

```json
"rl_sampling_temperature": 1.5,    // 2.0 → 1.5
"rl_min_temperature": 0.8,         // 1.0 → 0.8
"rl_temperature_decay": 0.035,     // 0.02 → 0.035
```

**効果**:
- 初期: 広く探索（温度1.5）
- 中期: 徐々に収束（温度1.0前後、epoch 10-15）
- 後期: ほぼ決定的（温度0.8、epoch 20+）

---

#### 3. ベースラインの安定化
**目的**: 報酬の変動を抑え、学習を安定化

```json
"rl_baseline_momentum": 0.95,  // 0.85 → 0.95
```

**効果**:
- ベースラインの急激な変動を防止
- Advantageの計算が安定
- 勾配の分散が減少

---

#### 4. エポック数の削減
**目的**: 実行時間の削減（品質はほぼ維持）

```json
"max_epochs": 25,      // 50 → 25（50%削減）
"val_every": 5,        // 3 → 5
"test_every": 5        // 3 → 5
```

**効果**:
- 訓練時間: 331秒 → 166秒（約2.8分）
- Load Factorはepoch 10で安定済み
- 後半40エポックの改善はわずか5%

---

### 【優先度: 中】追加の最適化

#### 5. 不完全パスへのペナルティ強化
**目的**: Complete Rateを95%以上に引き上げ

現在の実装（`reinforcement_strategy.py:310-315`）を修正:

```python
# 現在
if num_incomplete <= 1:
    reward_i = -2.0  # 軽すぎる
elif num_incomplete <= 2:
    reward_i = -5.0
else:
    reward_i = -10.0

# 推奨
if num_incomplete <= 1:
    reward_i = -5.0   # -2.0 → -5.0（2.5倍）
elif num_incomplete <= 2:
    reward_i = -10.0  # -5.0 → -10.0（2倍）
else:
    reward_i = -15.0  # -10.0 → -15.0（1.5倍）
```

**効果**:
- 不完全パスへの強いインセンティブ
- Complete Rate: 81% → 90%+（予想）
- 平均報酬: 1.4 → 4.0+（予想）

---

#### 6. 報酬設計の微調整
**目的**: より大きな報酬スパンで学習シグナルを強化

```python
# 実行可能解（LF ≤ 1.0）
# 現在: reward = 10.0 - LF × 3.0  （範囲: 7.0~10.0）
# 推奨: reward = 15.0 - LF × 5.0  （範囲: 10.0~15.0）

# 容量超過（LF > 1.0）
# 現在: reward = 5.0 - LF × 2.0
# 推奨: reward = 8.0 - LF × 3.0
```

**効果**:
- より明確な報酬シグナル
- 実行可能解と不実行可能解の差が拡大
- LF最適化のインセンティブが増加

---

### 【優先度: 低】長期的な改善

#### 7. Early Stopping の実装
**目的**: 収束後の無駄な訓練を防止

```python
# trainer.py に追加
early_stopping_patience = 10  # 10エポック改善なしで停止
best_val_metric = -inf
no_improvement_count = 0

if val_metric <= best_val_metric:
    no_improvement_count += 1
    if no_improvement_count >= early_stopping_patience:
        print("Early stopping triggered")
        break
```

---

#### 8. Curriculum Learning
**目的**: 段階的に難易度を上げる

```python
# 初期: 小規模問題（commodities=3）
# 中期: 標準問題（commodities=5）
# 後期: 大規模問題（commodities=7）
```

---

## 📋 推奨設定ファイル

### `configs/gcn/rl_min_unconstrained_improved.json`

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

## 🎯 期待される改善効果

### 訓練時間
- **現在**: 331秒（5.5分）
- **改善後**: 166秒（2.8分）
- **削減**: 50%

### 性能指標

| 指標 | 現在 | 目標 | 改善幅 |
|------|------|------|--------|
| 平均報酬 | 1.41 | 5.0~7.0 | +250%~400% |
| Complete Rate | 81.6% | 90%+ | +8%+ |
| Load Factor | 0.344 | 0.30~0.35 | -10%~維持 |
| 報酬の標準偏差 | 大 | < 1.5 | 安定化 |
| Val Approx Rate | 74.04% | 80%+ | +6%+ |
| Test Approx Rate | 67.06% | 75%+ | +8%+ |

---

## 🔬 実験プラン

### Phase 1: 基本改善（即実施推奨）
1. エントロピー重み: 0.1 → 0.05
2. 温度設定: initial 2.0 → 1.5, decay 0.02 → 0.035
3. ベースライン: momentum 0.85 → 0.95
4. エポック数: 50 → 25

**期待結果**: 訓練時間50%削減、報酬+100%改善

### Phase 2: ペナルティ調整（Phase 1の結果次第）
1. 不完全パスペナルティ強化
2. 報酬スパン拡大

**期待結果**: Complete Rate 90%+、報酬+200%改善

### Phase 3: 高度な最適化（オプション）
1. Early Stopping実装
2. Curriculum Learning

**期待結果**: さらなる効率化

---

## 📝 実装手順

### ステップ1: 設定ファイルのコピーと編集
```bash
cp configs/gcn/rl_min_unconstrained.json configs/gcn/rl_min_unconstrained_improved.json
# 上記の推奨設定を適用
```

### ステップ2: Phase 1の実行
```bash
python scripts/gcn/train_gcn.py --config configs/gcn/rl_min_unconstrained_improved.json
```

### ステップ3: 結果の比較
```bash
# ログを比較して改善を確認
# 平均報酬、Complete Rate、訓練時間をチェック
```

### ステップ4: Phase 2の実行（必要に応じて）
```python
# reinforcement_strategy.py のペナルティを修正
# 再実行して効果を検証
```

---

## 🎓 学んだ教訓

1. **エントロピーボーナスは諸刃の剣**
   - 探索を促進するが、収束を遅らせる
   - 初期は高め、後期は低めが理想

2. **温度スケジューリングが重要**
   - 初期: 高温で広く探索
   - 後期: 低温で収束
   - 適切な減衰率が学習効率を左右

3. **報酬設計のバランス**
   - ペナルティが弱すぎると悪い行動を学習
   - ペナルティが強すぎると探索が停止
   - 現在は「弱すぎる」側

4. **エポック数の最適化**
   - 多ければ良いわけではない
   - Epoch 10-15で主要な学習は完了
   - 後半は微調整のみ

---

## ✅ チェックリスト

- [ ] Phase 1設定ファイル作成
- [ ] Phase 1実行（25 epochs, 改善設定）
- [ ] 結果の分析（報酬、Complete Rate、時間）
- [ ] Phase 2の必要性判断
- [ ] （必要なら）ペナルティ調整の実装
- [ ] 最終的な設定の決定
- [ ] ドキュメント更新

---

**作成日**: 2025年10月21日
**ベース訓練ログ**: `logs/training_results_20251021_231845.txt`
**関連レポート**: `solution_20251020.md`
