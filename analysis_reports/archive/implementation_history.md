# 実装履歴

**統合元**: implementation_summary_20251019.md, training_analysis_20251019_fixed.md, learning_analysis_20251019.txt

---

## 2025-10-19: 致命的バグ修正 — Load Factor計算

### 問題
`src/common/graph/graph_utils.py:166-167` で `torch.where()` にスカラーテンソルを使用し、ブロードキャストエラーが発生。全てのLoad Factorが `inf` になり、全ての報酬が-5.0（無効エッジペナルティ）になっていた。

### 修正
```python
# 修正前: スカラーテンソル
torch.tensor(0.0, device=device)
torch.tensor(float('inf'), device=device)

# 修正後: 正しい形状のテンソル
torch.zeros_like(bs_edges_summed, dtype=torch.float)
torch.full_like(bs_edges_summed, float('inf'), dtype=torch.float)
```

### 効果
| 指標 | 修正前 | 修正後 |
|------|--------|--------|
| Load Factor | inf (100%) | 0.33-0.37 |
| Finite Solution Rate | 59.2% | 100.0% |
| 報酬範囲 | -70 ~ -81 | -1.5 ~ +0.81 |

---

## 2025-10-19: 報酬設計の改善

### 変更（reinforcement_strategy.py:272-304）

```python
# 旧: reward = -load_factor (無効なら-100)
# 新:
# 不完全パス: -5（1つ失敗）or -10（複数失敗）
# 無効エッジ: -5（パスは完成）
# 有効解（LF ≤ 1.0）: 10.0 - LF * 3.0 → 7~10
# 容量超過（LF > 1.0）: 5.0 - LF * 2.0
```

報酬スパン: 200 → 20（1/10に削減）

---

## 2025-10-19: 温度スケジューリングの導入

reinforcement_strategy.py に温度スケジューリングを追加:
- `T(epoch) = max(0.5, 1.0 - 0.05 * epoch)`
- Epoch 0: 温度1.0（探索重視）→ Epoch 10: 温度0.5（活用重視）

trainer.py にエポック情報設定と温度表示を追加。

---

## 2025-10-19: ベースラインモメンタム調整

`configs/gcn/rl_min_unconstrained.json`:
- `rl_baseline_momentum`: 0.9 → 0.95

---

## 2025-10-19: 初期学習分析（修正前）

修正前の状態:
- 報酬が常に負（-70 ~ -81）で悪化傾向
- アドバンテージがほぼ常に負（10エポック中9エポック）
- エントロピーが0.5787で完全固定
- 約40%のサンプルが無効パス（inf）

根本原因:
1. Load Factor計算のバグ（上記で修正）
2. 報酬設計の問題（常に負、スパンが大きすぎる）
3. 温度パラメータが固定

---

## 2025-10-19: 修正後の学習結果（10エポック）

```
Epoch  Reward   Complete%  Load Factor
1      -1.13    79.6%      0.3524
4      +0.81    85.0%      0.3684  ← 最良
10     -0.47    81.2%      0.3550
```

改善点:
- Load Factor有限化、正の報酬出現
- Finite Solution Rate 100%

残課題:
- 報酬の振動（+0.81 → -0.79）
- 不完全パス15-21%
- エントロピー0.5787で固定（モデル出力のEntropyであり、PathSampler分布ではないため）
