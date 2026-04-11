# 報酬設計の比較分析

**作成日**: 2025-10-19

---

## 結論

**ユーザー提案の改善案が最適**

```python
if not np.isinf(load_factor):
    if load_factor <= 1.0:
        reward = 10 - load_factor * 3  # 7~10
    else:
        reward = 5 - load_factor * 2   # 軽い罰
else:
    reward = -10  # 無効パス
```

---

## 数値比較

### 報酬スパン

| 設計 | スパン | 評価 |
|-----|--------|------|
| 旧実装 (`-load_factor`, 無効=-100) | 99.80 | 大きすぎる |
| レポート提案 (100-lf*10) | 198.00 | 非常に大きい |
| **採用案** (10-lf*3) | **19.40** | **適度** |

### 報酬マッピング

| Load Factor | 状態 | 報酬 |
|------------|------|------|
| 0.2 | 優秀 | 9.4 |
| 0.4 | 良好 | 8.8 |
| 0.6 | 普通 | 8.2 |
| 0.8 | やや悪い | 7.6 |
| 1.0 | 限界 | 7.0 |
| 1.2 | 軽度超過 | 2.6 |
| 2.0 | 重度超過 | 1.0 |
| 5.0 | 極度超過 | -5.0 |
| inf | 無効 | -10.0 |

---

## なぜこの設計が優れているか

### 1. 勾配爆発の防止

REINFORCEの勾配: `∇J = E[∇log π(a|s) × (R - baseline)]`

- 報酬範囲20 → 安定した勾配（レポート案の範囲200では勾配が10倍大きい）

### 2. ベースラインとのバランス

- 報酬7~10、ベースラインとの差が適度 → バランス良好

### 3. 探索と活用のバランス

- 無効解ペナルティ-10 → 探索を促進（-100では探索を抑制しすぎ）

---

## 不完全パスの段階的ペナルティ

```python
if not all_paths_complete:
    num_incomplete = sum(1 for path in sample_paths if len(path) == 0 or path[-1] != dst)
    if num_incomplete <= 1:
        reward_i = -2.0   # 現在: -5.0
    elif num_incomplete <= 2:
        reward_i = -5.0
    else:
        reward_i = -10.0
elif np.isinf(load_factor):
    reward_i = -5.0  # パスは完成したが無効なエッジ使用
elif load_factor <= 1.0:
    reward_i = 10.0 - load_factor * 3.0  # 7~10
else:
    reward_i = 5.0 - load_factor * 2.0   # 容量超過
```
