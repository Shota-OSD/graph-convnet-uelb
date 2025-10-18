# Unconstrained Beam Search

## 概要

`UnconstrainedBeamSearch`は、エッジ容量制約を無視してパス探索を行うビームサーチの変種です。

## StandardBeamSearchとの違い

### StandardBeamSearch（容量制約あり）

```python
# 155行目: 容量チェック
if demand <= remaining_edges_capacity[current_node, next_node]:
    updated_capacity = remaining_edges_capacity.clone()
    updated_capacity[current_node, next_node] -= demand
    new_path = path + [next_node]
    next_beam_queue.append((next_node, new_path, new_score, updated_capacity))
```

- **容量チェック**: `demand <= remaining_edges_capacity`
- **容量更新**: エッジを使うたびに容量を減らす
- **結果**: 容量制約を満たすパスのみ生成

### UnconstrainedBeamSearch（容量制約なし）

```python
# 437-438行目: 容量チェックなし
new_path = path + [next_node]
next_beam_queue.append((next_node, new_path, new_score))
```

- **容量チェック**: なし（エッジの存在のみ確認）
- **容量更新**: なし
- **結果**: 容量を超えるパスも生成可能

## 実装の詳細

### ファイル

[src/gcn/algorithms/beamsearch_uelb.py:378-454](src/gcn/algorithms/beamsearch_uelb.py#L378)

### 主な特徴

1. **容量制約を無視**
   - エッジが存在する（capacity > 0）ことのみ確認
   - 需要量による容量チェックなし
   - 容量の更新なし

2. **より多くのパスを探索可能**
   - 容量制約により除外されていたパスも考慮
   - より多様な経路の発見が可能

3. **実行可能性の評価は後で実施**
   - ビームサーチ後に最大負荷率を計算
   - 負荷率 > 1 の場合は実行不可能として扱う

## 使用方法

### 1. BeamSearchFactoryを使用

```python
from src.gcn.algorithms.beamsearch_uelb import BeamSearchFactory

beam_search = BeamSearchFactory.create_algorithm(
    'unconstrained',  # アルゴリズム名
    y_pred_edges=y_preds,
    beam_size=1280,
    batch_size=1,
    edges_capacity=edges_capacity,
    commodities=commodities,
    dtypeFloat=torch.float,
    dtypeLong=torch.long,
    mode_strict=True
)
paths, is_feasible = beam_search.search()
```

### 2. 直接インスタンス化

```python
from src.gcn.algorithms.beamsearch_uelb import UnconstrainedBeamSearch

beam_search = UnconstrainedBeamSearch(
    y_pred_edges=y_preds,
    beam_size=1280,
    batch_size=1,
    edges_capacity=edges_capacity,
    commodities=commodities,
    dtypeFloat=torch.float,
    dtypeLong=torch.long,
    mode_strict=True
)
paths, is_feasible = beam_search.search()
```

### 3. RL戦略での使用

設定ファイル: [configs/gcn/rl_unconstrained.json](configs/gcn/rl_unconstrained.json)

```json
{
  "training_strategy": "reinforcement",
  "rl_beam_search_type": "unconstrained"
}
```

実行:
```bash
python3 scripts/gcn/train_gcn.py --config configs/gcn/rl_unconstrained.json
```

## 利用可能なビームサーチアルゴリズム

| アルゴリズム | 説明 | 容量制約 | シャッフル |
|------------|------|---------|----------|
| `standard` | 標準ビームサーチ | あり | あり |
| `unconstrained` | 容量制約なし | **なし** | なし |
| `deterministic` | 決定論的 | あり | なし |
| `greedy` | 貪欲法（beam_size=1） | あり | あり |

## ユースケース

### 1. 強化学習での利用

**メリット:**
- 容量制約に縛られず、多様な経路を探索
- モデルが自由に経路を学習できる
- 探索空間が広がる

**デメリット:**
- 実行不可能な解が多く生成される可能性
- 報酬が不安定になる可能性

**推奨設定:**
```json
{
  "training_strategy": "reinforcement",
  "rl_beam_search_type": "unconstrained",
  "rl_reward_type": "load_factor",
  "rl_use_baseline": true,
  "rl_baseline_momentum": 0.95
}
```

### 2. 探索的な分析

容量制約を考慮しない「理想的な」経路パターンを発見したい場合。

### 3. 前処理段階

容量を考慮せずに有望な経路を見つけ、後で容量割り当てを調整。

## アルゴリズムの選択基準

### Standard (推奨: 実運用)
- 実行可能な解が必要
- 容量制約が重要
- 現実的な問題設定

### Unconstrained (推奨: 実験・探索)
- 強化学習での探索
- 容量制約を気にせず最適経路を見つけたい
- 理論的な上限を調査

### Deterministic (推奨: 再現性重視)
- デバッグ
- 再現性が必要な実験
- ランダム性を排除したい

### Greedy (推奨: 高速化)
- 計算時間を最小化
- 簡易的な解で十分
- ベースライン手法として

## 実装の詳細

### _unconstrained_beam_search_for_commodity

[src/gcn/algorithms/beamsearch_uelb.py:410-450](src/gcn/algorithms/beamsearch_uelb.py#L410)

```python
def _unconstrained_beam_search_for_commodity(self, edges_capacity, y_commodities, source, target):
    beam_queue = [(source, [source], 0)]  # 容量情報なし
    best_paths = []

    while beam_queue:
        beam_queue = sorted(beam_queue, key=lambda x: x[2], reverse=True)[:self.beam_size]

        next_beam_queue = []
        for current_node, path, current_score in beam_queue:
            if current_node == target:
                best_paths.append((path, current_score))
                continue

            for next_node in range(edges_capacity.shape[0]):
                if next_node in path:  # ループ検出のみ
                    continue
                if edges_capacity[current_node, next_node].item() == 0:  # エッジ存在確認
                    continue

                flow_probability = y_commodities[current_node, next_node]
                new_score = current_score + flow_probability
                new_path = path + [next_node]

                # 容量チェックなしで追加
                next_beam_queue.append((next_node, new_path, new_score))

        beam_queue = next_beam_queue

    # 最良パスを返す
    if not best_paths:
        return [0] * edges_capacity.shape[0], []

    best_paths = sorted(best_paths, key=lambda x: x[1], reverse=True)
    node_order = [0] * edges_capacity.shape[0]
    for idx, node in enumerate(best_paths[0][0]):
        node_order[node] = idx + 1
    return node_order, best_paths[0][0]
```

### 主な変更点

1. **beam_queueの構造**
   - Standard: `(node, path, score, remaining_capacity)`
   - Unconstrained: `(node, path, score)` ← 容量情報なし

2. **エッジ追加条件**
   - Standard: `if demand <= remaining_capacity[edge]`
   - Unconstrained: 条件なし（エッジ存在のみ）

3. **容量更新**
   - Standard: `remaining_capacity -= demand`
   - Unconstrained: 更新なし

## 注意事項

1. **実行可能性の保証なし**
   - 生成されたパスが容量制約を満たす保証はない
   - 必ず後で負荷率チェックが必要

2. **メモリ使用量**
   - 容量制約がないため、より多くのパスが候補になる
   - beam_sizeを大きくしすぎると メモリ不足の可能性

3. **報酬設計**
   - 実行不可能な解に対する適切なペナルティが必要
   - 推奨: `reward = -10.0` for infeasible solutions

## パフォーマンス比較

|  | Standard | Unconstrained |
|--|----------|---------------|
| **探索範囲** | 狭い（容量制約内） | 広い（全経路） |
| **実行可能解** | 常に保証 | 保証なし |
| **計算時間** | やや遅い | やや速い |
| **メモリ使用** | 少ない | 多い（beam拡大） |
| **RL学習** | 安定 | 不安定（要調整） |

## まとめ

`UnconstrainedBeamSearch`は、容量制約を無視して経路探索を行うことで、より自由な探索を可能にします。強化学習での探索や理論的分析に有用ですが、実行可能性は保証されないため、適切な報酬設計とペナルティが必要です。

**推奨用途:**
- 強化学習の探索段階
- 理論的な最適経路の発見
- 容量制約緩和後の問題分析

**非推奨用途:**
- 本番環境での実行可能解の生成
- 厳密な容量制約が必要な問題
