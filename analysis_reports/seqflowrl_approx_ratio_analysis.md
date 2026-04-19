# SeqFlowRL: Approximation Ratio が 100% を超える原因分析

**作成日**: 2026-04-17
**対象コード**: `src/seq_flow_rl/training/a2c_strategy.py`, `src/seq_flow_rl/algorithms/sequential_rollout.py`
**観測例**: `Epoch 2/30 | Approx Ratio: 100.57%`

---

## 背景

Approximation Ratio は「モデル解が最適解にどれだけ近いか」を表す指標で、**負荷率は小さいほど良い**ため、式は以下のように定義されている。

```python
# a2c_strategy.py:314
approximation_ratio = (mean_gt_lf / mean_model_lf) * 100
```

コメント上の定義:
- `100%` … モデル解が最適と一致
- `< 100%` … 最適より悪い
- `> 100%` … **理論上あり得ない**（最適解より良い解を見つけた）

しかし実際には `100.57%` のようにわずかに 100% を超えるログが観測される。本ドキュメントではその原因を分析する。

---

## 結論（先に）

主因は **モデル側の `mean_model_lf` が "真の負荷率" よりも過小に計算されていること** である。
特に次の2点が効いている:

1. **バグ**: `_update_edge_usage` が **src → 最初の中間ノードのエッジを edge_usage に積んでいない**（最も有力）
2. **設計上の問題**: `mean_gt_lf / mean_model_lf` が「バッチ平均の比」であり、「サンプルごとの比の平均」ではないため、バッチ内分散が大きい場合に 100% をまたぐ偏りが出る

---

## 原因1: `_update_edge_usage` が先頭エッジを積んでいない（バグ）

### 該当箇所

**パス生成** (`sequential_rollout.py:208-269`):

```python
# L208-209: パス配列は空で初期化（src は入らない）
paths = [[] for _ in range(batch_size)]

# L215: current_nodes に src をセット
current_nodes = src_nodes.clone()  # [B]

# L268: サンプリング結果のみを path に追加
for b in range(batch_size):
    if not reached_dst[b]:
        next_node = next_nodes[b].item()
        paths[b].append(next_node)   # ← src は append されない
```

**需要量の積算** (`sequential_rollout.py:355-363`):

```python
for i in range(len(path) - 1):
    u = path[i]
    v = path[i + 1]
    state['x_edges_usage'][b, u, v] += demand
```

### 何が起きているか

コモディティ `c` の src = `s`、到達先 = `d`、サンプリングで `s → v1 → v2 → d` という経路が生成された場合:

| 実際に使われるエッジ | `_update_edge_usage` で積まれる？ |
|---|---|
| `s → v1` | **× 抜け落ちる**（path に `s` が入っていない） |
| `v1 → v2` | ○ |
| `v2 → d` | ○ |

→ **各コモディティにつき 1 エッジ分、需要が `edge_usage` に反映されない**。

### 影響

- `edge_usage` が過小評価される → `load_factor = max(edge_usage / capacity)` も過小評価
- モデル側の `mean_model_lf` が下がり、`gt / model` の比が **100% を超えうる**
- `max_edge_load` を更新していたエッジがたまたま先頭エッジだった場合、誤差がさらに拡大する

### 修正案

パスの起点を明示的に記録する:

```python
# sequential_rollout.py:208-209 近辺
# src をパスの先頭に入れる
paths = [[src_nodes[b].item()] for b in range(batch_size)]
```

または `_update_edge_usage` 側で `src` を別途考慮する。前者の方が既存コードへの影響が少ない。

ただしこの修正は **train_step の報酬計算**・**approximation_ratio 計算** 両方に影響するため、挙動が変わる範囲を確認したうえで適用する必要がある。

---

## 原因2: 平均の取り方の偏り（設計上の問題）

### 該当箇所 (`a2c_strategy.py:295-316`)

```python
if isinstance(gt_load_factors, torch.Tensor):
    mean_gt_lf = gt_load_factors.mean().item()
else:
    mean_gt_lf = float(gt_load_factors.mean())

mean_model_lf = load_factors.mean().item()

if mean_model_lf > 0:
    approximation_ratio = (mean_gt_lf / mean_model_lf) * 100
```

### 問題点

これは **「バッチ平均の比」** である:

```
Approx = mean(gt) / mean(model)
```

本来必要なのは **「サンプルごとの比の平均」**:

```
Approx = mean(gt_i / model_i)
```

両者は一般に一致しない（Jensen の不等式、調和平均 vs 算術平均の関係）。

### 具体例

バッチに2サンプル、GT=[0.2, 0.6]、Model=[0.25, 0.4] の場合:

| 指標 | 計算 | 値 |
|---|---|---|
| バッチ平均の比（現行） | 0.4 / 0.325 | **123%** |
| サンプル毎の比の平均（正しい） | (0.2/0.25 + 0.6/0.4) / 2 | **115%** |

特に **GT が低いサンプル（モデルが真似しにくい問題）が混ざる** と `mean_gt_lf` が GT の平均を超過的に引き上げ、100% を越えた値になりやすい。

### 修正案

```python
# サンプルごとの比を計算してから平均
valid = (load_factors > 0) & (gt_load_factors > 0)
per_sample_ratio = (gt_load_factors[valid] / load_factors[valid]) * 100
approximation_ratio = per_sample_ratio.mean().item()
```

---

## 原因3: 不完全パス（未到達）の扱い

`sequential_rollout.py:227-233` では `max_path_length` （seqflowrl_base.json: 20）到達で打ち切られるが、**未到達パスに対するペナルティは報酬側でのみ適用され**、`edge_usage` には「パスの到達したところまでの需要しか積まれない」。

- 到達できなかったコモディティの需要 = 本来は全エッジに負荷がかかる可能性があった
- しかし実装上は「途中まで歩いた分のエッジ」だけが負荷に加算される
- → モデルの負荷率はさらに過小評価される

学習序盤（Epoch 2時点）は到達率が低いため、この効果が顕著に出る。

---

## 原因4（補足）: 報酬側との二重カウント不在

`_compute_rewards` では `load_factor` と `penalty` の両方を報酬に使うが、`_collect_metrics` の `load_factors` は **ペナルティ前の生の負荷率**。よって報酬の妥当性とは別に、approx ratio 計算は純粋な "model load factor vs gt load factor" の比較になっている点は正しい。

一方で、GT はソルバーが **制約をすべて守って出した最適解** なので「到達失敗」や「容量超過」のような状態は含まない。モデル側が不完全解を出していても、`approximation_ratio` はそれを罰しない（load_factor にしか目を向けない）。→ **approx ratio は "パスの到達性" を反映しない指標** であり、これを単独で見ると誤った楽観的評価になる。

---

## 推奨対応

### 短期（バグ修正）

1. **原因1 の修正**: `paths` の初期化を `[[src_nodes[b].item()]]` に変更し、`_update_edge_usage` が先頭エッジも積むようにする
2. 単体テストで「src → v1 → v2」のパスが `edge_usage[src, v1] += demand` を正しく行うか確認

### 中期（指標の正確化）

3. **原因2 の修正**: approximation_ratio をサンプル単位比の平均に変更
4. **原因3 の対処**: `approximation_ratio` を計算する際、**dst に到達しなかったサンプルを除外**（または不完全パス比率を併記）
5. ログに「Complete Rate」や「Invalid Solution Rate」を追加し、approx ratio 単独で評価しない

### 長期（設計）

6. approximation ratio の定義を「同一問題インスタンスでの比」に正規化
7. バリデーション・テストでは、model が全制約を満たした場合のみ approx を計算するゲートを設ける

---

## 参考: 関連コード

| 箇所 | 役割 |
|---|---|
| `src/seq_flow_rl/training/a2c_strategy.py:295-316` | approximation_ratio 計算（train） |
| `src/seq_flow_rl/training/a2c_strategy.py:394-415` | approximation_ratio 計算（eval） |
| `src/seq_flow_rl/algorithms/sequential_rollout.py:208-269` | パス生成（src を path に入れていない） |
| `src/seq_flow_rl/algorithms/sequential_rollout.py:341-363` | edge_usage 更新（先頭エッジ抜け） |
| `src/seq_flow_rl/training/a2c_strategy.py:176-200` | load_factor 計算 |
