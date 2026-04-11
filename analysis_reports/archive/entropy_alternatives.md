# エントロピー計算の代替手法

**作成日**: 2025-10-20

---

## 背景

エントロピーが0.5787で固定されていた問題に対する3つの対処法。

---

## 手法1: 行動可能ノードのみでエントロピー平均（推奨）

選択肢が1つしかないノード（自明な行動）をエントロピー平均から除外。

```python
probs = F.softmax(y_preds / current_temp, dim=2)
log_probs = F.log_softmax(y_preds / current_temp, dim=2)

multi_choice = (probs > 1e-8).sum(dim=2) >= 2  # [B, N_from, C]
row_entropy = -(probs * log_probs).sum(dim=2)   # [B, N_from, C]
entropy = row_entropy[multi_choice].mean() if multi_choice.any() else row_entropy.mean()
```

効果: 固定値（0.0005など）から脱却し、0.05~0.3程度で安定。

---

## 手法2: サンプリング軌跡上でエントロピー算出

全ノードの分布ではなく、実際に行動したノード列（軌跡）に基づいてエントロピーを計算。

```python
# PathSampler.sample() の戻り値に stepwise_entropies を追加
traj_entropy = torch.tensor([e for per_batch in stepwise_entropies for e in per_batch],
                            device=y_preds.device, dtype=torch.float32)
entropy = traj_entropy.mean()
```

効果: 学習に効く行動（実際に選んだ状態）にだけボーナスが入る。

---

## 手法3: epsilon-greedy的な有効近傍の一様混合

top-p が効きすぎると分布が尖る。小さな epsilon を足して常に少しだけ揺らす。

```python
if self.training and self.use_sampling:
    with torch.no_grad():
        valid = (~invalid_mask).unsqueeze(-1).float()
        uni = valid / (valid.sum(dim=2, keepdim=True) + 1e-8)
    eps = 0.05
    probs = (1 - eps) * probs + eps * uni
    probs = probs / (probs.sum(dim=2, keepdim=True) + 1e-8)
```

効果: 「候補ゼロで詰む」や「ほぼ1点集中」を回避。PathSampler側のtop-pと相性が良い。
