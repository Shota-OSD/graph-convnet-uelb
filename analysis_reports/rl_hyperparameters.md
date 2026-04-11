# RL ハイパーパラメータ早見表

`src/gcn/training/reinforcement_strategy.py` の探索・安定化に関わる主要パラメータ。

---

## パラメータ一覧

### rl_baseline_momentum
- **役割**: REINFORCE の移動平均ベースラインの慣性
- **効果**: 高いほど報酬変動への追従は遅いが、勾配分散が小さく学習が安定
- **デフォルト**: 0.9
- **調整目安**: 0.85~0.95。不安定→上げる、環境変化に鈍い→下げる

### rl_entropy_weight
- **役割**: エントロピー正則化の係数（探索促進）
- **効果**: 高いほど行動分布が広がり探索が増えるが、収束が遅く/不安定に
- **デフォルト**: 0.01
- **調整目安**: 0.01~0.1。探索不足→上げる、収束が遅い→下げる

### rl_use_trajectory_entropy
- **役割**: 軌跡（経路）ベースのエントロピーを使用するか
- **効果**: 経路全体の多様性を促進。系列依存の探索を強めたい場合に有効
- **デフォルト**: True

### rl_entropy_epsilon
- **役割**: 有効手への一様混合（epsilon-greedy風）の混合率
- **効果**: logitsが鋭いときでも一定の探索を担保
- **デフォルト**: 0.05
- **調整目安**: 0.01~0.1。決め打ちが強い→少し上げる

### rl_sampling_temperature
- **役割**: サンプリング温度（初期温度）。低いほど決定的、高いほど多様
- **デフォルト**: 1.0
- **調整目安**: 0.8~2.0。探索初期は高め→学習とともに下げる

### rl_min_temperature
- **役割**: 温度スケジューリングの下限値
- **効果**: 過度な決定性を防ぎ、最終段階でも少量の探索を残す
- **デフォルト**: 0.5
- **調整目安**: 0.5~0.8

### rl_temperature_decay
- **役割**: エポックごとの温度減衰量
- **効果**: 大きいほど探索→活用への切り替えが速い
- **デフォルト**: 0.05
- **調整目安**: 0.02~0.05。不安定→小さめ、早期収束狙い→大きめ

---

## 運用上の注意

- `config.get(...)` により読み込まれる
- 温度関連は `rl_use_sampling: true` のときに有効
- `rl_sampling_temperature` は初期温度として参照され、`rl_min_temperature`・`rl_temperature_decay` と併用してエポック進行に応じて低下

## 現行設定例（探索強め・安定性寄り）

```json
{
  "rl_baseline_momentum": 0.95,
  "rl_entropy_weight": 0.1,
  "rl_use_trajectory_entropy": true,
  "rl_entropy_epsilon": 0.05,
  "rl_use_sampling": true,
  "rl_sampling_temperature": 1.5,
  "rl_min_temperature": 0.8,
  "rl_temperature_decay": 0.035
}
```
