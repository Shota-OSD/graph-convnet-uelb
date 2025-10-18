# GNNモデルの評価を最大負荷率を使った強化学習に変更する計画

## 目次
1. [現状分析](#現状分析)
2. [問題点](#問題点)
3. [変更計画](#変更計画)
4. [実装手順](#実装手順)
5. [期待される効果](#期待される効果)

---

## 現状分析

### 1. 強化学習（RL）モードの実装状況

#### ファイル: `src/algorithms/rl_environment.py`
- **既に実装済み**: 最大負荷率を使用した強化学習環境
- **主要メソッド**:
  - `load_factor(grouping)` (145-155行目): 各エッジの負荷率を計算
  - `max_load_factor(grouping)` (157-160行目): 最大負荷率を計算
  - `get_reward_maxload(grouping)` (168-170行目): 報酬定義1（最大負荷率ベース）
  - `get_reward_difference(grouping, bf_maxloadfactor)` (172-176行目): 報酬定義2（差分ベース）

#### ファイル: `src/algorithms/rl_trainer.py`
- **DQNモデル**: Deep Q-Networkで学習
- **評価方法**:
  - テスト時に `max_load_factor` を直接計算
  - 厳密解との比較: `approximation_rate = gt_load_factor / max_load * 100` (452行目)
  - 100%を超える場合はエラーとして強制終了 (462-472行目)

### 2. GCNモードの実装状況

#### ファイル: `src/train/evaluator.py`
- **評価方法**: ビームサーチで経路を予測
- **負荷率計算**: `mean_feasible_load_factor` (72行目)
- **近似率計算**: `approximation_rate = mean_gt_load_factor / epoch_mean_maximum_load_factor * 100` (94行目)

#### ファイル: `src/models/model_utils.py`
- GCNモデルの負荷率計算用ユーティリティ関数が存在（推定）

---

## 問題点

### 評価指標の不一致

| モード | 負荷率計算方法 | 近似率計算方法 | データソース |
|--------|---------------|---------------|-------------|
| **RL** | `max_load_factor()` を直接計算 | `gt_load_factor / rl_max_load * 100` | 環境内で計算 |
| **GCN** | `mean_feasible_load_factor()` | `mean_gt_load_factor / predicted_load_factor * 100` | ビームサーチ結果 |

**問題**:
- 計算方法が異なるため、RLとGCNの性能を直接比較できない
- 評価基準が統一されていないため、どちらが優れているか判断しにくい

---

## 変更計画

### 目標
**RLとGCNの評価方法を統一し、同じ基準で性能比較できるようにする**

### 変更対象ファイル

#### 1. `src/train/evaluator.py` の修正
**目的**: GCNの評価方法をRLと統一

**変更箇所**:
```python
# 現在 (72-94行目)
mean_maximum_load_factor, _ = mean_feasible_load_factor(...)
approximation_rate = mean_gt_load_factor / epoch_mean_maximum_load_factor * 100

# 変更後: RLと同じ計算方法を使用
# - max_load_factor を直接計算
# - gt_load_factor / predicted_load_factor * 100
```

**具体的な変更**:
- [ ] `mean_feasible_load_factor` の計算方法を確認
- [ ] RLの `max_load_factor` と同じロジックに変更
- [ ] 近似率の計算式を統一

#### 2. `src/train/metrics.py` の拡張
**目的**: RL用のメトリクスを統合

**追加内容**:
```python
class MetricsLogger:
    def __init__(self):
        # 既存のメトリクス
        self.train_loss_list = []
        self.train_err_edges_list = []
        self.val_approximation_rate_list = []
        self.test_approximation_rate_list = []

        # RL用メトリクスを追加
        self.rl_train_reward_list = []         # 学習時の報酬
        self.rl_max_load_factor_list = []      # 最大負荷率
        self.rl_episode_steps_list = []        # エピソードのステップ数

    def log_rl_metrics(self, reward, max_load, steps):
        """RL用メトリクスを記録"""
        self.rl_train_reward_list.append(reward)
        self.rl_max_load_factor_list.append(max_load)
        self.rl_episode_steps_list.append(steps)
```

**変更箇所**:
- [ ] RL用メトリクスのフィールドを追加 (11-24行目付近)
- [ ] `log_rl_metrics()` メソッドを追加
- [ ] `save_results()` でRL用メトリクスも保存 (79-195行目)
- [ ] `print_summary()` でRL用メトリクスも表示 (197-224行目)

#### 3. `configs/tuning_config.json` の拡張
**目的**: RL用ハイパーパラメータを追加

**追加セクション**:
```json
{
  "reinforcement_learning": {
    "description": "Reinforcement Learning specific parameters",
    "parameters": {
      "K": {
        "type": "choice",
        "choices": [5, 10, 15, 20],
        "description": "Number of k-shortest paths"
      },
      "n_action": {
        "type": "choice",
        "choices": [10, 15, 20, 25, 30],
        "description": "Number of action candidates"
      },
      "reward_state": {
        "type": "choice",
        "choices": [1, 2],
        "description": "1: max load factor, 2: difference-based"
      },
      "initial_state": {
        "type": "choice",
        "choices": [1, 2],
        "description": "1: shortest path, 2: random"
      },
      "epsilon": {
        "type": "float",
        "min": 0.5,
        "max": 0.9,
        "log_scale": false,
        "description": "Exploration rate"
      },
      "epsilon_decay": {
        "type": "float",
        "min": 0.99,
        "max": 0.999,
        "log_scale": false,
        "description": "Epsilon decay rate"
      },
      "gamma": {
        "type": "float",
        "min": 0.8,
        "max": 0.95,
        "log_scale": false,
        "description": "Discount factor"
      },
      "hidden_dims": {
        "type": "choice",
        "choices": [[32, 32, 32], [64, 64, 64], [32, 64, 32], [128, 128, 128]],
        "description": "Hidden layer dimensions for DQN"
      },
      "episodes": {
        "type": "choice",
        "choices": [500, 1000, 2000],
        "description": "Number of training episodes"
      },
      "max_step": {
        "type": "choice",
        "choices": [10, 20, 30],
        "description": "Maximum steps per episode"
      }
    }
  },

  "evaluation": {
    "primary_metric": "test_approximation_rate",
    "secondary_metrics": [
      "val_approximation_rate",
      "test_infeasible_rate",
      "rl_max_load_factor",
      "rl_episode_steps"
    ],
    "optimization_direction": "maximize"
  }
}
```

**変更箇所**:
- [ ] `reinforcement_learning` セクションを追加
- [ ] `evaluation.secondary_metrics` にRL用メトリクスを追加

#### 4. 統合テストスクリプトの作成
**新規ファイル**: `compare_gcn_rl.py`

**目的**: GCNとRLを同じデータで評価し、結果を比較

**実装内容**:
```python
#!/usr/bin/env python3
"""
GCNとRLモードの性能比較スクリプト
"""

import argparse
import json
from src.config.config_manager import ConfigManager
from src.train.trainer import Trainer
from src.train.evaluator import Evaluator
from src.algorithms.rl_trainer import RLTrainer
from src.train.metrics import MetricsLogger

def compare_models(config_path):
    """GCNとRLを同じデータで評価して比較"""

    # 設定読み込み
    config_manager = ConfigManager(config_path)
    config = config_manager.get_config()

    results = {
        'gcn': {},
        'rl': {}
    }

    # GCNモードで評価
    print("="*60)
    print("GCN MODE EVALUATION")
    print("="*60)
    trainer = Trainer(config, ...)
    evaluator = Evaluator(config, ...)
    metrics_logger = MetricsLogger()

    # テスト実行
    test_result = evaluator.evaluate(trainer.get_model(), None, mode='test')
    results['gcn'] = {
        'approximation_rate': test_result[4],
        'infeasible_rate': test_result[5],
        'execution_time': test_result[0]
    }

    # RLモードで評価
    print("\n" + "="*60)
    print("RL MODE EVALUATION")
    print("="*60)
    rl_trainer = RLTrainer(dict(config))
    rl_trainer.test(test_episodes=config.get('num_test_data', 20))

    results['rl'] = {
        'approximation_rate': ...,  # RLの結果から取得
        'max_load_factor': ...,
        'execution_time': ...
    }

    # 結果の比較表示
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    print(f"{'Metric':<30} {'GCN':<15} {'RL':<15}")
    print("-"*60)
    print(f"{'Approximation Rate (%)':<30} {results['gcn']['approximation_rate']:<15.2f} {results['rl']['approximation_rate']:<15.2f}")
    print(f"{'Infeasible Rate (%)':<30} {results['gcn']['infeasible_rate']:<15.2f} {'-':<15}")
    print(f"{'Execution Time (s)':<30} {results['gcn']['execution_time']:<15.2f} {results['rl']['execution_time']:<15.2f}")

    # 結果をJSONで保存
    with open('comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default2.json')
    args = parser.parse_args()

    compare_models(args.config)
```

---

## 実装手順

### Phase 1: 評価方法の統一
1. **`evaluator.py` の分析**
   - `mean_feasible_load_factor()` の実装を確認
   - RLの `max_load_factor()` との違いを特定

2. **`evaluator.py` の修正**
   - 負荷率計算をRLと同じロジックに変更
   - 近似率計算式を統一
   - テストを実行して動作確認

### Phase 2: メトリクスの統合
1. **`metrics.py` の拡張**
   - RL用フィールドを追加
   - `log_rl_metrics()` メソッドを実装
   - `save_results()` と `print_summary()` を更新

2. **`rl_trainer.py` の修正**
   - `MetricsLogger` を使用するように変更
   - 学習・テスト時にメトリクスを記録

### Phase 3: 設定ファイルの統合
1. **`tuning_config.json` の更新**
   - `reinforcement_learning` セクションを追加
   - RL用パラメータを定義
   - 評価メトリクスにRL項目を追加

### Phase 4: 比較スクリプトの作成
1. **`compare_gcn_rl.py` の実装**
   - GCNとRLを同じデータで評価
   - 結果を比較表示
   - JSONで結果を保存

### Phase 5: テストと検証
1. **統合テスト**
   - 小規模データでGCNとRLを実行
   - 評価指標が正しく計算されているか確認
   - 近似率が妥当な範囲（80-100%）にあるか確認

2. **性能比較**
   - 同じテストデータでGCNとRLを実行
   - 近似率、実行時間、不実行率を比較
   - 結果をドキュメント化

---

## 期待される効果

### 1. 統一された評価基準
- RLとGCNを同じ基準で比較可能
- どちらの手法が優れているか明確に判断できる

### 2. ハイパーパラメータチューニングの改善
- RL用パラメータも `tuning_config.json` で管理
- グリッドサーチやランダムサーチでRL最適化が可能

### 3. 研究・開発の効率化
- 統一された評価フレームワークで新手法の評価が容易
- 比較スクリプトで自動的に性能比較が可能

### 4. 結果の信頼性向上
- 評価方法が一貫しているため、結果の再現性が高い
- 論文やレポートでの結果報告が明確になる

---

## 実装チェックリスト

### Phase 1: 評価方法の統一
- [ ] `mean_feasible_load_factor()` の実装を確認
- [ ] `evaluator.py` の負荷率計算をRLと統一
- [ ] 近似率計算式を変更
- [ ] テストデータで動作確認

### Phase 2: メトリクスの統合
- [ ] `metrics.py` にRL用フィールドを追加
- [ ] `log_rl_metrics()` を実装
- [ ] `save_results()` を更新
- [ ] `print_summary()` を更新
- [ ] `rl_trainer.py` でMetricsLoggerを使用

### Phase 3: 設定ファイルの統合
- [ ] `tuning_config.json` にRLセクションを追加
- [ ] RL用パラメータを定義
- [ ] 評価メトリクスを更新

### Phase 4: 比較スクリプトの作成
- [ ] `compare_gcn_rl.py` を作成
- [ ] GCN評価部分を実装
- [ ] RL評価部分を実装
- [ ] 結果比較・表示部分を実装
- [ ] JSON保存機能を実装

### Phase 5: テストと検証
- [ ] 小規模データでテスト実行
- [ ] 評価指標の妥当性確認
- [ ] 大規模データで性能比較
- [ ] 結果をドキュメント化

---

## 参考資料

### 関連ファイル
- `src/algorithms/rl_environment.py`: RL環境の実装
- `src/algorithms/rl_trainer.py`: RLトレーナーの実装
- `src/train/evaluator.py`: GCN評価器の実装
- `src/train/metrics.py`: メトリクス記録器
- `configs/tuning_config.json`: ハイパーパラメータ設定

### 重要な計算式
```python
# 負荷率計算（RL）
load = sum((zo_combination[l][e]) * (commodity_list[l][2]) for l in range(commodity)) / capacity_list[edge_list[e][1]]

# 最大負荷率
max_load_factor = max(loads)

# 近似率（統一後）
approximation_rate = gt_load_factor / predicted_load_factor * 100
```

### 評価基準
- **近似率**: 80-100% が理想的（100%超過は異常）
- **不実行率**: 0% が理想的（実行可能解が見つからない割合）
- **実行時間**: できるだけ短い方が望ましい

---

## 更新履歴
- 2025-10-16: 初版作成
