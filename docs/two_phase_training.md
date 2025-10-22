# 2段階学習: 教師あり事前学習 + 強化学習ファインチューニング

## 概要

このアプローチは、強化学習の不安定性を解決するために、教師あり学習で事前学習したモデルを強化学習でファインチューニングします。

```
Phase 1: 教師あり事前学習 (Supervised Pre-training)
  └─> 最適解を教師データとして、良い経路選択を学習

Phase 2: 強化学習ファインチューニング (RL Fine-tuning)
  └─> Phase 1の重みから開始し、Load Factor最適化で微調整
```

## 期待される効果

| メトリクス | RL単独 | 2段階学習 | 改善 |
|-----------|--------|----------|------|
| Complete Rate (初期) | 77-80% | **85-90%** | +8-10% |
| Complete Rate (最終) | 81-83% | **90-95%** | +9-12% |
| Reward (Epoch 1) | 0.7-1.7 | **3.0-5.0** | 2-3倍 |
| 学習安定性 | 不安定 | **安定** | 改善 |
| 収束速度 | 25-100 epochs | **15-25 epochs** | 高速化 |

## 技術的な詳細

### モデル出力の違い

```
教師あり学習 (Phase 1):
  voc_edges_out = 2
  出力形状: [batch, nodes, nodes, commodities, 2]
  意味: [P(使わない), P(使う)]
  損失関数: Cross Entropy

強化学習 (Phase 2):
  voc_edges_out = 1
  出力形状: [batch, nodes, nodes, commodities]
  意味: エッジスコア (連続値)
  損失関数: REINFORCE (Policy Gradient)
```

### モデル変換: Logit差分法

Phase 1からPhase 2へのモデル変換には**Logit差分法**を使用します：

```python
# 教師あり学習の重み
W0: class 0 (使わない) の重み [hidden_dim]
W1: class 1 (使う) の重み [hidden_dim]

# 強化学習の重み (変換後)
W_rl = W1 - W0

# なぜこれが良いか:
# 教師あり: logit_diff = (W1*h+b1) - (W0*h+b0)
#         = (W1-W0)*h + (b1-b0)
# 強化学習: score = W_rl*h + b_rl
#         = 同じ値！

# 意味:
# score > 0 ⟺ P(使う) > 0.5
# score < 0 ⟺ P(使わない) > 0.5
```

この方法により、教師あり学習で学んだ**相対的な確信度**が完全に保持されます。

## 使用方法

### 方法1: 統合スクリプトで実行 (推奨)

```bash
# 両フェーズを自動実行
python scripts/gcn/two_phase_training.py

# カスタム設定を使用
python scripts/gcn/two_phase_training.py \
  --supervised-config configs/gcn/my_supervised.json \
  --rl-config configs/gcn/my_rl.json

# Phase 1をスキップ (既存モデルを使用)
python scripts/gcn/two_phase_training.py --skip-supervised
```

### 方法2: 各フェーズを個別に実行

**Phase 1: 教師あり事前学習**
```bash
python scripts/gcn/train_gcn.py \
  --config configs/gcn/supervised_pretraining.json
```

この後、モデルが `saved_models/supervised_pretrained.pt` に保存されます。

**Phase 2: 強化学習ファインチューニング**
```bash
python scripts/gcn/train_gcn.py \
  --config configs/gcn/rl_finetuning.json
```

設定ファイル `rl_finetuning.json` で以下が指定されている必要があります：
```json
{
  "load_pretrained_model": true,
  "pretrained_model_path": "saved_models/supervised_pretrained.pt",
  "convert_supervised_to_rl": true
}
```

## 設定ファイル

### supervised_pretraining.json

```json
{
  "expt_name": "supervised_pretraining",
  "training_strategy": "supervised",
  "voc_edges_out": 2,
  "max_epochs": 30,
  // ... その他の設定
}
```

重要なパラメータ:
- `voc_edges_out: 2` - Binary分類用
- `training_strategy: "supervised"` - 教師あり学習を使用
- `max_epochs: 20-30` - 十分な学習のため

### rl_finetuning.json

```json
{
  "expt_name": "rl_finetuning",
  "training_strategy": "reinforcement",
  "voc_edges_out": 1,
  "max_epochs": 35,
  "load_pretrained_model": true,
  "pretrained_model_path": "saved_models/supervised_pretrained.pt",
  "convert_supervised_to_rl": true,
  // ... RL設定
}
```

重要なパラメータ:
- `voc_edges_out: 1` - スコア出力用
- `training_strategy: "reinforcement"` - 強化学習を使用
- `load_pretrained_model: true` - 事前学習モデルを読み込む
- `convert_supervised_to_rl: true` - Logit差分法で変換

## 実装ファイル

```
configs/gcn/
  ├── supervised_pretraining.json   # Phase 1 設定
  └── rl_finetuning.json           # Phase 2 設定

src/gcn/utils/
  └── model_converter.py           # モデル変換ユーティリティ
      ├── convert_supervised_to_rl()      # Logit差分法による変換
      ├── load_pretrained_supervised_model()  # 事前学習モデル読み込み
      └── verify_conversion()             # 変換の検証

src/gcn/train/
  └── trainer.py                   # 事前学習モデル読み込み機能追加

scripts/gcn/
  └── two_phase_training.py        # 統合実行スクリプト
```

## トラブルシューティング

### Q: Phase 1の後、モデルが保存されない

**A:** `supervised_pretraining.json` に以下を追加してください:
```json
{
  "save_model": true,
  "models_dir": "./saved_models",
  "model_name": "supervised_pretrained.pt"
}
```

### Q: Phase 2で "model not found" エラー

**A:** 以下を確認:
1. Phase 1が正常に完了したか
2. `pretrained_model_path` が正しいか
3. モデルファイルが存在するか (`ls saved_models/`)

### Q: 変換後の性能が悪い

**A:** 変換検証を実行してください:
```python
from src.gcn.utils.model_converter import verify_conversion

# サンプルデータで検証
results = verify_conversion(supervised_model, rl_model, sample_data)
# correlation, rank_correlation が 0.99以上であることを確認
```

### Q: Phase 1の学習が収束しない

**A:** 以下を試してください:
- エポック数を増やす (30 → 50)
- 学習率を下げる (0.001 → 0.0005)
- バッチサイズを増やす (20 → 40)

## ベストプラクティス

1. **Phase 1のエポック数**: 30エポック推奨
   - 少なすぎると十分に学習できない
   - 多すぎても過学習のリスク

2. **Phase 2の学習率**: Phase 1と同じか少し低め
   - 事前学習済みの重みを壊さないため

3. **Entropy weight**: Phase 2では 0.1 を推奨
   - 事前学習により探索が少なくても良い性能

4. **検証**: 変換後は必ず検証を実行
   - `verify_conversion()` で相関係数をチェック
   - 0.99以上であることを確認

## 参考文献

この手法は以下の研究で広く使用されています:

- **Imitation Learning → RL**: AlphaGo, AlphaZero
- **Supervised Pre-training → Fine-tuning**: BERT, GPT系モデル
- **Behavioral Cloning → RL**: ロボット制御、自動運転

## まとめ

2段階学習により:
- ✅ 安定した初期化 (良い方策の近くから開始)
- ✅ 高速な収束 (事前知識の活用)
- ✅ 高い最終性能 (より良い局所最適解)
- ✅ 学習の安定性向上 (探索の効率化)

が期待できます。
