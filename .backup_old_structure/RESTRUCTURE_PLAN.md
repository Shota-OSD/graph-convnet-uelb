# プロジェクト構造改善計画

## 問題点
現在の構造では、**GCN手法**と**RL-KSP手法**が混在しており、どのファイルがどちらの手法に属するか不明確です。

## 目標
- GCN手法とRL-KSP手法を明確に分離
- それぞれ独立して実験・実行可能な構造
- 共通モジュール（データ管理、グラフ処理など）は共有

## 重要な注意事項

### ⚠️ この改善は「実行スクリプトの入口を分ける」だけです

**変更されないこと**:
- ✅ トレーニング+テストの一括実行は**維持されます**
- ✅ 各手法の実行内容（学習→テスト→結果保存）は**変更ありません**
- ✅ 既存の機能・動作は**すべて保持されます**

**変更されること**:
- 📂 ファイル構造が整理され、GCN/RL-KSPが明確に分離されます
- 🚀 実行コマンドが手法ごとに専用化されます（`main.py --mode` → 専用スクリプト）

### 実行方法の変更イメージ

#### Before（現在）
```bash
# 両手法とも main.py から実行（--mode で切り替え）
python main.py --mode gcn --config configs/default.json      # GCN
python main.py --mode rl --config configs/rl_config.json     # RL-KSP
```

#### After（リストラクチャ後）
```bash
# 手法ごとに専用の入口スクリプトを用意
python scripts/gcn/train_gcn.py --config configs/gcn/default.json           # GCN
python scripts/rl_ksp/train_rl_ksp.py --config configs/rl_ksp/rl_config.json # RL-KSP
```

**実行内容は同じ**: どちらも「学習 → テスト → 結果保存」を一括実行します。

---

## 現在の構造分析

### ルートディレクトリのファイル
```
├── main.py                    # 統合メインスクリプト（GCN/RL両対応）
├── deepRLmain.py             # 旧RL専用スクリプト
├── generate_data.py          # データ生成（共通）
├── tune_hyperparameters.py   # ハイパーパラメータ調整（GCN用？）
└── calculate_averages.py     # 結果集計（共通）
```

### src/algorithms/
```
├── beamsearch.py              # ビームサーチ（GCN用）
├── beamsearch_uelb.py        # UELB用ビームサーチ（GCN用）
├── beamsearch_comparator.py  # ビームサーチ比較
├── rl_environment.py         # RL環境（RL-KSP用）
├── rl_trainer.py             # RLトレーナー（RL-KSP用）
└── algorithm_examples.py     # アルゴリズム例
```

### src/models/
```
├── gcn_model.py              # GCNモデル（GCN用）
├── gcn_layers.py             # GCNレイヤー（GCN用）
└── model_utils.py            # モデルユーティリティ（GCN用）
```

### src/train/
```
├── trainer.py                # GCNトレーナー（GCN用）
├── evaluator.py              # GCN評価器（GCN用）
└── metrics.py                # メトリクス記録（共通？）
```

### 共通モジュール
```
src/data_management/          # データ読み込み・生成（共通）
src/graph/                    # グラフ処理（共通）
src/config/                   # 設定管理（共通）
src/visualization/            # 可視化（共通）
```

---

## 新しい構造（提案）

```
graph-convnet-uelb/
│
├── README.md
├── RESTRUCTURE_PLAN.md
│
├── configs/                          # 設定ファイル
│   ├── gcn/                          # GCN用設定
│   │   ├── default.json
│   │   ├── tuning_config.json
│   │   └── nsfnet_*.json
│   │
│   └── rl_ksp/                       # RL-KSP用設定
│       ├── default.json
│       ├── nsfnet_*.json
│       └── rl_config.json
│
├── data/                             # データ（共通）
│   ├── train_data/
│   ├── val_data/
│   └── test_data/
│
├── saved_models/                     # 保存モデル
│   ├── gcn/                          # GCNモデル
│   └── rl_ksp/                       # RL-KSPモデル
│
├── results/                          # 実験結果
│   ├── gcn/
│   └── rl_ksp/
│
├── logs/                             # ログ
│   ├── gcn/
│   └── rl_ksp/
│
├── src/
│   │
│   ├── common/                       # 共通モジュール
│   │   ├── __init__.py
│   │   ├── data_management/         # データ読み込み・生成
│   │   │   ├── __init__.py
│   │   │   ├── dataset_reader.py
│   │   │   ├── dataset_manager.py
│   │   │   ├── data_maker.py
│   │   │   ├── create_data_files.py
│   │   │   ├── exact_solution.py
│   │   │   └── exact_flow.py
│   │   │
│   │   ├── graph/                   # グラフ処理
│   │   │   ├── __init__.py
│   │   │   ├── graph_making.py
│   │   │   ├── graph_utils.py
│   │   │   ├── flow.py
│   │   │   └── k_shortest_path.py
│   │   │
│   │   ├── config/                  # 設定管理
│   │   │   ├── __init__.py
│   │   │   ├── config.py
│   │   │   └── config_manager.py
│   │   │
│   │   ├── visualization/           # 可視化
│   │   │   ├── __init__.py
│   │   │   └── plot_utils.py
│   │   │
│   │   └── utils/                   # 共通ユーティリティ
│   │       ├── __init__.py
│   │       └── metrics.py           # 共通メトリクス
│   │
│   ├── gcn/                          # GCN手法専用
│   │   ├── __init__.py
│   │   ├── models/                   # GCNモデル
│   │   │   ├── __init__.py
│   │   │   ├── gcn_model.py
│   │   │   ├── gcn_layers.py
│   │   │   └── model_utils.py
│   │   │
│   │   ├── algorithms/               # GCN用アルゴリズム
│   │   │   ├── __init__.py
│   │   │   ├── beamsearch.py
│   │   │   ├── beamsearch_uelb.py
│   │   │   └── beamsearch_comparator.py
│   │   │
│   │   ├── train/                    # GCNトレーニング
│   │   │   ├── __init__.py
│   │   │   ├── trainer.py
│   │   │   ├── evaluator.py
│   │   │   └── metrics.py
│   │   │
│   │   └── tuning/                   # GCNハイパーパラメータ調整
│   │       ├── __init__.py
│   │       └── hyperparameter_tuner.py
│   │
│   └── rl_ksp/                       # RL-KSP手法専用
│       ├── __init__.py
│       ├── environment/              # RL環境
│       │   ├── __init__.py
│       │   └── rl_environment.py
│       │
│       ├── models/                   # RLモデル（DQN等）
│       │   ├── __init__.py
│       │   └── dqn_model.py
│       │
│       ├── train/                    # RLトレーニング
│       │   ├── __init__.py
│       │   ├── rl_trainer.py
│       │   └── metrics.py
│       │
│       └── tuning/                   # RLハイパーパラメータ調整
│           ├── __init__.py
│           └── hyperparameter_tuner.py
│
├── scripts/                          # 実行スクリプト
│   ├── gcn/
│   │   ├── train_gcn.py             # GCN学習実行
│   │   ├── test_gcn.py              # GCNテスト実行
│   │   ├── tune_gcn.py              # GCNハイパーパラメータ調整
│   │   └── evaluate_gcn.py          # GCN評価
│   │
│   ├── rl_ksp/
│   │   ├── train_rl_ksp.py          # RL-KSP学習実行
│   │   ├── test_rl_ksp.py           # RL-KSPテスト実行
│   │   ├── tune_rl_ksp.py           # RL-KSPハイパーパラメータ調整
│   │   └── evaluate_rl_ksp.py       # RL-KSP評価
│   │
│   ├── common/
│   │   ├── generate_data.py         # データ生成
│   │   └── calculate_averages.py    # 結果集計
│   │
│   └── comparison/
│       └── compare_methods.py       # GCN vs RL-KSP 比較
│
└── tests/                            # テストコード
    ├── test_gcn/
    ├── test_rl_ksp/
    └── test_common/
```

---

## 既存ファイルとの対応関係

### 新しいスクリプト ← 既存ファイル

| 新しいスクリプト | 既存ファイル | 対応する内容 | 備考 |
|----------------|-------------|------------|------|
| `scripts/gcn/train_gcn.py` | `main.py` の 102-175行 | GCNモード全体 | 学習+テスト一括実行 |
| `scripts/rl_ksp/train_rl_ksp.py` | `main.py` の 56-99行 | RLモード全体 | 学習+テスト一括実行 |
| `scripts/gcn/tune_gcn.py` | `tune_hyperparameters.py` | 全体 | ハイパーパラメータ調整 |
| `scripts/rl_ksp/tune_rl_ksp.py` | - | 新規作成 | RL用チューニング |
| `scripts/common/generate_data.py` | `generate_data.py` | 全体 | そのまま移動 |
| `scripts/common/calculate_averages.py` | `calculate_averages.py` | 全体 | そのまま移動 |
| `scripts/comparison/compare_methods.py` | - | 新規作成 | GCN vs RL比較 |

### 実行コマンドの対応

#### 現在の実行方法
```bash
# GCN実験（学習+テスト一括）
python main.py --mode gcn --config configs/default.json

# RL-KSP実験（学習+テスト一括）
python main.py --mode rl --config configs/rl_config.json

# データ生成
python generate_data.py --config configs/default.json

# ハイパーパラメータ調整（GCN）
python tune_hyperparameters.py --config configs/tuning_config.json
```

#### リストラクチャ後の実行方法
```bash
# GCN実験（学習+テスト一括）
python scripts/gcn/train_gcn.py --config configs/gcn/default.json

# RL-KSP実験（学習+テスト一括）
python scripts/rl_ksp/train_rl_ksp.py --config configs/rl_ksp/rl_config.json

# データ生成
python scripts/common/generate_data.py --config configs/gcn/default.json

# ハイパーパラメータ調整（GCN）
python scripts/gcn/tune_gcn.py --config configs/gcn/tuning_config.json

# ハイパーパラメータ調整（RL-KSP）
python scripts/rl_ksp/tune_rl_ksp.py --config configs/rl_ksp/tuning_config.json
```

### 各スクリプトの実行内容（変更なし）

#### `train_gcn.py` の実行フロー
```
1. 設定ファイル読み込み
2. データセット読み込み
3. GCNモデル初期化
4. 学習実行（複数エポック）
   ├─ 各エポックで検証実行
   └─ ベストモデル保存
5. テスト実行（最終評価）
6. 結果保存（ログ、メトリクス）
```

#### `train_rl_ksp.py` の実行フロー
```
1. 設定ファイル読み込み
2. データセット読み込み
3. RL環境・DQNモデル初期化
4. 学習実行（複数エピソード）
   └─ ε-greedy方策で経路探索
5. テスト実行（学習済みモデルで評価）
6. 結果保存（ログ、メトリクス）
```

**重要**: 両スクリプトとも「学習→テスト→結果保存」を**一括実行**します。

---

## 移行手順

### Phase 1: 共通モジュールの整理
1. **src/common/ ディレクトリを作成**
2. **共通モジュールを移動**
   ```bash
   src/data_management/ → src/common/data_management/
   src/graph/           → src/common/graph/
   src/config/          → src/common/config/
   src/visualization/   → src/common/visualization/
   ```
3. **src/common/utils/ を作成し、共通メトリクスを移動**

### Phase 2: GCN専用モジュールの整理
1. **src/gcn/ ディレクトリを作成**
2. **GCN関連ファイルを移動**
   ```bash
   src/models/          → src/gcn/models/
   src/train/           → src/gcn/train/
   src/tuning/          → src/gcn/tuning/
   ```
3. **src/gcn/algorithms/ を作成**
   ```bash
   src/algorithms/beamsearch*.py → src/gcn/algorithms/
   ```

### Phase 3: RL-KSP専用モジュールの整理
1. **src/rl_ksp/ ディレクトリを作成**
2. **RL関連ファイルを移動**
   ```bash
   src/algorithms/rl_environment.py → src/rl_ksp/environment/
   src/algorithms/rl_trainer.py     → src/rl_ksp/train/
   ```
3. **DQNモデルを分離**
   - `rl_trainer.py` 内の `DQNModel` クラスを `src/rl_ksp/models/dqn_model.py` に分離

### Phase 4: 設定ファイルの整理
1. **configs/gcn/ ディレクトリを作成**
   ```bash
   configs/default*.json        → configs/gcn/
   configs/nsfnet_*_commodities.json → configs/gcn/
   configs/tuning_config.json   → configs/gcn/
   ```
2. **configs/rl_ksp/ ディレクトリを作成**
   ```bash
   configs/nsfnet_*_rl.json     → configs/rl_ksp/
   configs/rl_config.json       → configs/rl_ksp/
   ```

### Phase 5: 実行スクリプトの分離
1. **scripts/gcn/ を作成**
   ```python
   # scripts/gcn/train_gcn.py
   from src.gcn.train.trainer import Trainer
   from src.common.config.config_manager import ConfigManager
   # ... GCN学習コード
   ```
2. **scripts/rl_ksp/ を作成**
   ```python
   # scripts/rl_ksp/train_rl_ksp.py
   from src.rl_ksp.train.rl_trainer import RLTrainer
   from src.common.config.config_manager import ConfigManager
   # ... RL-KSP学習コード
   ```
3. **main.py を廃止または統合スクリプトに変更**

### Phase 6: インポート文の更新
すべてのファイルでインポート文を新しい構造に合わせて更新

**例: GCNファイル内**
```python
# 旧
from ..data_management.dataset_reader import DatasetReader
from ..models.gcn_model import GCNModel

# 新
from src.common.data_management.dataset_reader import DatasetReader
from src.gcn.models.gcn_model import GCNModel
```

**例: RL-KSPファイル内**
```python
# 旧
from ..data_management.dataset_reader import DatasetReader
from .rl_environment import MinMaxLoadKSPsEnv

# 新
from src.common.data_management.dataset_reader import DatasetReader
from src.rl_ksp.environment.rl_environment import MinMaxLoadKSPsEnv
```

### Phase 7: テストと検証
1. **各手法を独立して実行**
   ```bash
   # GCN実行
   python scripts/gcn/train_gcn.py --config configs/gcn/default.json

   # RL-KSP実行
   python scripts/rl_ksp/train_rl_ksp.py --config configs/rl_ksp/rl_config.json
   ```
2. **データ生成が両手法で動作確認**
   ```bash
   python scripts/common/generate_data.py --config configs/gcn/default.json
   ```

---

## 実装チェックリスト

### Phase 1: 共通モジュール (✓ = 完了)
- [ ] `src/common/` ディレクトリ作成
- [ ] `src/common/data_management/` に移動
- [ ] `src/common/graph/` に移動
- [ ] `src/common/config/` に移動
- [ ] `src/common/visualization/` に移動
- [ ] `src/common/utils/` 作成（metrics.py等）

### Phase 2: GCN専用モジュール
- [ ] `src/gcn/` ディレクトリ作成
- [ ] `src/gcn/models/` に移動
- [ ] `src/gcn/train/` に移動
- [ ] `src/gcn/tuning/` に移動
- [ ] `src/gcn/algorithms/` 作成（beamsearch系）

### Phase 3: RL-KSP専用モジュール
- [ ] `src/rl_ksp/` ディレクトリ作成
- [ ] `src/rl_ksp/environment/` 作成（rl_environment.py）
- [ ] `src/rl_ksp/models/` 作成（dqn_model.py）
- [ ] `src/rl_ksp/train/` 作成（rl_trainer.py）
- [ ] `src/rl_ksp/tuning/` 作成

### Phase 4: 設定ファイル
- [ ] `configs/gcn/` ディレクトリ作成
- [ ] GCN用設定ファイルを移動
- [ ] `configs/rl_ksp/` ディレクトリ作成
- [ ] RL-KSP用設定ファイルを移動

### Phase 5: 実行スクリプト
- [ ] `scripts/` ディレクトリ作成
- [ ] `scripts/gcn/` に実行スクリプト作成
- [ ] `scripts/rl_ksp/` に実行スクリプト作成
- [ ] `scripts/common/` に共通スクリプト移動
- [ ] `scripts/comparison/` に比較スクリプト作成

### Phase 6: インポート更新
- [ ] GCNファイルのインポート更新
- [ ] RL-KSPファイルのインポート更新
- [ ] 共通モジュールのインポート更新

### Phase 7: テスト
- [ ] GCN学習が正常に動作
- [ ] RL-KSP学習が正常に動作
- [ ] データ生成が正常に動作
- [ ] 両手法の結果が正しく保存される

---

## 期待される効果

### 1. 明確な分離
- GCN手法とRL-KSP手法がディレクトリレベルで明確に分離
- どのファイルがどちらの手法に属するか一目瞭然

### 2. 独立性の向上
- 各手法を独立して開発・実験可能
- 一方の変更が他方に影響しない

### 3. 保守性の向上
- コードの役割が明確になり、バグ修正が容易
- 新しい手法の追加が容易（`src/new_method/` を追加するだけ）

### 4. 実行の簡潔化
```bash
# GCN実験
python scripts/gcn/train_gcn.py --config configs/gcn/default.json

# RL-KSP実験
python scripts/rl_ksp/train_rl_ksp.py --config configs/rl_ksp/rl_config.json

# 比較実験
python scripts/comparison/compare_methods.py
```

---

## 注意事項

### 互換性の維持
- 既存の保存モデル・結果ファイルへのアクセスを維持
- 設定ファイルの形式は変更しない（パスのみ変更）

### 段階的な移行
- 一度にすべて変更せず、Phase単位で実施
- 各Phaseでテストを実行して動作確認

### ドキュメント更新
- README.mdを新しい構造に合わせて更新
- 実行方法のドキュメントを作成

---

## タイムライン（推定）

| Phase | 作業内容 | 推定時間 |
|-------|---------|---------|
| Phase 1 | 共通モジュール整理 | 1-2時間 |
| Phase 2 | GCN専用モジュール整理 | 1-2時間 |
| Phase 3 | RL-KSP専用モジュール整理 | 1-2時間 |
| Phase 4 | 設定ファイル整理 | 30分 |
| Phase 5 | 実行スクリプト作成 | 2-3時間 |
| Phase 6 | インポート更新 | 2-3時間 |
| Phase 7 | テスト・検証 | 1-2時間 |
| **合計** | | **9-15時間** |

---

## よくある質問（FAQ）

### Q1: トレーニングとテストを別々に実行できますか？
**A**: はい、可能です。リストラクチャ後、必要に応じて以下のような分離スクリプトを追加できます：

```bash
# 学習のみ
python scripts/gcn/train_only_gcn.py --config configs/gcn/default.json

# テストのみ（保存済みモデルを使用）
python scripts/gcn/test_only_gcn.py --config configs/gcn/default.json --model saved_models/gcn/model_latest.pt
```

ただし、**デフォルトでは従来通り「学習+テスト一括」のスクリプトのみ提供**します。

### Q2: 既存のコマンド（`python main.py --mode gcn`）は使えなくなりますか？
**A**: リストラクチャ完了後は非推奨になりますが、互換性のため`main.py`を残すことも可能です：

```python
# main.py（互換性維持版）
import sys
print("警告: main.py は非推奨です。以下のコマンドを使用してください:")
if '--mode gcn' in ' '.join(sys.argv):
    print("  python scripts/gcn/train_gcn.py --config <config_file>")
elif '--mode rl' in ' '.join(sys.argv):
    print("  python scripts/rl_ksp/train_rl_ksp.py --config <config_file>")
```

### Q3: データ生成は両手法で共通ですか？
**A**: はい、共通です。`scripts/common/generate_data.py`を使用します。

```bash
# GCN用データ生成
python scripts/common/generate_data.py --config configs/gcn/default.json

# RL-KSP用データ生成（同じコマンド）
python scripts/common/generate_data.py --config configs/rl_ksp/rl_config.json
```

### Q4: 保存済みモデルの場所は変わりますか？
**A**: はい、手法ごとに分離されます：

```
Before: saved_models/model_xxxxx_latest.pt
After:  saved_models/gcn/gcn_model_xxxxx_latest.pt
        saved_models/rl_ksp/rl_model_xxxxx_latest.pt
```

既存モデルは手動で移動するか、互換性スクリプトで自動移行できます。

### Q5: どのくらいの作業時間がかかりますか？
**A**: 推定9-15時間です（Phase 1-7の合計）。段階的に実施するため、途中で中断・再開も可能です。

---

## まとめ

### この改善の本質
- **目的**: GCN手法とRL-KSP手法を**ファイル構造で明確に分離**する
- **方法**: 実行スクリプトの入口を手法ごとに専用化する
- **重要**: トレーニング+テストの一括実行は**維持される**（動作変更なし）

### ユーザーから見た変化
| 項目 | Before | After |
|------|--------|-------|
| **実行コマンド** | `python main.py --mode gcn` | `python scripts/gcn/train_gcn.py` |
| **ファイル構造** | GCN/RL混在 | GCN/RL明確に分離 |
| **実行内容** | 学習+テスト一括 | 学習+テスト一括（同じ） |
| **データ共有** | 共通 | 共通（変わらず） |

### 次のステップ
1. この計画を確認・承認
2. Phase 1から順次実装開始
3. 各Phase完了時に動作確認
4. 全Phase完了後、READMEとドキュメント更新

---

**更新履歴**:
- 2025-10-16: 初版作成
- 2025-10-16: 「実行内容は変わらない」ことを明記、FAQ追加
