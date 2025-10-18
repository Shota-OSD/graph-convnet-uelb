# リファクタリング進捗状況

## 完了したフェーズ

### ✅ Phase 1: 共通モジュールの整理（完了）
- `src/common/` ディレクトリ作成
- 以下のモジュールを `src/common/` に移動:
  - `data_management/` - データ読み込み・生成
  - `graph/` - グラフ処理
  - `config/` - 設定管理
  - `visualization/` - 可視化
  - `utils/` - 共通ユーティリティ（新規作成）

### ✅ Phase 2: GCN専用モジュールの整理（完了）
- `src/gcn/` ディレクトリ作成
- 以下のモジュールを `src/gcn/` に移動:
  - `models/` - GCNモデル、レイヤー
  - `train/` - トレーナー、評価器、メトリクス
  - `tuning/` - ハイパーパラメータ調整
  - `algorithms/` - ビームサーチ関連

### ✅ Phase 3: RL-KSP専用モジュールの整理（完了）
- `src/rl_ksp/` ディレクトリ作成
- 以下のモジュールを `src/rl_ksp/` に作成・移動:
  - `environment/` - RL環境（`rl_environment.py`）
  - `models/` - DQNモデル（`rl_trainer.py`から分離）
  - `train/` - RLトレーナー
  - `tuning/` - ハイパーパラメータ調整（準備のみ）

### ✅ Phase 4: 設定ファイルの整理（完了）
- `configs/gcn/` ディレクトリ作成
  - GCN用設定ファイル（13ファイル）を移動
- `configs/rl_ksp/` ディレクトリ作成
  - RL-KSP用設定ファイル（7ファイル）を移動

### ✅ Phase 5: 実行スクリプトの作成（完了）
- `scripts/gcn/train_gcn.py` - GCN学習+テスト一括実行
- `scripts/rl_ksp/train_rl_ksp.py` - RL-KSP学習+テスト一括実行
- `scripts/common/generate_data.py` - データ生成（共通）
- `scripts/common/calculate_averages.py` - 結果集計（共通）

---

## 現在の課題

### ⚠️ Phase 6: インポート文の更新（未完了）

**問題**: 新しい構造に移動したファイル内のインポート文が古いパスを参照しています。

**影響を受けるファイル**:
1. **GCNモジュール** (`src/gcn/`)
   - `train/trainer.py`
   - `train/evaluator.py`
   - `models/gcn_model.py`
   - `algorithms/beamsearch_uelb.py`

2. **RL-KSPモジュール** (`src/rl_ksp/`)
   - `train/rl_trainer.py`
   - `environment/rl_environment.py`

3. **実行スクリプト** (`scripts/`)
   - `gcn/train_gcn.py`
   - `rl_ksp/train_rl_ksp.py`

**必要な変更例**:
```python
# 旧（src/train/trainer.py）
from ..data_management.dataset_reader import DatasetReader
from ..models.gcn_model import GCNModel

# 新（src/gcn/train/trainer.py）
from src.common.data_management.dataset_reader import DatasetReader
from src.gcn.models.gcn_model import GCNModel
```

---

## 新しい実行方法

### GCN実験
```bash
# 学習+テスト一括実行
python scripts/gcn/train_gcn.py --config configs/gcn/default.json
```

### RL-KSP実験
```bash
# 学習+テスト一括実行
python scripts/rl_ksp/train_rl_ksp.py --config configs/rl_ksp/rl_config.json
```

### データ生成
```bash
python scripts/common/generate_data.py --config configs/gcn/default.json
```

---

## 次のステップ

### Phase 6の完了に必要な作業

1. **GCNモジュールのインポート更新**
   - [ ] `src/gcn/train/trainer.py`
   - [ ] `src/gcn/train/evaluator.py`
   - [ ] `src/gcn/models/gcn_model.py`
   - [ ] `src/gcn/models/model_utils.py`
   - [ ] `src/gcn/algorithms/beamsearch_uelb.py`

2. **RL-KSPモジュールのインポート更新**
   - [ ] `src/rl_ksp/train/rl_trainer.py` - DQNModelのインポート追加
   - [ ] `src/rl_ksp/environment/rl_environment.py`

3. **動作確認**
   - [ ] GCN学習スクリプトの実行テスト
   - [ ] RL-KSP学習スクリプトの実行テスト

### Phase 7: ドキュメント作成

- [ ] README.mdの更新（新しい構造を反映）
- [ ] 実行方法の説明
- [ ] 移行ガイド（旧→新コマンド対応表）

---

## 旧構造との対応（参考）

| 旧ファイル/ディレクトリ | 新ファイル/ディレクトリ |
|----------------------|---------------------|
| `main.py --mode gcn` | `scripts/gcn/train_gcn.py` |
| `main.py --mode rl` | `scripts/rl_ksp/train_rl_ksp.py` |
| `src/data_management/` | `src/common/data_management/` |
| `src/graph/` | `src/common/graph/` |
| `src/models/` | `src/gcn/models/` |
| `src/train/` | `src/gcn/train/` |
| `src/algorithms/beamsearch*.py` | `src/gcn/algorithms/` |
| `src/algorithms/rl_*.py` | `src/rl_ksp/` |
| `configs/*.json` | `configs/gcn/*.json` or `configs/rl_ksp/*.json` |
| `generate_data.py` | `scripts/common/generate_data.py` |

---

## 作業履歴

- **2025-10-16 16:37**: Phase 1-5 完了
- **2025-10-16**: インポート更新作業を開始

---

## 補足

### 旧ファイルの扱い
- 旧構造のファイルは**まだ削除していません**
- 新構造で動作確認後に削除予定
- 当面は両方の構造が共存します

### テスト方針
1. インポート文を全て更新
2. GCNスクリプトで小規模データをテスト実行
3. RL-KSPスクリプトで小規模データをテスト実行
4. 問題なければ旧ファイルを削除
