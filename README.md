# Graph ConvNet UELB - Training Pipeline

このリポジトリは、グラフニューラルネットワーク（GCN）を用いたUELB問題の学習・評価パイプラインを提供します。

## 必要条件
- Python 3.9以降
- CUDA対応GPU（推奨）
- Conda（MinicondaまたはAnaconda）

## セットアップ

### 1. Conda環境の作成とアクティベート

```sh
# 環境の作成
conda env create -f environment.yml

# 環境のアクティベート
conda activate gcn-uelb-env
```

### 2. 環境の確認

```sh
# 環境が正しく作成されたか確認
conda list
```

### 3. 設定ファイルの編集

設定ファイル（例: `configs/default2.json`）を編集して、学習条件やデータパスを調整します。

## 使い方

### 1. データ生成

まず、学習・評価用のデータセットを生成します：

```sh
# デフォルト設定でデータ生成
python generate_data.py

# 特定の設定ファイルでデータ生成
python generate_data.py --config configs/nsfnet_15_commodities.json

# 特定のモードのみ生成
python generate_data.py --config configs/nsfnet_15_commodities.json --modes train val

# 確認なしで強制実行
python generate_data.py --config configs/nsfnet_15_commodities.json --force

# 既存データの削除のみ
python generate_data.py --clean-only
```

### 2. 学習・評価の実行

データ生成後、モデルの学習と評価を実行します：

```sh
# GCNモード（デフォルト）
python main.py --config configs/nsfnet_15_commodities.json

# 強化学習モード
python main.py --config configs/nsfnet_15_commodities_rl.json --mode rl
```

- データが存在しない場合は、適切なエラーメッセージとともにデータ生成コマンドが表示されます
- ログやモデルの重みは`saved_models/`ディレクトリに保存されます

### 3. モデルの保存・再利用機能

#### 新規トレーニング（モデル保存あり）
```sh
python main.py --config configs/with_model_saving.json
```

#### 保存済みモデルの再利用
```sh
# 最新のモデルを自動読み込み
python main.py --config configs/load_saved_model.json

# 特定のエポックのモデルを指定読み込み
python main.py --config configs/load_specific_epoch.json
```

#### モデル保存の設定オプション

設定ファイルに以下のオプションを追加することで、モデルの保存・読み込み動作をカスタマイズできます：

```json
{
  "models_dir": "./saved_models",        // モデル保存ディレクトリ
  "save_model": true,                    // モデル保存の有効/無効
  "save_every_epoch": true,              // 毎エポックでモデル保存
  "load_saved_model": true,              // 起動時に保存済みモデルを読み込み
  "load_model_epoch": 5,                 // 特定エポックのモデルを読み込み（省略時は最新版）
  "cleanup_old_models": true             // 古いモデルファイルの自動削除
}
```

#### モデル選択の仕組み

- **自動識別**: モデル構造の設定（`hidden_dim`, `num_layers`等）から生成されるハッシュで同一構造のモデルのみ読み込み
- **ファイル形式**: 
  - 最新版: `model_{hash}_latest.pt`
  - エポック別: `model_{hash}_epoch_{番号}.pt`
- **選択ロジック**:
  - `load_model_epoch`**未指定** → 最新版(`latest.pt`)を自動選択
  - `load_model_epoch: 5` → エポック5のモデル(`epoch_5.pt`)を選択
- **利用可能なモデル表示**: 指定したモデルが見つからない場合、利用可能なモデル一覧を自動表示

### Jupyter Notebook版の実行

```sh
jupyter notebook main.ipynb
```

### 4. 設定のカスタマイズ
- 各設定ファイルを編集することで、エポック数やバッチサイズ、学習率などを変更できます。
- GPUの指定は`gpu_id`で行います。

#### 利用可能な設定ファイル

**Commodities別設定:**
- `configs/nsfnet_5_commodities.json`: 5コモディティ設定
- `configs/nsfnet_10_commodities.json`: 10コモディティ設定
- `configs/nsfnet_15_commodities.json`: 15コモディティ設定
- `configs/nsfnet_20_commodities.json`: 20コモディティ設定
- `configs/nsfnet_25_commodities.json`: 25コモディティ設定

**強化学習設定:**
- `configs/nsfnet_*_commodities_rl.json`: 各種コモディティ数対応の強化学習設定

**その他:**
- `configs/default2.json`: 基本設定（モデル保存なし）
- `configs/with_model_saving.json`: モデル保存機能を有効化
- `configs/load_saved_model.json`: 保存済みモデルの最新版を読み込み
- `configs/load_specific_epoch.json`: 特定エポックのモデルを読み込み

### 環境の管理

```sh
# 環境の非アクティベート
conda deactivate

# 環境の削除（必要に応じて）
conda env remove -n gcn-uelb-env

# 環境の更新
conda env update -f environment.yml
```

### 実行例

```sh
# 1. データ生成
python generate_data.py --config configs/nsfnet_15_commodities.json

# 2. GCNトレーニング
python main.py --config configs/nsfnet_15_commodities.json --mode gcn

# 3. 強化学習トレーニング
python main.py --config configs/nsfnet_15_commodities_rl.json --mode rl
```

### 参考
- **データ生成**: `generate_data.py` - データセット生成専用スクリプト
- **メインスクリプト**: `main.py` - 学習・評価パイプライン
- **コード詳細**: `src/`ディレクトリ内の各ファイル
- **プロジェクト構造**: `README_REFACTORED.md`
- **Notebook版**: `main.ipynb`

## ライセンス
本リポジトリのコードは研究目的での利用を想定しています。
