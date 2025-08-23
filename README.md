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
### 学習・評価パイプラインの実行

以下のコマンドで、グラフデータの生成、モデルの学習、検証、テストまで一括で実行できます。

```sh
python main.py
```

- 実行すると、`configs/default2.json`の設定に従い、
    - データセットの前処理
    - モデルの構築
    - 学習ループ
    - 検証・テスト
  が自動で行われます。
- ログやモデルの重みは`logs/`ディレクトリに保存されます。

### モデルの保存・再利用機能

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

### 設定のカスタマイズ
- `configs/default2.json`を編集することで、エポック数やバッチサイズ、学習率などを変更できます。
- GPUの指定は`gpu_id`で行います。

#### 利用可能な設定ファイル

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

### 参考
- コードの詳細な流れや関数の説明は`main.py`および`main.ipynb`を参照してください。
- データセットやモデルの詳細は`src/`ディレクトリ内の各ファイルを参照してください。
- プロジェクト構造の詳細は`README_REFACTORED.md`を参照してください。

## ライセンス
本リポジトリのコードは研究目的での利用を想定しています。
