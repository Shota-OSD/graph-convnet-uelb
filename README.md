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

### Jupyter Notebook版の実行

```sh
jupyter notebook main.ipynb
```

### 設定のカスタマイズ
- `configs/default2.json`を編集することで、エポック数やバッチサイズ、学習率などを変更できます。
- GPUの指定は`gpu_id`で行います。

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
