# Graph Convolutional Network for UELB (Refactored)

## プロジェクト構造

```
graph-convnet-uelb/
├── main.py                    # メインファイル
├── main_old.py                # 古いバージョンのメインファイル
├── main.ipynb                 # Jupyter notebook版
├── config.py                  # 設定ファイル
├── configs/                   # 設定ファイルディレクトリ
│   ├── actual.json
│   ├── cpu.json
│   ├── default.json
│   └── default2.json
├── data/                      # データセット専用
├── edge_flow_matrix.csv       # エッジフローマトリックス
├── flow_edge_matrix.csv       # フローエッジマトリックス
├── environment.yml            # 環境設定
├── requirements.txt           # Python依存関係
├── setup_environment.sh       # 環境セットアップスクリプト
├── src/                       # すべてのソースコード
│   ├── algorithms/            # アルゴリズム実装
│   │   ├── __init__.py
│   │   ├── beamsearch_uelb.py
│   │   └── beamsearch.py
│   ├── config/                # 設定関連
│   │   ├── __init__.py
│   │   ├── config_manager.py  # 設定管理
│   │   └── config.py          # 設定読み込み
│   ├── data_management/       # データ管理モジュール
│   │   ├── __init__.py
│   │   ├── create_data_files.py
│   │   ├── data_maker.py
│   │   ├── dataset_manager.py # データセット管理
│   │   ├── dataset_reader.py
│   │   ├── exact_flow.py
│   │   └── exact_solution.py
│   ├── graph/                 # グラフ関連モジュール
│   │   ├── __init__.py
│   │   ├── flow.py
│   │   ├── graph_making.py
│   │   └── graph_utils.py
│   ├── models/                # モデル定義
│   │   ├── __init__.py
│   │   ├── gcn_layers.py
│   │   ├── gcn_model.py
│   │   └── model_utils.py
│   ├── train/                 # トレーニング関連モジュール
│   │   ├── __init__.py
│   │   ├── trainer.py         # トレーニングロジック
│   │   ├── evaluator.py       # 評価ロジック
│   │   └── metrics.py         # メトリクス計算・出力
│   ├── visualization/         # 可視化モジュール
│   │   ├── __init__.py
│   │   └── plot_utils.py
│   └── tests/                 # テストモジュール
│       ├── __init__.py
│       └── test.py
├── models/                    # ルートレベルのモデル（重複）
│   ├── __init__.py
│   ├── gcn_layers.py
│   └── gcn_model.py
├── logs/                      # ログファイル
├── test.ipynb                 # テスト用notebook
└── utils/                     # ユーティリティ（重複）
    └── __pycache__ 2/
```

## 使用方法

### メインファイル

```bash
python main.py
```

### Jupyter Notebook版

```bash
jupyter notebook main.ipynb
```

### 環境セットアップ

```bash
# 環境のセットアップ
bash setup_environment.sh

# または conda を使用
conda env create -f environment.yml
```

## モジュール説明

### src/train/
- **trainer.py**: トレーニングロジックを担当
  - モデルとオプティマイザーの初期化
  - 1エポックのトレーニング実行
  - 学習率の更新

- **evaluator.py**: 評価ロジックを担当
  - 検証・テストデータでの評価
  - 負荷率と近似率の計算

- **metrics.py**: メトリクス管理を担当
  - トレーニング結果の記録
  - 表形式での結果出力
  - 最終サマリーの生成

### src/data_management/
- **dataset_manager.py**: データセット管理を担当
  - データセットの作成・削除
  - データローディングのテスト
- **data_maker.py**: データ生成ロジック
- **dataset_reader.py**: データセット読み込み
- **exact_flow.py**: 正確なフロー計算
- **exact_solution.py**: 正確な解の計算

### src/graph/
- **graph_making.py**: グラフ生成ロジック
- **graph_utils.py**: グラフユーティリティ
- **flow.py**: フロー計算

### src/models/
- **gcn_model.py**: GCNモデルの定義
- **gcn_layers.py**: GCNレイヤーの実装
- **model_utils.py**: モデルユーティリティ

### src/algorithms/
- **beamsearch_uelb.py**: UELB用ビームサーチ
- **beamsearch.py**: 汎用ビームサーチ

### src/config/
- **config_manager.py**: 設定管理を担当
  - 設定ファイルの読み込み
  - GPU設定の管理
- **config.py**: 設定読み込み

### src/visualization/
- **plot_utils.py**: 可視化ユーティリティ

## 設定ファイル

プロジェクトには複数の設定ファイルが用意されています：

- `configs/default.json`: デフォルト設定
- `configs/default2.json`: 代替設定
- `configs/cpu.json`: CPU用設定
- `configs/actual.json`: 実際の実行用設定

## 利点

1. **モジュール性**: 各機能が独立したクラスに分離
2. **可読性**: コードが整理され、理解しやすい
3. **保守性**: 各モジュールを個別に修正・拡張可能
4. **再利用性**: 各クラスを他のプロジェクトで再利用可能
5. **テスト性**: 各モジュールを個別にテスト可能
6. **柔軟性**: 複数の設定ファイルで異なる実行環境に対応

## データファイル

- `edge_flow_matrix.csv`: エッジフローマトリックス
- `flow_edge_matrix.csv`: フローエッジマトリックス

## 今後の拡張

この構造により、以下のような拡張が容易になります：

- 新しい評価指標の追加
- 異なるトレーニング戦略の実装
- 可視化機能の強化
- ハイパーパラメータチューニングの統合
- 新しいアルゴリズムの追加
- 異なるグラフ構造のサポート 