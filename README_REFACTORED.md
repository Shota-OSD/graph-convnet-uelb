# Graph Convolutional Network for UELB (Refactored)

## プロジェクト構造

```
graph-convnet-uelb/
├── main.py                    # 元のメインファイル（非推奨）
├── main_refactored.py         # リファクタリングされたメインファイル（推奨）
├── configs/                   # 設定ファイル
│   ├── default.json
│   └── default2.json
├── data/                      # データセット専用
│   ├── train_data/
│   ├── val_data/
│   └── test_data/
├── src/                       # すべてのソースコード
│   ├── train/                 # トレーニング関連モジュール
│   │   ├── __init__.py
│   │   ├── trainer.py         # トレーニングロジック
│   │   ├── evaluator.py       # 評価ロジック
│   │   └── metrics.py         # メトリクス計算・出力
│   ├── data_management/       # データ管理モジュール
│   │   ├── __init__.py
│   │   └── dataset_manager.py # データセット管理
│   ├── utils/                 # ユーティリティモジュール
│   │   ├── __init__.py
│   │   ├── config_manager.py  # 設定管理
│   │   ├── beamsearch_uelb.py
│   │   ├── create_data_files.py
│   │   ├── dataset_reader.py
│   │   ├── exact_solution.py
│   │   ├── flow.py
│   │   ├── graph_utils.py
│   │   ├── model_utils.py
│   │   └── plot_utils.py
│   ├── models/                # モデル定義
│   │   ├── __init__.py
│   │   ├── gcn_layers.py
│   │   └── gcn_model.py
│   └── config/                # 設定関連
│       ├── __init__.py
│       └── config.py          # 設定読み込み
└── logs/                      # ログファイル
```

## 使用方法

### リファクタリングされたバージョン（推奨）

```bash
python main_refactored.py
```

### 元のバージョン

```bash
python main.py
```

## モジュール説明

### train/
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

### data/
- **dataset_manager.py**: データセット管理を担当
  - データセットの作成・削除
  - データローディングのテスト

### utils/
- **config_manager.py**: 設定管理を担当
  - 設定ファイルの読み込み
  - GPU設定の管理

## 利点

1. **モジュール性**: 各機能が独立したクラスに分離
2. **可読性**: コードが整理され、理解しやすい
3. **保守性**: 各モジュールを個別に修正・拡張可能
4. **再利用性**: 各クラスを他のプロジェクトで再利用可能
5. **テスト性**: 各モジュールを個別にテスト可能

## 移行ガイド

元の`main.py`から新しい`main_refactored.py`への移行：

1. 機能は同じですが、コードが整理されています
2. 出力形式は同じです
3. 設定ファイルは変更不要です
4. 既存のデータセットはそのまま使用可能です

## 今後の拡張

この構造により、以下のような拡張が容易になります：

- 新しい評価指標の追加
- 異なるトレーニング戦略の実装
- 可視化機能の強化
- ハイパーパラメータチューニングの統合 