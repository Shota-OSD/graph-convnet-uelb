# データ生成ガイド

## 概要

GCNモデル用のトレーニング・検証・テストデータを生成するためのガイドです。

## 基本的な使用方法

### 全データ生成
```bash
# デフォルト設定で全データ（train, val, test）を生成
python generate_data.py

# 設定ファイルを指定
python generate_data.py --config configs/default.json
```

### 特定データのみ生成
```bash
# テストデータのみ
python generate_data.py --modes test

# トレーニングと検証データのみ
python generate_data.py --modes train val
```

## コマンドオプション

| オプション | 説明 | デフォルト |
|------------|------|------------|
| `--config` | 設定ファイルのパス | `configs/default2.json` |
| `--modes` | 生成するデータモード | `train val test` |
| `--force` | 確認なしで既存データを削除して再生成 | なし |
| `--clean-only` | データを削除するのみ（再生成しない） | なし |

## 使用例

### 初回データ生成
```bash
python generate_data.py --config configs/default.json
```

### データ再生成
```bash
# 既存データを削除して再生成
python generate_data.py --force

# テストデータのみ再生成
python generate_data.py --modes test --force
```

### データ削除
```bash
# 既存データを削除のみ
python generate_data.py --clean-only
```

## 生成されるデータ構造

```
data/
├── train_data/
│   ├── commodity_file/
│   ├── graph_file/
│   └── node_flow_file/
├── val_data/
│   ├── commodity_file/
│   ├── graph_file/
│   └── node_flow_file/
└── test_data/
    ├── commodity_file/
    ├── graph_file/
    └── node_flow_file/
```

## データ数の設定

設定ファイル（例: `configs/default.json`）で制御:
```json
{
  "num_train_data": 3200,
  "num_val_data": 320,
  "num_test_data": 320,
  "num_commodities": 10,
  "num_nodes": 14
}
```

## 注意事項

- データ生成には時間がかかる場合があります
- 既存データがある場合は確認メッセージが表示されます
- `--force`オプションで確認をスキップできます
- 設定ファイルでネットワーク構造とデータサイズを調整できます

## トラブルシューティング

### メモリ不足
- `num_train_data`を小さくする
- `num_commodities`を減らす

### 時間がかかりすぎる
- データ数を減らす
- `--modes`で必要なデータのみ生成