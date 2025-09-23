# コンフィグファイル設定項目説明

このディレクトリには、グラフ畳み込みネットワーク（Graph Convolutional Network）の実験設定ファイルが含まれています。

## ファイル形式

- `.json`: 標準的なJSON形式のコンフィグファイル
- `.jsonc`: コメント付きJSON形式のコンフィグファイル（可読性向上）

## 設定項目カテゴリ

### 1. 実験設定 (Experiment Settings)
- `expt_name`: 実験名（ログやモデル保存時の識別子）

### 2. ハードウェア設定 (Hardware Settings)
- `gpu_id`: 使用するGPUのID
- `use_gpu`: GPU使用の有無（true/false）

### 3. データファイルパス (Data File Paths)
- `graph_filepath`: グラフファイルのパス
- `edge_numbering_filepath`: エッジ番号付けファイルのパス
- `train_filepath`: 訓練データファイルのパス
- `val_filepath`: 検証データファイルのパス
- `test_filepath`: テストデータファイルのパス

### 4. グラフ生成設定 (Graph Generation Settings)
- `solver_type`: ソルバーの種類（例: "pulp"）
- `graph_model`: グラフモデルの種類（例: "random", "nsfnet"）

### 5. データセット設定 (Dataset Settings)
- `num_train_data`: 訓練データ数
- `num_test_data`: テストデータ数
- `num_val_data`: 検証データ数
- `num_nodes`: ノード数
- `num_neighbors`: 隣接ノード数（一部の設定で使用）
- `num_commodities`: 商品数
- `sample_size`: サンプルサイズ
- `capacity_lower`: 容量の下限
- `capacity_higher`: 容量の上限
- `demand_lower`: 需要の下限
- `demand_higher`: 需要の上限

### 6. モデルアーキテクチャ設定 (Model Architecture Settings)
- `node_dim`: ノード次元
- `voc_nodes_in`: 入力ノード語彙サイズ
- `voc_nodes_out`: 出力ノード語彙サイズ
- `voc_edges_in`: 入力エッジ語彙サイズ
- `voc_edges_out`: 出力エッジ語彙サイズ
- `hidden_dim`: 隠れ層の次元
- `num_layers`: レイヤー数
- `mlp_layers`: MLPレイヤー数
- `aggregation`: 集約方法（例: "mean"）

### 7. ビームサーチ設定 (Beam Search Settings)
- `beam_size`: ビームサイズ

### 8. 学習設定 (Training Settings)
- `max_epochs`: 最大エポック数
- `val_every`: 検証実行間隔（エポック）
- `test_every`: テスト実行間隔（エポック）
- `batch_size`: バッチサイズ
- `accumulation_steps`: 勾配蓄積ステップ数
- `learning_rate`: 学習率
- `decay_rate`: 減衰率
- `dropout_rate`: ドロップアウト率

### 9. モデル保存設定 (Model Saving Settings)
- `models_dir`: モデル保存ディレクトリ
- `save_model`: モデル保存の有無
- `save_every_epoch`: エポックごとの保存の有無
- `load_saved_model`: 保存済みモデルの読み込みの有無
- `load_model_epoch`: 読み込むモデルのエポック（一部の設定で使用）
- `cleanup_old_models`: 古いモデルの削除の有無

## 利用可能なコンフィグファイル

1. **cpu-train-3200.jsonc**: CPU環境での大規模訓練用（3200サンプル）
2. **default.jsonc**: デフォルト設定（NSFnetグラフモデル）
3. **with_model_saving.jsonc**: モデル保存機能付き設定
4. **load_saved_model.jsonc**: 保存済みモデル読み込み用
5. **load_specific_epoch.jsonc**: 特定エポックのモデル読み込み用
6. **default2.jsonc**: デフォルト設定の変形版
7. **cpu.jsonc**: CPU環境用の基本設定

## 使用方法

```python
# JSONCファイルを読み込む場合
import json
import re

def load_jsonc(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    # コメントを除去
    content = re.sub(r'//.*$', '', content, flags=re.MULTILINE)
    return json.loads(content)

config = load_jsonc('configs/cpu-train-3200.jsonc')
```

## 注意事項

- JSONCファイルはコメント付きのJSON形式です
- 標準のJSONパーサーでは直接読み込めません
- コメントを除去してからJSONとして解析してください
