# Type Safety Guide

## Overview

このプロジェクトでは、型エラーを防ぐために以下の3層の防御策を実装しています：

1. **Type Hints（型ヒント）**: 静的解析ツールで検出
2. **TypedDict**: 辞書の構造を明示的に定義
3. **Runtime Validation**: 実行時に型チェック

## 主な型エラーの原因

### 1. Embedding層への誤った型

**問題**: `nn.Embedding`は整数インデックスを必要とするが、floatテンソルを渡してしまう

```python
# ❌ 間違い
x_nodes = torch.from_numpy(batch.nodes).float()
x_embedded = self.nodes_embedding(x_nodes)  # RuntimeError!

# ✅ 正しい
x_nodes = torch.from_numpy(batch.nodes).long()
x_embedded = self.nodes_embedding(x_nodes)
```

### 2. Linear層との混同

```python
# nn.Embedding: 整数が必要
self.nodes_embedding = nn.Embedding(vocab_size, hidden_dim)
x = self.nodes_embedding(x_nodes)  # x_nodes must be torch.long

# nn.Linear: floatが必要
self.commodities_embedding = nn.Linear(1, hidden_dim)
x = self.commodities_embedding(x_commodities)  # x_commodities must be torch.float
```

## 型定義の使用方法

### 1. BatchData型の使用

```python
from src.common.types import BatchData, validate_batch_types

def _prepare_batch(self, batch) -> BatchData:
    """型ヒント付きでバッチを準備"""
    batch_data: BatchData = {
        'x_nodes': torch.from_numpy(batch.nodes).long(),  # 必ずlong
        'x_commodities': torch.from_numpy(batch.commodities).float(),
        'x_edges_capacity': torch.from_numpy(batch.edges_capacity).float(),
        # ... その他
    }

    # 実行時検証（開発時のみ推奨）
    validate_batch_types(batch_data, strict=True)

    return batch_data
```

### 2. Encoder入力の型チェック

```python
from src.common.types import validate_encoder_input_types

def forward(
    self,
    x_nodes: Tensor,  # torch.long
    x_commodities: Tensor,  # torch.float
    x_edges_capacity: Tensor,  # torch.float
) -> Tuple[Tensor, Tensor, Tensor]:
    """型ヒント付きforward"""

    # Critical: embedding層の前に型チェック
    if x_nodes.dtype != torch.long:
        raise TypeError(
            f"x_nodes must be torch.long for embedding layer, "
            f"got {x_nodes.dtype}. Use .long() to convert."
        )

    # embedding層を使用
    x_embedded = self.nodes_embedding(x_nodes)  # 安全
    ...
```

## 静的解析ツールの使用

### mypy（推奨）

```bash
# インストール
pip install mypy

# 型チェック実行
mypy src/seq_flow_rl/training/trainer.py
mypy src/seq_flow_rl/models/hybrid_gnn_encoder.py

# プロジェクト全体
mypy src/
```

### pylance（VSCode）

VSCodeの`settings.json`に追加：

```json
{
  "python.analysis.typeCheckingMode": "basic",  // or "strict"
  "python.analysis.diagnosticMode": "workspace"
}
```

## 型チェックのベストプラクティス

### 1. 開発時は厳密、本番環境では軽量

```python
# configで制御
def _prepare_batch(self, batch) -> BatchData:
    batch_data = {...}

    # 開発時のみ実行時検証
    if self.config.get('validate_batch_types', __debug__):
        validate_batch_types(batch_data, strict=True)

    return batch_data
```

### 2. Critical な箇所には明示的なチェック

```python
def forward(self, x_nodes: Tensor, ...) -> ...:
    # Embedding層の前には必ずチェック
    assert x_nodes.dtype == torch.long, \
        f"x_nodes must be long, got {x_nodes.dtype}"

    x_embedded = self.nodes_embedding(x_nodes)
```

### 3. 型ヒントを活用したドキュメント

```python
def process_batch(
    self,
    x_nodes: Tensor,  # [B, V, C] - torch.long
    x_commodities: Tensor,  # [B, C, 3] - torch.float
) -> Tensor:  # [B, H] - torch.float
    """
    型ヒントとコメントで期待される形状とdtypeを明示
    """
    ...
```

## トラブルシューティング

### エラー: "Expected tensor for argument #1 'indices' to have ... Long, Int; but got torch.FloatTensor"

**原因**: Embedding層にfloatテンソルを渡している

**解決策**:
```python
# 変換箇所を確認
x_nodes = torch.from_numpy(batch.nodes).long()  # .float() → .long()
```

### エラー: "RuntimeError: mat1 and mat2 shapes cannot be multiplied"

**原因**: Linear層に誤った形状のテンソルを渡している（型ではなく形状の問題）

**解決策**: 型ヒントのコメントで期待される形状を確認
```python
x: Tensor  # [B, H] - 期待される形状をチェック
```

## まとめ

型安全性を高めるために：

1. ✅ **必ず型ヒントを使う**: `def func(...) -> ReturnType:`
2. ✅ **TypedDictで構造を定義**: `BatchData`, `EncoderInput`
3. ✅ **Critical な箇所で検証**: Embedding層の前に型チェック
4. ✅ **mypyで静的解析**: 定期的に実行
5. ✅ **コメントで形状とdtypeを明示**: `# [B, V, C] - torch.long`

これにより、型エラーを**開発時**に発見し、**実行時**のクラッシュを防ぐことができます。
