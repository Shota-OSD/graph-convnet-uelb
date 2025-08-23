# ビームサーチアルゴリズム抽象化システム

このシステムは、異なるビームサーチアルゴリズムを簡単に切り替えて比較できるように設計された抽象化フレームワークです。

## 概要

元の `beamsearch_uelb.py` では単一のビームサーチアルゴリズムしか実装されていませんでしたが、このシステムでは以下の機能を提供します：

- **抽象基底クラス**: 共通のインターフェースを定義
- **複数のアルゴリズム実装**: 異なる戦略のビームサーチ
- **ファクトリーパターン**: 簡単なアルゴリズム選択
- **性能比較機能**: アルゴリズム間の比較とベンチマーク
- **拡張性**: 新しいアルゴリズムの簡単な追加

## ファイル構成

```
src/algorithms/
├── beamsearch_uelb.py              # 抽象化されたビームサーチシステム
├── beamsearch_comparison_simple.py # アルゴリズム比較・ベンチマーク機能（簡略版）
├── simple_example.py              # 使用例（簡略版）
├── beamsearch.py                  # 元のビームサーチ実装（参考用）
├── __init__.py                    # Pythonパッケージ用
└── README_BEAMSEARCH.md           # このファイル
```

## バージョン選択

### 完全版（beamsearch_comparison.py）
- **依存関係**: PyTorch, NumPy, Pandas, Matplotlib, Seaborn
- **機能**: 詳細な比較、グラフ表示、CSVエクスポート
- **用途**: 詳細な分析やレポート作成

### 簡略版（beamsearch_comparison_simple.py）
- **依存関係**: PyTorch, NumPy（標準ライブラリのみ）
- **機能**: 基本的な比較、テーブル表示、JSONエクスポート
- **用途**: 軽量な比較や基本的な性能評価

## 利用可能なアルゴリズム

### 1. StandardBeamSearch (標準)
- **特徴**: 元の実装と同じ（ランダムシャッフル）
- **用途**: ベースライン比較用
- **後方互換性**: `BeamsearchUELB` として利用可能

### 2. DeterministicBeamSearch (決定論的)
- **特徴**: シャッフルなしの決定論的実行
- **用途**: 再現性が必要な場合
- **利点**: 同じ入力に対して常に同じ結果

### 3. GreedyBeamSearch (貪欲的)
- **特徴**: ビームサイズ1の貪欲探索
- **用途**: 高速実行が必要な場合
- **利点**: 計算時間が短い

## 基本的な使用方法

### 1. 単一アルゴリズムの使用

```python
from beamsearch_uelb import BeamSearchFactory

# データの準備
y_pred_edges = torch.randn(2, 10, 10, 5, 2)
edges_capacity = torch.randint(0, 10, (2, 10, 10))
commodities = torch.randint(0, 10, (2, 5, 3))

# アルゴリズムの作成
algorithm = BeamSearchFactory.create_algorithm(
    'standard',  # または 'deterministic', 'greedy'
    y_pred_edges=y_pred_edges,
    beam_size=3,
    batch_size=2,
    edges_capacity=edges_capacity,
    commodities=commodities,
    dtypeFloat=torch.float32,
    dtypeLong=torch.long,
    mode_strict=False,
    max_iter=5
)

# 実行
commodity_paths, is_feasible = algorithm.search()

# 性能情報の取得
performance_info = algorithm.get_performance_info()
print(f"実行時間: {performance_info['execution_time']:.4f}秒")
```

### 2. アルゴリズム比較（簡略版）

```python
from beamsearch_comparison_simple import SimpleBeamSearchComparator

# 比較器の作成
comparator = SimpleBeamSearchComparator()

# 複数アルゴリズムの比較実行
results = comparator.compare_algorithms(
    y_pred_edges=y_pred_edges,
    beam_size=3,
    batch_size=2,
    edges_capacity=edges_capacity,
    commodities=commodities,
    algorithms=['standard', 'deterministic', 'greedy']
)

# 比較テーブルの出力
comparator.print_comparison_table()

# 詳細比較結果の出力
comparator.print_detailed_comparison()

# 最良のアルゴリズムを見つける
best_by_time = comparator.find_best_algorithm('execution_time')
print(f"最速アルゴリズム: {best_by_time['algorithm']}")

# 結果のエクスポート（JSON形式）
comparator.export_results_json("comparison_results.json")
```

### 3. アルゴリズム比較（完全版）

```python
from beamsearch_comparison import BeamSearchComparator

# 比較器の作成
comparator = BeamSearchComparator()

# 複数アルゴリズムの比較実行
results = comparator.compare_algorithms(
    y_pred_edges=y_pred_edges,
    beam_size=3,
    batch_size=2,
    edges_capacity=edges_capacity,
    commodities=commodities,
    algorithms=['standard', 'deterministic', 'greedy']
)

# 詳細比較結果の出力
comparator.print_detailed_comparison()

# 性能サマリーの取得
summary_df = comparator.get_performance_summary()
print(summary_df)

# 比較プロットの表示
comparator.plot_comparison()

# 結果のエクスポート（CSV形式）
comparator.export_results("comparison_results.csv")
```

### 4. ベンチマーク実行（簡略版）

```python
from beamsearch_comparison_simple import SimpleAlgorithmBenchmark

# テストケースの定義
test_cases = [
    {
        'description': '小規模テスト',
        'y_pred_edges': torch.randn(1, 5, 5, 3, 2),
        'beam_size': 2,
        'batch_size': 1,
        'edges_capacity': torch.randint(0, 5, (1, 5, 5)),
        'commodities': torch.tensor([[[0, 1, 2], [1, 2, 1], [2, 3, 3]]]),
        'dtypeFloat': torch.float32,
        'dtypeLong': torch.long,
        'mode_strict': False,
        'max_iter': 3
    }
]

# ベンチマーク実行
benchmark = SimpleAlgorithmBenchmark()
benchmark_results = benchmark.run_benchmark(
    test_cases=test_cases,
    algorithms=['standard', 'deterministic', 'greedy']
)

# レポート生成（JSON形式）
benchmark.generate_benchmark_report(benchmark_results, "./benchmark_results")
```

## カスタムアルゴリズムの作成

新しいアルゴリズムを追加するには、`BeamSearchAlgorithm` を継承して `_search_single_batch` メソッドを実装します：

```python
from beamsearch_uelb import BeamSearchAlgorithm

class CustomBeamSearch(BeamSearchAlgorithm):
    """カスタムビームサーチアルゴリズム"""
    
    def _search_single_batch(self, batch: int) -> Tuple[List[List[int]], List[List[int]], bool]:
        """
        カスタムアルゴリズムによる単一バッチ検索
        
        Args:
            batch: バッチインデックス
            
        Returns:
            Tuple[List[List[int]], List[List[int]], bool]: (node_orders, commodity_paths, is_feasible)
        """
        # カスタムアルゴリズムの実装
        # このメソッド内で、前処理、パス探索、後処理など全てを自由に実装できます
        
        batch_y_pred_edges = self.y[batch]
        commodities = self.commodities[batch]
        
        # 例：需要量に基づく優先度付きソート
        demands = commodities[:, 2]
        sorted_indices = torch.argsort(demands, descending=True)
        sorted_commodities = commodities[sorted_indices]
        sorted_pred_edges = batch_y_pred_edges[:, :, sorted_indices]
        _, original_indices = torch.sort(sorted_indices)
        
        # パス探索の実装
        node_orders = []
        commodity_paths = []
        is_feasible = True
        
        # ... カスタムロジック ...
        
        return node_orders, commodity_paths, is_feasible

# 使用
custom_algorithm = CustomBeamSearch(
    y_pred_edges=y_pred_edges,
    beam_size=3,
    batch_size=2,
    edges_capacity=edges_capacity,
    commodities=commodities,
    dtypeFloat=torch.float32,
    dtypeLong=torch.long
)

commodity_paths, is_feasible = custom_algorithm.search()
```

### 抽象化の利点

この設計により、以下の利点があります：

1. **柔軟性**: 各アルゴリズムが独自の実装方法を選択可能
2. **拡張性**: 新しいアルゴリズムの追加が容易
3. **独立性**: アルゴリズム間の依存関係が最小限
4. **保守性**: 共通部分とアルゴリズム固有部分の分離が明確

## 性能比較の指標

システムは以下の指標でアルゴリズムを比較します：

1. **実行時間**: アルゴリズムの実行にかかった時間
2. **実行可能性**: 全てのコモディティに対してパスが見つかったかどうか
3. **見つかったパス数**: 成功したパス探索の数
4. **総パス長**: 全てのパスの長さの合計
5. **ビームサイズ**: 使用されたビームサイズ
6. **最大反復回数**: 実際に使用された反復回数

## 後方互換性

既存のコードとの互換性を保つため、元の `BeamsearchUELB` クラスは `StandardBeamSearch` のエイリアスとして提供されています：

```python
# 既存のコード（変更不要）
from beamsearch_uelb import BeamsearchUELB

algorithm = BeamsearchUELB(
    y_pred_edges=y_pred_edges,
    beam_size=3,
    batch_size=2,
    edges_capacity=edges_capacity,
    commodities=commodities,
    dtypeFloat=torch.float32,
    dtypeLong=torch.long
)
```

## 実行例

### 簡略版の実行（推奨）

```bash
python src/algorithms/simple_example.py
```

これにより、以下の内容が実行されます：
- 単一アルゴリズムの使用例
- アルゴリズム比較の例
- ベンチマーク実行の例
- カスタムアルゴリズム作成の例

## 依存関係

### 簡略版
- PyTorch
- NumPy

### 完全版
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Seaborn

## 実行結果の例

簡略版の実行結果例：

```
================================================================================
ビームサーチアルゴリズム比較結果
================================================================================
アルゴリズム          実行時間(秒)      実行可能     パス数      総パス長      
--------------------------------------------------------------------------------
standard        0.0363       True     10       44        
deterministic   0.0328       True     10       44        
greedy          0.0175       True     10       85        
--------------------------------------------------------------------------------

最速アルゴリズム: greedy (0.0175秒)
最多パスアルゴリズム: standard (10パス)
```

## 注意事項

1. **メモリ使用量**: 大きなグラフやビームサイズではメモリ使用量が増加します
2. **実行時間**: アルゴリズムによって実行時間が大きく異なる場合があります
3. **結果の再現性**: ランダムシャッフルを使用するアルゴリズムは結果が変動する可能性があります
4. **GPU対応**: 現在はCPUでの実行を想定していますが、GPU対応も可能です

## 今後の拡張予定

- より多くのビームサーチアルゴリズムの追加
- GPU対応の最適化
- 並列処理による高速化
- より詳細な性能分析機能
- 設定ファイルによるアルゴリズム選択 