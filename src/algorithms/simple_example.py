import torch
import numpy as np
from typing import List, Tuple
from beamsearch_uelb import BeamSearchFactory, BeamSearchAlgorithm
from beamsearch_comparison_simple import SimpleBeamSearchComparator, SimpleAlgorithmBenchmark

def create_sample_data(batch_size=2, num_nodes=10, num_commodities=5):
    """サンプルデータの作成"""
    
    # 予測エッジテンソル (batch_size, num_nodes, num_nodes, num_commodities, num_vec)
    y_pred_edges = torch.randn(batch_size, num_nodes, num_nodes, num_commodities, 2)
    
    # エッジ容量テンソル (batch_size, num_nodes, num_nodes)
    edges_capacity = torch.randint(0, 10, (batch_size, num_nodes, num_nodes))
    # 対角成分を0に設定（自己ループを防ぐ）
    for i in range(num_nodes):
        edges_capacity[:, i, i] = 0
    
    # コモディティテンソル (batch_size, num_commodities, 3) -> (source, target, demand)
    commodities = torch.zeros(batch_size, num_commodities, 3, dtype=torch.long)
    for b in range(batch_size):
        for c in range(num_commodities):
            source = torch.randint(0, num_nodes, (1,))
            target = torch.randint(0, num_nodes, (1,))
            while target == source:  # 同じノードを避ける
                target = torch.randint(0, num_nodes, (1,))
            demand = torch.randint(1, 5, (1,))
            commodities[b, c] = torch.cat([source, target, demand])
    
    return y_pred_edges, edges_capacity, commodities

def example_single_algorithm():
    """単一アルゴリズムの使用例"""
    print("=== 単一アルゴリズムの使用例 ===")
    
    # サンプルデータ作成
    y_pred_edges, edges_capacity, commodities = create_sample_data()
    
    # 標準的なビームサーチアルゴリズムの使用
    print("\n1. 標準的なビームサーチアルゴリズム")
    standard_algorithm = BeamSearchFactory.create_algorithm(
        'standard',
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
    
    commodity_paths, is_feasible = standard_algorithm.search()
    performance_info = standard_algorithm.get_performance_info()
    
    print(f"実行時間: {performance_info['execution_time']:.4f}秒")
    print(f"実行可能: {is_feasible}")
    print(f"見つかったパス数: {sum(len(paths) for paths in commodity_paths)}")
    
    # 決定論的なビームサーチアルゴリズムの使用
    print("\n2. 決定論的なビームサーチアルゴリズム")
    deterministic_algorithm = BeamSearchFactory.create_algorithm(
        'deterministic',
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
    
    commodity_paths, is_feasible = deterministic_algorithm.search()
    performance_info = deterministic_algorithm.get_performance_info()
    
    print(f"実行時間: {performance_info['execution_time']:.4f}秒")
    print(f"実行可能: {is_feasible}")
    print(f"見つかったパス数: {sum(len(paths) for paths in commodity_paths)}")

def example_algorithm_comparison():
    """アルゴリズム比較の使用例"""
    print("\n=== アルゴリズム比較の使用例 ===")
    
    # サンプルデータ作成
    y_pred_edges, edges_capacity, commodities = create_sample_data()
    
    # 比較器の作成
    comparator = SimpleBeamSearchComparator()
    
    # 全アルゴリズムの比較実行
    results = comparator.compare_algorithms(
        y_pred_edges=y_pred_edges,
        beam_size=3,
        batch_size=2,
        edges_capacity=edges_capacity,
        commodities=commodities,
        dtypeFloat=torch.float32,
        dtypeLong=torch.long,
        mode_strict=False,
        max_iter=5,
        algorithms=['standard', 'deterministic', 'greedy']
    )
    
    # 比較テーブルの出力
    comparator.print_comparison_table()
    
    # 詳細比較結果の出力
    comparator.print_detailed_comparison()
    
    # 最良のアルゴリズムを見つける
    best_by_time = comparator.find_best_algorithm('execution_time')
    best_by_paths = comparator.find_best_algorithm('num_paths_found')
    
    print(f"\n最速アルゴリズム: {best_by_time.get('algorithm', 'N/A')} ({best_by_time.get('execution_time', 0):.4f}秒)")
    print(f"最多パスアルゴリズム: {best_by_paths.get('algorithm', 'N/A')} ({best_by_paths.get('num_paths_found', 0)}パス)")
    
    # 結果のエクスポート
    comparator.export_results_json("algorithm_comparison_results.json")

def example_benchmark():
    """ベンチマークの使用例"""
    print("\n=== ベンチマークの使用例 ===")
    
    # テストケースの定義
    test_cases = [
        {
            'description': '小規模テスト (5ノード, 3コモディティ)',
            'y_pred_edges': torch.randn(1, 5, 5, 3, 2),
            'beam_size': 2,
            'batch_size': 1,
            'edges_capacity': torch.randint(0, 5, (1, 5, 5)),
            'commodities': torch.tensor([[[0, 1, 2], [1, 2, 1], [2, 3, 3]]]),
            'dtypeFloat': torch.float32,
            'dtypeLong': torch.long,
            'mode_strict': False,
            'max_iter': 3
        },
        {
            'description': '中規模テスト (10ノード, 5コモディティ)',
            'y_pred_edges': torch.randn(2, 10, 10, 5, 2),
            'beam_size': 3,
            'batch_size': 2,
            'edges_capacity': torch.randint(0, 8, (2, 10, 10)),
            'commodities': torch.randint(0, 10, (2, 5, 3)),
            'dtypeFloat': torch.float32,
            'dtypeLong': torch.long,
            'mode_strict': False,
            'max_iter': 5
        }
    ]
    
    # ベンチマーク実行
    benchmark = SimpleAlgorithmBenchmark()
    benchmark_results = benchmark.run_benchmark(
        test_cases=test_cases,
        algorithms=['standard', 'deterministic', 'greedy']
    )
    
    # ベンチマークレポートの生成
    benchmark.generate_benchmark_report(benchmark_results, "./benchmark_results")

def example_custom_algorithm():
    """カスタムアルゴリズムの作成例"""
    print("\n=== カスタムアルゴリズムの作成例 ===")
    
    from beamsearch_uelb import BeamSearchAlgorithm
    
    class CustomBeamSearch(BeamSearchAlgorithm):
        """カスタムビームサーチアルゴリズム（需要量優先）"""
        
        def _search_single_batch(self, batch: int) -> Tuple[List[List[int]], List[List[int]], bool]:
            """需要量優先のビームサーチによる単一バッチ検索"""
            batch_y_pred_edges = self.y[batch]
            commodities = self.commodities[batch]
            count = 0

            while count < self.max_iter:
                batch_edges_capacity = self.edges_capacity[batch]
                
                # 需要量に基づく優先度付きソート
                demands = commodities[:, 2]
                sorted_indices = torch.argsort(demands, descending=True)
                sorted_commodities = commodities[sorted_indices]
                sorted_pred_edges = batch_y_pred_edges[:, :, sorted_indices]
                _, original_indices = torch.sort(sorted_indices)

                node_orders = []
                commodity_paths = []
                is_feasible = True

                for index, commodity in enumerate(sorted_commodities):
                    source_node = commodity[0].item()
                    target_node = commodity[1].item()
                    demand = commodity[2].item()
                    
                    # ビームサーチによるパス探索
                    node_order, remaining_edges_capacity, best_path = self._beam_search_for_commodity(
                        batch_edges_capacity, sorted_pred_edges[:, :, index], 
                        source_node, target_node, demand
                    )
                    
                    if best_path == []:
                        break
                        
                    node_orders.append(node_order)
                    if self.mode_strict:
                        batch_edges_capacity = remaining_edges_capacity
                    commodity_paths.append(best_path)

                if len(commodity_paths) == commodities.shape[0]:
                    count = self.max_iter
                    unshuffled_commodity_paths = [commodity_paths[i] for i in original_indices]
                else:
                    is_feasible = False
                    unshuffled_commodity_paths = self._get_fallback_paths(commodities.shape[0])
                    count += 1
            
            return node_orders, unshuffled_commodity_paths, is_feasible

        def _beam_search_for_commodity(self, edges_capacity, y_commodities, source, target, demand):
            """標準的なビームサーチ（元の実装と同じ）"""
            beam_queue = [(source, [source], 0, edges_capacity.clone())]
            best_paths = []

            while beam_queue:
                current_scores = [item[2] for item in beam_queue]
                beam_queue = sorted(beam_queue, key=lambda x: x[2], reverse=True)[:self.beam_size]

                next_beam_queue = []
                for current_node, path, current_score, remaining_edges_capacity in beam_queue:
                    if current_node == target:
                        best_paths.append((path, current_score, remaining_edges_capacity))
                        continue

                    for next_node in range(edges_capacity.shape[0]):
                        if next_node in path:
                            continue
                        if (edges_capacity[current_node, next_node].item() == 0):
                            continue

                        flow_probability = y_commodities[current_node, next_node]
                        new_score = current_score + flow_probability

                        if demand <= remaining_edges_capacity[current_node, next_node]:
                            updated_capacity = remaining_edges_capacity.clone()
                            updated_capacity[current_node, next_node] -= demand
                            new_path = path + [next_node]
                            next_beam_queue.append((next_node, new_path, new_score, updated_capacity))
                            
                beam_queue = next_beam_queue

            if not best_paths:
                node_order = [0] * edges_capacity.shape[0]
                return node_order, edges_capacity, []

            best_paths = sorted(best_paths, key=lambda x: x[1], reverse=True)
            node_order = [0] * edges_capacity.shape[0]
            for idx, node in enumerate(best_paths[0][0]):
                node_order[node] = idx + 1
            return node_order, best_paths[0][2], best_paths[0][0]

        def _get_fallback_paths(self, num_commodities: int) -> List[List[int]]:
            """フォールバックパスの生成"""
            return [[0,1,2,3,4,5,6,7,8,9] for _ in range(num_commodities)]
    
    # カスタムアルゴリズムの使用
    y_pred_edges, edges_capacity, commodities = create_sample_data()
    
    custom_algorithm = CustomBeamSearch(
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
    
    commodity_paths, is_feasible = custom_algorithm.search()
    performance_info = custom_algorithm.get_performance_info()
    
    print(f"カスタムアルゴリズム実行時間: {performance_info['execution_time']:.4f}秒")
    print(f"実行可能: {is_feasible}")
    print(f"見つかったパス数: {sum(len(paths) for paths in commodity_paths)}")

def main():
    """メイン関数"""
    print("ビームサーチアルゴリズム抽象化の使用例（簡略版）")
    print("=" * 60)
    
    # 利用可能なアルゴリズムの表示
    available_algorithms = BeamSearchFactory.get_available_algorithms()
    print(f"利用可能なアルゴリズム: {', '.join(available_algorithms)}")
    
    # 各使用例の実行
    example_single_algorithm()
    example_algorithm_comparison()
    example_benchmark()
    example_custom_algorithm()
    
    print("\n=== 使用例完了 ===")

if __name__ == "__main__":
    main() 