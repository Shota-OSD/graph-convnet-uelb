import torch
import time
import json
from typing import Dict, List, Tuple, Any
from beamsearch_uelb import BeamSearchFactory, BeamSearchAlgorithm

class SimpleBeamSearchComparator:
    """簡略版ビームサーチアルゴリズム比較クラス（標準ライブラリのみ使用）"""
    
    def __init__(self):
        self.results = {}
        self.comparison_data = []
    
    def compare_algorithms(self, 
                          y_pred_edges: torch.Tensor,
                          beam_size: int,
                          batch_size: int,
                          edges_capacity: torch.Tensor,
                          commodities: torch.Tensor,
                          dtypeFloat=torch.float32,
                          dtypeLong=torch.long,
                          mode_strict=False,
                          max_iter=5,
                          algorithms: List[str] = None) -> Dict[str, Any]:
        """
        複数のアルゴリズムを比較実行
        
        Args:
            y_pred_edges: 予測エッジテンソル
            beam_size: ビームサイズ
            batch_size: バッチサイズ
            edges_capacity: エッジ容量テンソル
            commodities: コモディティテンソル
            dtypeFloat: 浮動小数点データ型
            dtypeLong: 長整数データ型
            mode_strict: 厳密モード
            max_iter: 最大反復回数
            algorithms: 比較するアルゴリズムのリスト（Noneの場合は全て）
            
        Returns:
            Dict[str, Any]: 比較結果
        """
        if algorithms is None:
            algorithms = BeamSearchFactory.get_available_algorithms()
        
        self.results = {}
        self.comparison_data = []
        
        print(f"\n{'='*60}")
        print("ビームサーチアルゴリズム比較開始")
        print(f"{'='*60}")
        
        for algorithm_name in algorithms:
            print(f"\n実行中: {algorithm_name}")
            
            try:
                # アルゴリズムのインスタンス作成
                algorithm = BeamSearchFactory.create_algorithm(
                    algorithm_name,
                    y_pred_edges=y_pred_edges,
                    beam_size=beam_size,
                    batch_size=batch_size,
                    edges_capacity=edges_capacity,
                    commodities=commodities,
                    dtypeFloat=dtypeFloat,
                    dtypeLong=dtypeLong,
                    mode_strict=mode_strict,
                    max_iter=max_iter
                )
                
                # 実行
                start_time = time.time()
                commodity_paths, is_feasible = algorithm.search()
                execution_time = time.time() - start_time
                
                # 結果の保存
                performance_info = algorithm.get_performance_info()
                performance_info['is_feasible'] = is_feasible
                performance_info['actual_execution_time'] = execution_time
                performance_info['num_paths_found'] = sum(len(paths) for paths in commodity_paths)
                performance_info['total_path_length'] = sum(
                    sum(len(path) for path in paths) for paths in commodity_paths
                )
                
                self.results[algorithm_name] = {
                    'performance': performance_info,
                    'commodity_paths': commodity_paths,
                    'is_feasible': is_feasible
                }
                
                # 比較データに追加
                self.comparison_data.append({
                    'algorithm': algorithm_name,
                    'execution_time': execution_time,
                    'is_feasible': is_feasible,
                    'num_paths_found': performance_info['num_paths_found'],
                    'total_path_length': performance_info['total_path_length'],
                    'beam_size': beam_size,
                    'batch_size': batch_size
                })
                
                print(f"  ✓ 完了: {execution_time:.4f}秒, 実行可能: {is_feasible}")
                
            except Exception as e:
                print(f"  ✗ エラー: {e}")
                self.results[algorithm_name] = {
                    'error': str(e),
                    'performance': None,
                    'commodity_paths': None,
                    'is_feasible': False
                }
        
        return self.results
    
    def print_comparison_table(self) -> None:
        """比較結果をテーブル形式で出力"""
        if not self.comparison_data:
            print("比較データがありません")
            return
        
        print(f"\n{'='*80}")
        print("ビームサーチアルゴリズム比較結果")
        print(f"{'='*80}")
        
        # ヘッダー
        print(f"{'アルゴリズム':<15} {'実行時間(秒)':<12} {'実行可能':<8} {'パス数':<8} {'総パス長':<10}")
        print("-" * 80)
        
        # データ行
        for data in self.comparison_data:
            print(f"{data['algorithm']:<15} {data['execution_time']:<12.4f} {str(data['is_feasible']):<8} {data['num_paths_found']:<8} {data['total_path_length']:<10}")
        
        print("-" * 80)
    
    def print_detailed_comparison(self) -> None:
        """詳細な比較結果を出力"""
        if not self.results:
            print("比較結果がありません")
            return
        
        print(f"\n{'='*80}")
        print("ビームサーチアルゴリズム詳細比較")
        print(f"{'='*80}")
        
        for algorithm_name, result in self.results.items():
            print(f"\n【{algorithm_name}】")
            print("-" * 40)
            
            if 'error' in result:
                print(f"エラー: {result['error']}")
                continue
            
            performance = result['performance']
            print(f"実行時間: {performance['execution_time']:.4f}秒")
            print(f"実行可能: {result['is_feasible']}")
            print(f"見つかったパス数: {performance['num_paths_found']}")
            print(f"総パス長: {performance['total_path_length']}")
            print(f"ビームサイズ: {performance['beam_size']}")
            print(f"バッチサイズ: {performance['batch_size']}")
            print(f"最大反復回数: {performance['max_iter']}")
    
    def export_results_json(self, filepath: str) -> None:
        """結果をJSONファイルにエクスポート"""
        if not self.comparison_data:
            print("エクスポートするデータがありません")
            return
        
        # JSON形式で保存可能なデータに変換
        export_data = []
        for data in self.comparison_data:
            export_data.append({
                'algorithm': data['algorithm'],
                'execution_time': float(data['execution_time']),
                'is_feasible': bool(data['is_feasible']),
                'num_paths_found': int(data['num_paths_found']),
                'total_path_length': int(data['total_path_length']),
                'beam_size': int(data['beam_size']),
                'batch_size': int(data['batch_size'])
            })
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"結果をJSONファイルにエクスポートしました: {filepath}")
    
    def find_best_algorithm(self, metric: str = 'execution_time') -> Dict[str, Any]:
        """
        指定された指標で最良のアルゴリズムを見つける
        
        Args:
            metric: 比較指標 ('execution_time', 'num_paths_found', 'total_path_length')
            
        Returns:
            Dict[str, Any]: 最良のアルゴリズムの情報
        """
        if not self.comparison_data:
            return {}
        
        if metric == 'execution_time':
            # 実行時間が短いほど良い
            best_data = min(self.comparison_data, key=lambda x: x[metric])
        else:
            # パス数やパス長は多いほど良い
            best_data = max(self.comparison_data, key=lambda x: x[metric])
        
        return best_data


class SimpleAlgorithmBenchmark:
    """簡略版アルゴリズムベンチマーククラス"""
    
    def __init__(self, comparator: SimpleBeamSearchComparator = None):
        self.comparator = comparator or SimpleBeamSearchComparator()
    
    def run_benchmark(self, 
                     test_cases: List[Dict[str, Any]],
                     algorithms: List[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        複数のテストケースでベンチマーク実行
        
        Args:
            test_cases: テストケースのリスト
            algorithms: テストするアルゴリズムのリスト
            
        Returns:
            Dict[str, List[Dict[str, Any]]]: ベンチマーク結果
        """
        benchmark_results = {}
        
        print(f"\n{'='*60}")
        print("ベンチマーク実行開始")
        print(f"{'='*60}")
        
        for i, test_case in enumerate(test_cases):
            print(f"\nテストケース {i+1}/{len(test_cases)}")
            print(f"設定: {test_case.get('description', 'No description')}")
            
            # テストケースの実行
            results = self.comparator.compare_algorithms(
                algorithms=algorithms,
                **{k: v for k, v in test_case.items() if k != 'description'}
            )
            
            benchmark_results[f"test_case_{i+1}"] = {
                'description': test_case.get('description', f'Test Case {i+1}'),
                'results': results
            }
            
            # 各テストケースの結果を表示
            self.comparator.print_comparison_table()
        
        return benchmark_results
    
    def generate_benchmark_report(self, benchmark_results: Dict[str, List[Dict[str, Any]]], 
                                 output_dir: str = "./benchmark_results") -> None:
        """ベンチマーク結果のレポート生成"""
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n{'='*60}")
        print("ベンチマークレポート生成")
        print(f"{'='*60}")
        
        # 各テストケースの結果をJSONに保存
        for test_name, test_data in benchmark_results.items():
            json_path = os.path.join(output_dir, f"{test_name}.json")
            
            # JSON形式で保存可能なデータに変換
            export_data = {
                'description': test_data['description'],
                'results': {}
            }
            
            for alg_name, result in test_data['results'].items():
                if 'performance' in result and result['performance']:
                    export_data['results'][alg_name] = {
                        'execution_time': float(result['performance']['execution_time']),
                        'is_feasible': bool(result['is_feasible']),
                        'num_paths_found': int(result['performance']['num_paths_found']),
                        'total_path_length': int(result['performance']['total_path_length']),
                        'beam_size': int(result['performance']['beam_size']),
                        'batch_size': int(result['performance']['batch_size'])
                    }
                else:
                    export_data['results'][alg_name] = {
                        'error': result.get('error', 'Unknown error')
                    }
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            print(f"テストケース結果を保存: {json_path}")
        
        # 総合レポート生成
        self._generate_summary_report(benchmark_results, output_dir)
    
    def _generate_summary_report(self, benchmark_results: Dict[str, List[Dict[str, Any]]], 
                                output_dir: str) -> None:
        """総合レポートの生成"""
        import os
        
        summary_data = []
        
        for test_name, test_data in benchmark_results.items():
            for algorithm_name, result in test_data['results'].items():
                if 'performance' in result and result['performance']:
                    summary_data.append({
                        'test_case': test_name,
                        'description': test_data['description'],
                        'algorithm': algorithm_name,
                        'execution_time': result['performance']['execution_time'],
                        'is_feasible': result['is_feasible'],
                        'num_paths_found': result['performance']['num_paths_found']
                    })
        
        if summary_data:
            # 総合レポートをJSON形式で保存
            summary_path = os.path.join(output_dir, "benchmark_summary.json")
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)
            
            print(f"総合レポートを生成: {summary_path}")
            
            # 総合レポートの表示
            print(f"\n{'='*60}")
            print("総合ベンチマーク結果")
            print(f"{'='*60}")
            
            print(f"{'テストケース':<15} {'アルゴリズム':<15} {'実行時間(秒)':<12} {'実行可能':<8} {'パス数':<8}")
            print("-" * 80)
            
            for data in summary_data:
                print(f"{data['test_case']:<15} {data['algorithm']:<15} {data['execution_time']:<12.4f} {str(data['is_feasible']):<8} {data['num_paths_found']:<8}")
            
            print("-" * 80) 