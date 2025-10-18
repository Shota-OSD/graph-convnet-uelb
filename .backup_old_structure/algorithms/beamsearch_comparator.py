#!/usr/bin/env python3
"""
ビームサーチ性能比較ツール
トレーニング済みモデルを使用してビームサーチの効果を測定
"""

import torch
import time
import json
import csv
import os
import argparse
import numpy as np
from typing import List, Tuple, Dict, Any
from sklearn.utils.class_weight import compute_class_weight

# 相対インポートを絶対インポートに変更
import sys
sys.path.append('.')

from src.config.config_manager import ConfigManager
from src.train.trainer import Trainer
from src.data_management.dataset_reader import DatasetReader
from src.algorithms.beamsearch_uelb import BeamsearchUELB, BeamSearchFactory
from src.models.model_utils import mean_feasible_load_factor

class BeamSearchComparator:
    """ビームサーチ性能比較ツール"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.results = []
        self._setup_output_dir()
        self._load_model_and_config()
    
    def _setup_output_dir(self):
        """出力ディレクトリの準備"""
        self.output_dir = self.config.get('output_dir', './beamsearch_results')
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _load_model_and_config(self):
        """トレーニング済みモデルと設定の読み込み"""
        model_config = self.config.get('model_config', {})
        config_path = model_config.get('config_path')
        
        if not config_path or not os.path.exists(config_path):
            raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")
        
        # 設定の読み込み
        self.config_manager = ConfigManager(config_path)
        self.train_config = self.config_manager.get_config()
        self.dtypeFloat, self.dtypeLong = self.config_manager.get_dtypes()
        
        # トレーナーの初期化（モデル読み込み）
        self.trainer = Trainer(self.train_config, self.dtypeFloat, self.dtypeLong)
        self.model = self.trainer.get_model()
        self.model.eval()
        
        print(f"モデルを読み込みました")
        print(f"設定: {config_path}")
        print(f"モデルハッシュ: {self.trainer._get_config_hash()}")
    
    def _get_model_predictions(self, batch):
        """モデルから予測を取得"""
        # テンソルに変換
        x_edges = torch.LongTensor(batch.edges).contiguous()
        x_edges_capacity = torch.FloatTensor(batch.edges_capacity).contiguous()
        x_nodes = torch.LongTensor(batch.nodes).contiguous()
        y_edges = torch.LongTensor(batch.edges_target).contiguous()
        batch_commodities = torch.LongTensor(batch.commodities).contiguous()
        x_commodities = batch_commodities[:, :, 2].to(torch.float)
        
        # GPU対応
        device = next(self.model.parameters()).device
        x_edges = x_edges.to(device)
        x_edges_capacity = x_edges_capacity.to(device)
        x_nodes = x_nodes.to(device)
        y_edges = y_edges.to(device)
        batch_commodities = batch_commodities.to(device)
        x_commodities = x_commodities.to(device)
        
        # モデル推論
        with torch.no_grad():
            edge_labels = y_edges.cpu().numpy().flatten()
            edge_cw = compute_class_weight("balanced", classes=np.unique(edge_labels), y=edge_labels)
            y_preds, loss = self.model.forward(x_edges, x_commodities, x_edges_capacity, x_nodes, y_edges, edge_cw)
        
        return y_preds, x_edges_capacity, batch_commodities, batch.load_factor
    
    def run_comparison(self):
        """ビームサーチ比較を実行"""
        print(f"\n{'='*80}")
        print(f"ビームサーチ比較実行: {self.config['comparison_name']}")
        print(f"{'='*80}")
        
        test_config = self.config.get('test_config', {})
        algorithm_configs = self.config['algorithms']
        
        # テストデータの読み込み
        data_mode = test_config.get('data_mode', 'test')
        num_test_data = self.train_config.get(f'num_{data_mode}_data', 20)
        batch_size = test_config.get('batch_size') or self.train_config.get('batch_size', 1)
        num_batches = test_config.get('num_batches', 5)
        
        dataset = DatasetReader(num_test_data, batch_size, data_mode)
        batches_to_test = min(num_batches, dataset.max_iter)
        
        print(f"テストデータ: {data_mode}")
        print(f"バッチ数: {batches_to_test}, バッチサイズ: {batch_size}")
        print(f"アルゴリズム設定数: {len(algorithm_configs)}")
        
        batch_count = 0
        for batch in dataset:
            if batch_count >= batches_to_test:
                break
            
            print(f"\nバッチ {batch_count + 1}/{batches_to_test}")
            
            # モデル予測を取得
            y_preds, edges_capacity, commodities, gt_load_factors = self._get_model_predictions(batch)
            
            # 各アルゴリズム設定でテスト
            for algorithm_config in algorithm_configs:
                algorithm_name = algorithm_config['name']
                print(f"  実行中: {algorithm_name}")
                
                try:
                    result = self._run_algorithm(
                        algorithm_config, batch_count,
                        y_preds, edges_capacity, commodities, gt_load_factors
                    )
                    
                    self.results.append(result)
                    print(f"    ✓ 完了: {result['execution_time']:.4f}秒, 近似率: {result['approximation_rate']:.2f}%")
                    
                except Exception as e:
                    error_result = {
                        'batch_id': batch_count,
                        'algorithm_name': algorithm_name,
                        'algorithm_type': algorithm_config.get('algorithm_type', 'unknown'),
                        'beam_size': algorithm_config['beam_size'],
                        'error': str(e),
                        'execution_time': 0,
                        'approximation_rate': 0,
                        'infeasible_rate': 100,
                        'mean_load_factor': 0
                    }
                    self.results.append(error_result)
                    print(f"    ✗ エラー: {e}")
            
            batch_count += 1
        
        # 結果の出力
        self._output_results()
    
    def _run_algorithm(self, algorithm_config: Dict[str, Any], batch_id: int,
                      y_preds: torch.Tensor, edges_capacity: torch.Tensor,
                      commodities: torch.Tensor, gt_load_factors: np.ndarray) -> Dict[str, Any]:
        """アルゴリズムの実行と評価"""
        algorithm_type = algorithm_config['algorithm_type']
        algorithm_name = algorithm_config['name']
        beam_size = algorithm_config['beam_size']
        mode_strict = algorithm_config.get('mode_strict', True)
        batch_size = y_preds.shape[0]
        
        # BeamSearchFactoryを使用してアルゴリズムを作成
        beam_search = BeamSearchFactory.create_algorithm(
            algorithm_type,
            y_pred_edges=y_preds,
            beam_size=beam_size,
            batch_size=batch_size,
            edges_capacity=edges_capacity,
            commodities=commodities,
            dtypeFloat=self.dtypeFloat,
            dtypeLong=self.dtypeLong,
            mode_strict=mode_strict,
            max_iter=5
        )
        
        start_time = time.time()
        pred_paths, is_feasible = beam_search.search()
        execution_time = time.time() - start_time
        
        # 負荷率の計算
        mean_maximum_load_factor, _ = mean_feasible_load_factor(
            batch_size, self.train_config.num_commodities, self.train_config.num_nodes,
            pred_paths, edges_capacity, commodities
        )
        
        # 近似率の計算
        if mean_maximum_load_factor > 1:
            approximation_rate = 0
            infeasible_rate = 100
            mean_load_factor = 0
        else:
            gt_load_factor = np.mean(gt_load_factors)
            approximation_rate = gt_load_factor / mean_maximum_load_factor * 100 if mean_maximum_load_factor != 0 else 0
            infeasible_rate = 0
            mean_load_factor = mean_maximum_load_factor
        
        return {
            'batch_id': batch_id,
            'algorithm_name': algorithm_name,
            'algorithm_type': algorithm_type,
            'beam_size': beam_size,
            'mode_strict': mode_strict,
            'execution_time': execution_time,
            'approximation_rate': approximation_rate,
            'infeasible_rate': infeasible_rate,
            'mean_load_factor': mean_load_factor,
            'is_feasible': mean_maximum_load_factor <= 1
        }
    
    def _output_results(self):
        """結果の出力"""
        output_formats = self.config.get('output_formats', ['console'])
        
        if 'console' in output_formats:
            self._output_console()
        
        if 'json' in output_formats:
            self._output_json()
        
        if 'csv' in output_formats:
            self._output_csv()
    
    def _output_console(self):
        """コンソール出力"""
        print(f"\n{'='*100}")
        print("ビームサーチ性能比較結果")
        print(f"{'='*100}")
        
        # ヘッダー
        print(f"{'バッチID':<8} {'アルゴリズム':<20} {'タイプ':<12} {'サイズ':<6} {'実行時間(秒)':<12} {'近似率(%)':<10} {'負荷率':<8} {'実行可能':<8}")
        print("-" * 120)
        
        # データ行
        for result in self.results:
            if 'error' in result:
                print(f"{result['batch_id']:<8} {result['algorithm_name']:<20} {result['algorithm_type']:<12} {result['beam_size']:<6} {'ERROR':<12} {'0.00':<10} {'0.00':<8} {'False':<8}")
            else:
                print(f"{result['batch_id']:<8} {result['algorithm_name']:<20} {result['algorithm_type']:<12} {result['beam_size']:<6} "
                      f"{result['execution_time']:<12.4f} {result['approximation_rate']:<10.2f} "
                      f"{result['mean_load_factor']:<8.3f} {str(result['is_feasible']):<8}")
        
        print("-" * 100)
        
        # 統計情報
        self._print_statistics()
    
    def _print_statistics(self):
        """統計情報の表示"""
        successful_results = [r for r in self.results if 'error' not in r]
        
        if successful_results:
            print("\n統計情報 (アルゴリズム別):")
            
            # アルゴリズム別統計
            algorithm_stats = {}
            for result in successful_results:
                algorithm_name = result['algorithm_name']
                if algorithm_name not in algorithm_stats:
                    algorithm_stats[algorithm_name] = {
                        'times': [],
                        'approx_rates': [],
                        'load_factors': [],
                        'feasible_count': 0,
                        'total_count': 0,
                        'algorithm_type': result['algorithm_type'],
                        'beam_size': result['beam_size']
                    }
                
                stats = algorithm_stats[algorithm_name]
                stats['times'].append(result['execution_time'])
                stats['approx_rates'].append(result['approximation_rate'])
                stats['load_factors'].append(result['mean_load_factor'])
                stats['total_count'] += 1
                
                if result['is_feasible']:
                    stats['feasible_count'] += 1
            
            for algorithm_name, stats in algorithm_stats.items():
                avg_time = sum(stats['times']) / len(stats['times'])
                avg_approx = sum(stats['approx_rates']) / len(stats['approx_rates'])
                avg_load = sum(stats['load_factors']) / len(stats['load_factors'])
                success_rate = stats['feasible_count'] / stats['total_count'] * 100
                
                print(f"  {algorithm_name} ({stats['algorithm_type']}, beam={stats['beam_size']}):")
                print(f"    平均実行時間: {avg_time:.4f}秒")
                print(f"    平均近似率: {avg_approx:.2f}%")
                print(f"    平均負荷率: {avg_load:.3f}")
                print(f"    成功率: {success_rate:.1f}%")
    
    def _output_json(self):
        """JSON出力"""
        json_path = os.path.join(self.output_dir, 'comparison_results.json')
        
        output_data = {
            'comparison_name': self.config['comparison_name'],
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'config': self.config,
            'results': self.results
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nJSON結果を保存: {json_path}")
    
    def _output_csv(self):
        """CSV出力"""
        csv_path = os.path.join(self.output_dir, 'beamsearch_comparison.csv')
        
        if self.results:
            fieldnames = ['batch_id', 'algorithm_name', 'algorithm_type', 'beam_size', 'mode_strict', 
                         'execution_time', 'approximation_rate', 'infeasible_rate',
                         'mean_load_factor', 'is_feasible']
            
            # エラー結果も含める
            if any('error' in r for r in self.results):
                fieldnames.append('error')
            
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for result in self.results:
                    row = {k: result.get(k, '') for k in fieldnames}
                    writer.writerow(row)
            
            print(f"CSV結果を保存: {csv_path}")

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='ビームサーチ性能比較ツール')
    parser.add_argument('--config', '-c', type=str, 
                       default='src/algorithms/configs/algorithm_comparison.json',
                       help='比較設定ファイルのパス')
    
    args = parser.parse_args()
    
    # 設定ファイルの存在確認
    if not os.path.exists(args.config):
        print(f"エラー: 設定ファイルが見つかりません: {args.config}")
        return 1
    
    try:
        # ビームサーチ比較実行
        comparator = BeamSearchComparator(args.config)
        comparator.run_comparison()
        
        print(f"\n比較完了。結果は {comparator.output_dir} に保存されました。")
        return 0
        
    except Exception as e:
        print(f"エラー: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())