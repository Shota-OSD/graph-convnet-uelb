#!/usr/bin/env python3
"""
Graph Convolutional Network for UELB (Undirected Edge Load Balancing)
Refactored main script with modular architecture
"""

import sys
import os
import argparse
from src.config.config_manager import ConfigManager
from src.train.trainer import Trainer
from src.train.evaluator import Evaluator
from src.train.metrics import MetricsLogger
from src.algorithms.rl_trainer import RLTrainer

def _check_data_exists():
    """データディレクトリの存在を確認"""
    data_dirs = ['./data/train_data', './data/val_data', './data/test_data']
    required_subdirs = ['commodity_file', 'graph_file', 'node_flow_file']
    
    for data_dir in data_dirs:
        if not os.path.exists(data_dir):
            return False
        
        # 必要なサブディレクトリが存在するかチェック
        for subdir in required_subdirs:
            subdir_path = os.path.join(data_dir, subdir)
            if not os.path.exists(subdir_path):
                return False
        
        # commodity_fileディレクトリに数値名のサブディレクトリが存在するかチェック
        commodity_dir = os.path.join(data_dir, 'commodity_file')
        subdirs = [d for d in os.listdir(commodity_dir) 
                  if os.path.isdir(os.path.join(commodity_dir, d)) and d.isdigit()]
        if len(subdirs) == 0:
            return False
            
    return True

def main():
    """メイン関数"""
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description='Graph Convolutional Network for UELB')
    parser.add_argument('--config', type=str, default='configs/default2.json',
                       help='設定ファイルのパス (default: configs/default2.json)')
    parser.add_argument('--mode', type=str, default='gcn', choices=['gcn', 'rl'],
                       help='訓練モード: gcn (GCN訓練) or rl (強化学習) (default: gcn)')
    args = parser.parse_args()
    
    # 設定の初期化
    config_manager = ConfigManager(args.config)
    config = config_manager.get_config()
    dtypeFloat, dtypeLong = config_manager.get_dtypes()
    
    # 強化学習モードかGCNモードかで分岐
    if args.mode == 'rl':
        # 強化学習モード
        print("\n" + "="*60)
        print("REINFORCEMENT LEARNING MODE")
        print("="*60)
        
        # データの存在確認
        if not _check_data_exists():
            print("データが存在しません。以下のコマンドでデータを生成してください:")
            print(f"python generate_data.py --config {args.config}")
            sys.exit(1)
        
        rl_trainer = RLTrainer(dict(config))
        
        # 学習の実行
        rl_trainer.train()
        
        # テストの実行
        test_episodes = config.get('test_episodes', 100)
        rl_trainer.test(test_episodes)
        
        # 設定情報を辞書形式で準備（GCNと同じ形式）
        safe_config_info = {
            'config_file': args.config,
            'mode': 'reinforcement_learning',
            'episodes': config.get('episodes', 100),
            'test_episodes': test_episodes,
            'num_train_data': config.get('num_train_data', 100),
            'num_test_data': config.get('num_test_data', 20),
            'K': config.get('K', 10),
            'n_action': config.get('n_action', 20),
            'learning_rate': config.get('learning_rate', 0.0005),
            'epsilon': config.get('epsilon', 0.8),
            'gamma': config.get('gamma', 0.85)
        }
        
        # 結果の出力とログ保存（GCNと同じスタイル）
        rl_trainer.metrics_logger.print_summary(safe_config_info)
        rl_trainer.metrics_logger.save_results(safe_config_info)
        
        print("\n" + "="*60)
        print("REINFORCEMENT LEARNING COMPLETED")
        print("="*60)
        return
    
    # GCNモード（従来の処理）
    print("\n" + "="*60)
    print("GRAPH CONVOLUTIONAL NETWORK MODE")
    print("="*60)
    
    # データの存在確認
    if not _check_data_exists():
        print("データが存在しません。以下のコマンドでデータを生成してください:")
        print(f"python generate_data.py --config {args.config}")
        sys.exit(1)
    
    # トレーナーとエバリュエーターの初期化
    trainer = Trainer(config, dtypeFloat, dtypeLong)
    evaluator = Evaluator(config, dtypeFloat, dtypeLong)
    metrics_logger = MetricsLogger()
    
    # load_saved_modelがtrueの場合は評価のみ実行
    if config.get('load_saved_model', False):
        # 評価のみ実行
        val_result, test_result = trainer.evaluate_only(evaluator, metrics_logger)

        # 設定情報を辞書形式で準備（評価モード用）
        safe_config_info = {
            'config_file': args.config,
            'mode': 'evaluation_only',
            'model_hash': trainer._get_config_hash(),
            'num_val_data': config.get('num_val_data', 0),
            'num_test_data': config.get('num_test_data', 0),
        }

        # 評価結果のサマリーを表示
        metrics_logger.print_evaluation_summary(safe_config_info)

        # 評価結果をファイルに保存
        metrics_logger.save_results(safe_config_info)

        print("\n" + "="*60)
        print("EVALUATION COMPLETED")
        print("="*60)
        return
    
    # トレーニング実行
    trainer.train(evaluator, metrics_logger)
    
    # 設定情報を辞書形式で準備（安全な形式のみ）
    safe_config_info = {
        'config_file': args.config,
        'mode': 'training',
        'model_hash': trainer._get_config_hash(),
        'max_epochs': max_epochs,
        'num_train_data': config.get('num_train_data', 0),
        'num_test_data': config.get('num_test_data', 0),
        'num_val_data': config.get('num_val_data', 0),
    }
    
    # 結果の出力
    metrics_logger.print_summary(safe_config_info)
    
    # 設定オブジェクトから安全に変換できる属性のみを追加
    try:
        config_dict = dict(config)
        # 基本的な設定値のみを抽出
        basic_config = {}
        for key, value in config_dict.items():
            if isinstance(value, (int, float, str, bool, list, dict)) or value is None:
                basic_config[key] = value
        safe_config_info['basic_config'] = basic_config
    except Exception as e:
        print(f"Warning: Could not extract full config: {e}")
        safe_config_info['basic_config'] = {}
    
    # 結果をファイルに保存
    metrics_logger.save_results(safe_config_info)
    
    print("Training finished.")

if __name__ == "__main__":
    main() 