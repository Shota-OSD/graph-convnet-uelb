#!/usr/bin/env python3
"""
GCN Training and Evaluation Script
Trains GCN model and evaluates on test data
"""

import sys
import os
import argparse

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.gcn.train.trainer import Trainer
from src.gcn.train.evaluator import Evaluator
from src.gcn.train.metrics import MetricsLogger
from src.common.config.config_manager import ConfigManager


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
    parser = argparse.ArgumentParser(description='GCN Training and Evaluation')
    parser.add_argument('--config', type=str, default='configs/gcn/default2.json',
                       help='設定ファイルのパス (default: configs/gcn/default2.json)')
    args = parser.parse_args()

    print("\n" + "="*60)
    print("GCN (GRAPH CONVOLUTIONAL NETWORK) TRAINING")
    print("="*60)

    # データの存在確認
    if not _check_data_exists():
        print("データが存在しません。以下のコマンドでデータを生成してください:")
        print(f"python scripts/common/generate_data.py --config {args.config}")
        sys.exit(1)

    # 設定の初期化
    config_manager = ConfigManager(args.config)
    config = config_manager.get_config()
    dtypeFloat, dtypeLong = config_manager.get_dtypes()

    # トレーナーとエバリュエーターの初期化
    trainer = Trainer(config, dtypeFloat, dtypeLong)
    evaluator = Evaluator(config, dtypeFloat, dtypeLong, strategy=trainer.strategy)
    metrics_logger = MetricsLogger()

    # load_saved_modelがtrueの場合は評価のみ実行
    if config.get('load_saved_model', False):
        # 評価のみ実行
        val_result, test_result = evaluator.evaluate_saved_model(trainer, metrics_logger)

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
        'max_epochs': config.get('max_epochs', 0),
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

    print("\n" + "="*60)
    print("GCN TRAINING COMPLETED")
    print("="*60)


if __name__ == "__main__":
    main()
