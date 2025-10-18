#!/usr/bin/env python3
"""
RL-KSP Training and Testing Script
Trains RL-KSP model using DQN and evaluates on test data
"""

import sys
import os
import argparse

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.rl_ksp.train.rl_trainer import RLTrainer
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
    parser = argparse.ArgumentParser(description='RL-KSP Training and Testing')
    parser.add_argument('--config', type=str, default='configs/rl_ksp/rl_config.json',
                       help='設定ファイルのパス (default: configs/rl_ksp/rl_config.json)')
    args = parser.parse_args()

    print("\n" + "="*60)
    print("RL-KSP (REINFORCEMENT LEARNING WITH K-SHORTEST PATHS)")
    print("="*60)

    # データの存在確認
    if not _check_data_exists():
        print("データが存在しません。以下のコマンドでデータを生成してください:")
        print(f"python scripts/common/generate_data.py --config {args.config}")
        sys.exit(1)

    # 設定の初期化
    config_manager = ConfigManager(args.config)
    config = config_manager.get_config()

    # RL トレーナーの初期化
    rl_trainer = RLTrainer(dict(config))

    # 学習の実行
    print("\n" + "="*60)
    print("TRAINING PHASE")
    print("="*60)
    rl_trainer.train()

    # テストの実行
    print("\n" + "="*60)
    print("TESTING PHASE")
    print("="*60)
    test_episodes = config.get('test_episodes', 100)
    rl_trainer.test(test_episodes)

    # 設定情報を辞書形式で準備
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

    # 結果の出力とログ保存
    rl_trainer.metrics_logger.print_summary(safe_config_info)
    rl_trainer.metrics_logger.save_results(safe_config_info)

    print("\n" + "="*60)
    print("RL-KSP TRAINING COMPLETED")
    print("="*60)


if __name__ == "__main__":
    main()
