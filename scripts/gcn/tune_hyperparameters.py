#!/usr/bin/env python3
"""
Hyperparameter Tuning Script for GCN UELB Model
使用例:
  python tune_hyperparameters.py --search grid --config configs/gcn/default.json
  python tune_hyperparameters.py --search random --trials 50
"""

import argparse
import os
import sys
from datetime import datetime

from src.gcn.tuning.hyperparameter_tuner import HyperparameterTuner


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='Hyperparameter Tuning for GCN UELB Model')

    # 基本設定
    parser.add_argument('--search', type=str, choices=['grid', 'random'], required=True,
                       help='探索手法: grid (グリッドサーチ) or random (ランダムサーチ)')
    parser.add_argument('--config', type=str, default='configs/gcn/default.json',
                       help='ベース設定ファイルのパス (default: configs/gcn/default.json)')
    parser.add_argument('--tuning-config', type=str, default='configs/gcn/tuning_config.json',
                       help='チューニング設定ファイルのパス (default: configs/gcn/tuning_config.json)')
    parser.add_argument('--results-dir', type=str, default='tuning_results',
                       help='結果保存ディレクトリ (default: tuning_results)')

    # ランダムサーチ専用
    parser.add_argument('--trials', type=int, default=None,
                       help='ランダムサーチの試行回数 (設定ファイルの値を上書き)')

    # その他
    parser.add_argument('--seed', type=int, default=42,
                       help='ランダムシード (default: 42)')

    args = parser.parse_args()

    # ランダムシードの設定
    import random
    import numpy as np
    import torch

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # 設定ファイルの存在確認
    if not os.path.exists(args.config):
        print(f"Error: Base config file not found: {args.config}")
        sys.exit(1)

    if not os.path.exists(args.tuning_config):
        print(f"Error: Tuning config file not found: {args.tuning_config}")
        sys.exit(1)

    # データの存在確認
    if not _check_data_exists():
        print("Error: Required data directories not found.")
        print("Please run 'python scripts/common/generate_data.py' first to create training data.")
        sys.exit(1)

    print("="*60)
    print("HYPERPARAMETER TUNING FOR GCN UELB MODEL")
    print("="*60)
    print(f"Search Method: {args.search.upper()}")
    print(f"Base Config: {args.config}")
    print(f"Tuning Config: {args.tuning_config}")
    print(f"Results Directory: {args.results_dir}")
    print(f"Random Seed: {args.seed}")
    if args.search == 'random' and args.trials:
        print(f"Number of Trials: {args.trials}")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # チューナーの初期化
    tuner = HyperparameterTuner(
        base_config_path=args.config,
        tuning_config_path=args.tuning_config,
        results_dir=args.results_dir
    )

    try:
        # チューニング実行
        if args.search == 'grid':
            print("Starting Grid Search...")
            best_result = tuner.grid_search()
        elif args.search == 'random':
            print("Starting Random Search...")
            best_result = tuner.random_search(n_trials=args.trials)

        # 結果表示
        print("\n" + "="*60)
        print("TUNING COMPLETED SUCCESSFULLY!")
        print("="*60)

        if best_result:
            print("BEST RESULT:")
            print("-" * 30)
            print(f"Trial: {best_result['trial']}")

            # Check if RL or supervised based on metrics available
            if 'test_mean_load_factor' in best_result and best_result.get('test_mean_load_factor', 0) > 0:
                # RL results
                print(f"Test Mean Load Factor: {best_result.get('test_mean_load_factor', 'N/A'):.4f}")
                print(f"Val Mean Load Factor: {best_result.get('val_mean_load_factor', 'N/A'):.4f}")
            else:
                # Supervised results
                print(f"Test Approximation Rate: {best_result['test_approximation_rate']:.2f}%")
                print(f"Val Approximation Rate: {best_result['val_approximation_rate']:.2f}%")

            print(f"Test Infeasible Rate: {best_result['test_infeasible_rate']:.2f}%")
            print(f"Val Infeasible Rate: {best_result['val_infeasible_rate']:.2f}%")

            print("\nBest Parameters:")
            for param, value in best_result['params'].items():
                print(f"  {param}: {value}")

            print(f"\nResults saved in: {args.results_dir}")

            # 最適設定ファイルの生成
            _create_best_config(best_result, args.config, args.results_dir)
        else:
            print("No valid results found.")

    except KeyboardInterrupt:
        print("\n\nTuning interrupted by user.")
        print("Partial results may be saved in:", args.results_dir)
        sys.exit(1)
    except Exception as e:
        print(f"\nError occurred during tuning: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


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


def _create_best_config(best_result, base_config_path, results_dir):
    """最適パラメータでの設定ファイルを作成"""
    import json
    from src.common.config.config_manager import ConfigManager

    # ベース設定を読み込み
    config_manager = ConfigManager(base_config_path)
    base_config = config_manager.get_config()

    # 最適パラメータを適用
    best_config = dict(base_config)
    best_config.update(best_result['params'])

    # 最適設定ファイルを保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    best_config_path = os.path.join(results_dir, f"best_config_{timestamp}.json")

    with open(best_config_path, 'w') as f:
        json.dump(best_config, f, indent=2)

    print(f"Best configuration saved to: {best_config_path}")
    print(f"Use this config with: python scripts/gcn/train_gcn.py --config {best_config_path}")


if __name__ == "__main__":
    main()
