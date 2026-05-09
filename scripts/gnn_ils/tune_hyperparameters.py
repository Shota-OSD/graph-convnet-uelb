#!/usr/bin/env python3
"""
Hyperparameter Tuning Script for GNN-ILS Model

Usage:
  python scripts/gnn_ils/tune_hyperparameters.py --search grid --config configs/gnn_ils/gnn_ils_base.json
  python scripts/gnn_ils/tune_hyperparameters.py --search random --trials 20
  python scripts/gnn_ils/tune_hyperparameters.py --search random --config configs/gnn_ils/gnn_ils_base.json --tuning-config configs/gnn_ils/tuning_config.json --trials 30
"""

import argparse
import os
import sys
import json
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.gnn_ils.tuning.hyperparameter_tuner import GNNILSHyperparameterTuner
from src.common.config.config_manager import ConfigManager
from src.common.config.paths import dataset_exists


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter Tuning for GNN-ILS Model')

    parser.add_argument('--search', type=str, choices=['grid', 'random'], required=True,
                        help='探索手法: grid (グリッドサーチ) or random (ランダムサーチ)')
    parser.add_argument('--config', type=str, default='configs/gnn_ils/gnn_ils_base.json',
                        help='ベース設定ファイルのパス')
    parser.add_argument('--tuning-config', type=str, default='configs/gnn_ils/tuning_config.json',
                        help='チューニング設定ファイルのパス')
    parser.add_argument('--results-dir', type=str, default='tuning_results_gnn_ils',
                        help='結果保存ディレクトリ')
    parser.add_argument('--trials', type=int, default=None,
                        help='ランダムサーチの試行回数 (設定ファイルの値を上書き)')
    parser.add_argument('--seed', type=int, default=42,
                        help='ランダムシード')

    args = parser.parse_args()

    # シード設定
    import random
    import numpy as np
    import torch

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # ファイル存在確認
    if not os.path.exists(args.config):
        print(f"Error: Base config file not found: {args.config}")
        sys.exit(1)

    if not os.path.exists(args.tuning_config):
        print(f"Error: Tuning config file not found: {args.tuning_config}")
        sys.exit(1)

    # データ存在確認
    base_config = ConfigManager(args.config).get_config()
    if not dataset_exists(base_config):
        print("Error: Required data directories not found.")
        print(f"Please run: python scripts/common/generate_data.py --config {args.config}")
        sys.exit(1)

    print("=" * 60)
    print("HYPERPARAMETER TUNING FOR GNN-ILS MODEL")
    print("=" * 60)
    print(f"Search Method:    {args.search.upper()}")
    print(f"Base Config:      {args.config}")
    print(f"Tuning Config:    {args.tuning_config}")
    print(f"Results Directory: {args.results_dir}")
    print(f"Random Seed:      {args.seed}")
    if args.search == 'random' and args.trials:
        print(f"Number of Trials: {args.trials}")
    print(f"Start Time:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    tuner = GNNILSHyperparameterTuner(
        base_config_path=args.config,
        tuning_config_path=args.tuning_config,
        results_dir=args.results_dir,
    )

    try:
        if args.search == 'grid':
            best_result = tuner.grid_search()
        else:
            best_result = tuner.random_search(n_trials=args.trials)

        print("\n" + "=" * 60)
        print("TUNING COMPLETED SUCCESSFULLY!")
        print("=" * 60)

        if best_result:
            print("BEST RESULT:")
            print("-" * 30)
            print(f"Trial: {best_result['trial']}")
            print(f"Best Val Load Factor: {best_result['best_val_load_factor']:.4f}")

            if best_result.get('best_val_approx_ratio') is not None:
                print(f"Best Val Approx Ratio: {best_result['best_val_approx_ratio']:.2f}%")
            if best_result.get('final_train_load_factor') is not None:
                print(f"Final Train Load Factor: {best_result['final_train_load_factor']:.4f}")

            print("\nBest Parameters:")
            for param, value in best_result['params'].items():
                print(f"  {param}: {value}")

            print(f"\nResults saved in: {args.results_dir}")

            _create_best_config(best_result, args.config, args.results_dir)
        else:
            print("No valid results found.")

    except KeyboardInterrupt:
        print("\n\nTuning interrupted by user.")
        print(f"Partial results may be saved in: {args.results_dir}")
        sys.exit(1)
    except Exception as e:
        print(f"\nError occurred during tuning: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def _create_best_config(best_result, base_config_path, results_dir):
    """最適パラメータでの設定ファイルを作成"""
    config_manager = ConfigManager(base_config_path)
    base_config = config_manager.get_config()

    best_config = dict(base_config)
    best_config.update(best_result['params'])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    best_config_path = os.path.join(results_dir, f"best_config_{timestamp}.json")

    with open(best_config_path, 'w') as f:
        json.dump(best_config, f, indent=2)

    print(f"Best configuration saved to: {best_config_path}")
    print(f"Use this config with: python scripts/gnn_ils/train_gnn_ils.py --config {best_config_path}")


if __name__ == "__main__":
    main()
