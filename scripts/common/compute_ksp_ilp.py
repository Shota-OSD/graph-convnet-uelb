#!/usr/bin/env python3
"""KSP-ILP 事前計算スクリプト.

既存データセットに対して KSP-ILP を計算し、ksp_ilp_K{K}_solution.csv に保存する。
train/val/test 全モードを一括処理する。
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.common.config.config_manager import ConfigManager
from src.common.config.paths import dataset_exists, get_mode_dir
from src.common.data_management.create_data_files import compute_ksp_ilp_solutions


def main():
    parser = argparse.ArgumentParser(description='KSP-ILP 事前計算')
    parser.add_argument('--config', type=str, required=True,
                        help='設定ファイルのパス')
    parser.add_argument('--K', type=int, default=None,
                        help='KSP 候補パス数 (default: config の K 値)')
    parser.add_argument('--time-limit', type=int, default=30,
                        help='ILP ソルバーの制限時間 [秒] (default: 30)')
    parser.add_argument('--recompute', action='store_true',
                        help='既存の CSV を削除して再計算する')
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("KSP-ILP PRE-COMPUTATION")
    print("=" * 60)

    config_manager = ConfigManager(args.config)
    config = config_manager.get_config()

    K = args.K if args.K is not None else getattr(config, 'K', 10)

    print(f"  Config: {args.config}")
    print(f"  K: {K}")
    print(f"  Time limit: {args.time_limit}s")

    modes = ['train', 'val', 'test']

    for mode in modes:
        if not dataset_exists(config, modes=(mode,)):
            print(f"\n[{mode}] データが存在しません。スキップ。")
            continue

        num_data = getattr(config, f'num_{mode}_data', 0)
        if num_data == 0:
            print(f"\n[{mode}] num_{mode}_data=0。スキップ。")
            continue

        # --recompute の場合、既存 CSV を削除
        if args.recompute:
            from src.common.config.paths import get_ksp_ilp_solution_file
            csv_path = get_ksp_ilp_solution_file(mode, config, K)
            if csv_path.exists():
                csv_path.unlink()
                print(f"\n[{mode}] 既存 CSV を削除: {csv_path}")

        print(f"\n[{mode}] {num_data} 件")
        compute_ksp_ilp_solutions(config, mode, num_data, K, args.time_limit)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == '__main__':
    main()
