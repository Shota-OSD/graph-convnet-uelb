#!/usr/bin/env python3
"""KSP-ILP Runner Script.

KSP 候補パス内で ILP を厳密に解き、最適なパス割当を求める。
事前計算済み CSV があればそれを読み、なければオンデマンドで解く。
"""

import sys
import os
import argparse
import csv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.common.config.config_manager import ConfigManager
from src.common.config.paths import (
    dataset_exists,
    get_exact_solution_file,
    get_ksp_ilp_solution_file,
)
from src.common.solvers.ksp_ilp import KspIlpSolver


def load_csv_column(filepath, col=0):
    """CSV ファイルの指定列を float dict で読み込む.

    Returns:
        {row_idx: value} の辞書
    """
    result = {}
    if not filepath.exists():
        return result
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader):
            if len(row) > col:
                try:
                    result[idx] = float(row[col])
                except ValueError:
                    pass
    return result


def main():
    parser = argparse.ArgumentParser(description='KSP-ILP Solver')
    parser.add_argument('--config', type=str, default='configs/rl_ksp/rl_config.json',
                        help='設定ファイルのパス')
    parser.add_argument('--mode', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='データモード (default: test)')
    parser.add_argument('--K', type=int, default=None,
                        help='KSP 候補パス数 (default: config の K 値)')
    parser.add_argument('--time-limit', type=int, default=30,
                        help='ILP ソルバーの制限時間 [秒] (default: 30)')
    parser.add_argument('--solver', type=str, default='HiGHS',
                        choices=['HiGHS', 'CBC'],
                        help='ILP ソルバー (default: HiGHS)')
    parser.add_argument('--recompute', action='store_true',
                        help='事前計算済み CSV を無視して再計算する')
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("KSP-ILP SOLVER")
    print("=" * 60)

    # 設定読込
    config_manager = ConfigManager(args.config)
    config = config_manager.get_config()

    # データ存在確認
    if not dataset_exists(config, modes=(args.mode,)):
        print(f"データが存在しません ({args.mode})。以下のコマンドでデータを生成してください:")
        print(f"  python scripts/common/generate_data.py --config {args.config}")
        sys.exit(1)

    K = args.K if args.K is not None else getattr(config, 'K', 10)
    num_data = getattr(config, f'num_{args.mode}_data', 20)

    # 事前計算済み CSV の確認
    ksp_ilp_file = get_ksp_ilp_solution_file(args.mode, config, K)
    precomputed = load_csv_column(ksp_ilp_file, col=0) if not args.recompute else {}
    use_precomputed = len(precomputed) >= num_data

    print(f"  Config: {args.config}")
    print(f"  Mode: {args.mode}")
    print(f"  K: {K}")
    print(f"  Solver: {args.solver}")
    print(f"  Time limit: {args.time_limit}s")
    print(f"  Num data: {num_data}")
    if use_precomputed:
        print(f"  Source: {ksp_ilp_file} (precomputed)")
    else:
        print(f"  Source: on-demand computation")

    # GT load factor 読込
    gt_lf = load_csv_column(get_exact_solution_file(args.mode, config), col=0)
    if not gt_lf:
        print("Warning: exact_solution.csv not found or empty")

    # ソルバー (事前計算がない場合のみ使用)
    solver = None if use_precomputed else KspIlpSolver(solver_name=args.solver, time_limit=args.time_limit)

    # 結果集計
    mlus = []
    approx_rates = []
    times = []
    infeasible_count = 0

    print("\n" + "-" * 60)
    print(f"{'idx':>4} | {'Status':>10} | {'MLU':>8} | {'GT MLU':>8} | {'Approx%':>8} | {'Time(s)':>8}")
    print("-" * 60)

    for idx in range(num_data):
        if use_precomputed:
            alpha = precomputed[idx]
            status = 'Optimal'
            elapsed = 0.0
        else:
            result = solver.solve_from_files(idx, args.mode, config, K)
            alpha = result.alpha
            status = result.status
            elapsed = result.elapsed_time

        times.append(elapsed)

        if status != 'Optimal':
            infeasible_count += 1
            gt_str = f"{gt_lf.get(idx, float('nan')):.4f}" if idx in gt_lf else "N/A"
            print(f"{idx:>4} | {status:>10} | {'N/A':>8} | {gt_str:>8} | {'N/A':>8} | {elapsed:>8.3f}")
            continue

        mlus.append(alpha)

        # Approximation Rate 計算
        gt_val = gt_lf.get(idx)
        if gt_val is not None and alpha > 0:
            approx = (gt_val / alpha) * 100.0
            approx_rates.append(approx)
            approx_str = f"{approx:>7.2f}%"
        else:
            approx_str = "N/A"

        gt_str = f"{gt_val:.4f}" if gt_val is not None else "N/A"
        print(f"{idx:>4} | {status:>10} | {alpha:>8.4f} | {gt_str:>8} | {approx_str:>8} | {elapsed:>8.3f}")

    # サマリー
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if mlus:
        avg_mlu = sum(mlus) / len(mlus)
        print(f"  Average MLU:           {avg_mlu:.6f}")
    if approx_rates:
        avg_approx = sum(approx_rates) / len(approx_rates)
        print(f"  Average Approx Rate:   {avg_approx:.2f}%")
    avg_time = sum(times) / len(times) if times else 0
    print(f"  Average Time:          {avg_time:.3f}s")
    print(f"  Optimal solutions:     {len(mlus)}/{num_data}")
    if infeasible_count > 0:
        print(f"  Infeasible:            {infeasible_count}/{num_data}")
    print("=" * 60)


if __name__ == '__main__':
    main()
