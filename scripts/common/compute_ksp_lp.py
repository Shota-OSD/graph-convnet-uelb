#!/usr/bin/env python3
"""既存データセットに対して KSP-LP を後付け計算するスタンドアロンスクリプト."""

import sys
import os
import csv
import argparse
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import networkx as nx
from src.common.data_management.ksp_lp_solution import SolveKspLpSolution
from src.common.config.paths import BUCKET_SIZE


def _count_existing_rows(filepath: Path) -> int:
    """CSV の行数を返す。ファイルが無ければ 0。"""
    if not filepath.exists():
        return 0
    try:
        with open(filepath, 'r') as f:
            return sum(1 for _ in csv.reader(f))
    except Exception:
        return 0


def _get_graph_path(mode_dir: Path, idx: int) -> Path:
    bucket = idx - (idx % BUCKET_SIZE)
    return mode_dir / "graph_file" / str(bucket) / f"graph_{idx}.gml"


def _get_commodity_path(mode_dir: Path, idx: int) -> Path:
    bucket = idx - (idx % BUCKET_SIZE)
    return mode_dir / "commodity_file" / str(bucket) / f"commodity_data_{idx}.csv"


def _load_commodity_list(filepath: Path) -> list:
    with open(filepath, newline='') as f:
        reader = csv.reader(f)
        return [[int(item) for item in row] for row in reader]


def main():
    parser = argparse.ArgumentParser(description='Compute KSP-LP for existing datasets')
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='データセットディレクトリ (例: ~/Projects/ml-data/.../gnn_ils_nsfnet_c10)')
    parser.add_argument('--modes', type=str, nargs='+', default=['test'],
                        choices=['train', 'val', 'test'],
                        help='計算するデータモード (default: test)')
    parser.add_argument('--K', type=int, default=10,
                        help='KSP 候補経路数 (default: 10)')
    parser.add_argument('--solver_time_limit', type=float, default=None,
                        help='ソルバー制限時間（秒）。未指定で制限なし')
    parser.add_argument('--resume', action='store_true',
                        help='既存の ksp_lp_solution.csv の続きから再開')
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).expanduser()
    if not dataset_dir.exists():
        print(f"Error: Dataset directory not found: {dataset_dir}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("KSP-LP COMPUTATION")
    print("=" * 60)
    print(f"Dataset: {dataset_dir}")
    print(f"Modes: {', '.join(args.modes)}")
    print(f"K: {args.K}")
    print(f"Solver time limit: {args.solver_time_limit or 'None (unlimited)'}")

    for mode in args.modes:
        mode_dir = dataset_dir / f"{mode}_data"
        if not mode_dir.exists():
            print(f"\nSkipping {mode}: {mode_dir} not found")
            continue

        # サンプル数を exact_solution.csv から検出
        exact_file = mode_dir / "exact_solution.csv"
        num_data = _count_existing_rows(exact_file)
        if num_data == 0:
            print(f"\nSkipping {mode}: exact_solution.csv is empty or missing")
            continue

        ksp_lp_file = mode_dir / "ksp_lp_solution.csv"

        # レジューム対応
        start_index = 0
        if args.resume:
            start_index = _count_existing_rows(ksp_lp_file)
            if start_index >= num_data:
                print(f"\n{mode}: All {num_data} samples already computed. Skipping.")
                continue
            if start_index > 0:
                print(f"\n{mode}: Resuming from index {start_index}/{num_data}")
        else:
            # 既存ファイルがあれば削除
            if ksp_lp_file.exists():
                ksp_lp_file.unlink()

        print(f"\n{mode.upper()}: Computing KSP-LP for {num_data - start_index} samples...")

        for i in range(start_index, num_data):
            graph_path = _get_graph_path(mode_dir, i)
            commodity_path = _get_commodity_path(mode_dir, i)

            if not graph_path.exists() or not commodity_path.exists():
                print(f"  Warning: Missing files for sample {i}, writing fallback row")
                with open(ksp_lp_file, 'a', newline='') as f:
                    csv.writer(f).writerow([1.0, 0.0, 0.0, 0])
                continue

            G = nx.read_gml(str(graph_path), destringizer=int)
            commodity_list = _load_commodity_list(commodity_path)

            try:
                solver = SolveKspLpSolution(
                    G, commodity_list, K=args.K,
                    solver_time_limit=args.solver_time_limit
                )
                result = solver.solve()

                with open(ksp_lp_file, 'a', newline='') as f:
                    csv.writer(f).writerow([
                        result['objective_value'],
                        result['elapsed_time'],
                        result['ksp_elapsed_time'],
                        1 if result['is_optimal'] else 0,
                    ])

                print(f"  Sample {i + 1}/{num_data}, "
                      f"KSP-LP MLU: {result['objective_value']:.4f}, "
                      f"Time: {result['elapsed_time']:.3f}s")

            except Exception as e:
                print(f"  Error for sample {i}: {e}")
                with open(ksp_lp_file, 'a', newline='') as f:
                    csv.writer(f).writerow([1.0, 0.0, 0.0, 0])

        final_count = _count_existing_rows(ksp_lp_file)
        print(f"  {mode} completed: {final_count} rows in ksp_lp_solution.csv")

    print("\n" + "=" * 60)
    print("KSP-LP computation finished.")
    print("=" * 60)


if __name__ == "__main__":
    main()
