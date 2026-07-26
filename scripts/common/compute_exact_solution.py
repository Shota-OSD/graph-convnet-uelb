#!/usr/bin/env python3
"""--skip-exact で生成したデータセットに厳密解を後付け計算するスクリプト.

既存の graph_*.gml + commodity_*.csv を読み込み、SolveExactSolution を実行して
exact_solution.csv を生成・追記する。config の solver_ratio_gap / solver_time_limit
がそのまま適用される。

使い方:
    # 全モード後付け計算
    python scripts/common/compute_exact_solution.py --config configs/rl_ksp/nsfnet_c30_rho04.json

    # パイロット: 5件だけ計算して時間見積もり
    python scripts/common/compute_exact_solution.py --config configs/rl_ksp/nsfnet_c30_rho04.json --num-samples 5

    # 既存 CSV を削除して再計算
    python scripts/common/compute_exact_solution.py --config configs/rl_ksp/nsfnet_c30_rho04.json --recompute

    # time_limit を上書き
    python scripts/common/compute_exact_solution.py --config configs/rl_ksp/nsfnet_c30_rho04.json --time-limit 60
"""

import sys
import os
import csv
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.common.config.config_manager import ConfigManager
from src.common.config.paths import (
    get_exact_solution_file,
    get_graph_file,
    get_commodity_file,
    get_mode_dir,
    BUCKET_SIZE,
)
from src.common.solvers.exact_ilp import SolveExactSolution


def compute_exact_solutions(config, mode: str, num_data: int,
                             time_limit: int, ratio_gap, recompute: bool = False):
    """既存データに対して厳密解を後付け計算し exact_solution.csv に保存する.

    Args:
        config: ConfigManager から取得した設定オブジェクト
        mode: 'train' / 'val' / 'test'
        num_data: 対象サンプル数
        time_limit: ILP ソルバーの制限時間 [秒]
        ratio_gap: CBC の ratioGap オプション（None なら真の最適解を要求）
        recompute: True なら既存 CSV を削除して最初から計算
    """
    solver_type = getattr(config, 'solver_type', 'pulp')
    exact_file = get_exact_solution_file(mode, config)

    if recompute and exact_file.exists():
        exact_file.unlink()
        print(f"[{mode}] 既存 CSV を削除: {exact_file}")

    # 再開サポート: 既存行数分はスキップ
    start_index = 0
    if exact_file.exists():
        try:
            with open(exact_file, 'r') as f:
                start_index = sum(1 for _ in csv.reader(f))
        except Exception:
            start_index = 0

    if start_index >= num_data:
        print(f"[{mode}] 全 {num_data} 件計算済み。スキップ。")
        return

    if start_index > 0:
        print(f"[{mode}] {start_index}/{num_data} 件完了済み。{start_index} から再開。")

    gap_str = f"ratioGap={ratio_gap}" if ratio_gap is not None else "厳密最適"
    print(f"[{mode}] {num_data - start_index} 件計算開始 (time_limit={time_limit}s, {gap_str})")

    skipped = 0
    warned = 0

    for i in range(start_index, num_data):
        graph_file = get_graph_file(mode, i, config)
        commodity_file = get_commodity_file(mode, i, config)

        if not graph_file.exists() or not commodity_file.exists():
            print(f"  [{mode}:{i}] ファイルが存在しません。スキップ。"
                  f" graph={graph_file.exists()} commodity={commodity_file.exists()}")
            # CSV の行数を合わせるため NaN 行を書く
            with open(exact_file, 'a', newline='') as f:
                csv.writer(f).writerow([None, None, None])
            skipped += 1
            continue

        try:
            solver = SolveExactSolution(solver_type, str(commodity_file), str(graph_file))
            _, _, obj_val, elapsed_time, is_optimal, mip_gap = solver.solve_exact_solution_to_env(
                time_limit=time_limit, ratio_gap=ratio_gap
            )
        except Exception as e:
            print(f"  [{mode}:{i}] エラー: {e}")
            with open(exact_file, 'a', newline='') as f:
                csv.writer(f).writerow([None, None, None])
            skipped += 1
            continue

        if obj_val is None or obj_val >= 1.0:
            print(f"  ⚠️  [{mode}:{i}] obj_val={obj_val} (>= 1.0 または None)。"
                  f" --skip-exact 時の再生成ループをスキップしたサンプルの可能性。そのまま記録。")
            warned += 1

        with open(exact_file, 'a', newline='') as f:
            csv.writer(f).writerow([obj_val, elapsed_time, mip_gap])

        if i % BUCKET_SIZE == 0:
            gap_info = f", Gap={mip_gap:.4f}" if mip_gap is not None else ""
            opt_mark = "✓" if is_optimal else "~"
            print(f"  {opt_mark} [{mode}:{i}/{num_data}] MLU={obj_val:.6f}{gap_info} ({elapsed_time:.1f}s)")

    print(f"[{mode}] 完了: {num_data} 件 (スキップ={skipped}, 警告={warned})")


def main():
    parser = argparse.ArgumentParser(description='--skip-exact データへの厳密解後付け計算')
    parser.add_argument('--config', type=str, required=True, help='設定ファイルのパス')
    parser.add_argument('--modes', type=str, nargs='+', default=['train', 'val', 'test'],
                        choices=['train', 'val', 'test'], help='対象モード (default: train val test)')
    parser.add_argument('--time-limit', type=int, default=None,
                        help='ILP ソルバー制限時間 [秒]（省略時は config の solver_time_limit を使用）')
    parser.add_argument('--recompute', action='store_true',
                        help='既存の exact_solution.csv を削除して最初から計算')
    parser.add_argument('--num-samples', type=int, default=None,
                        help='パイロットテスト用: 指定数だけ計算して時間を見積もる')
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("EXACT SOLUTION POST-COMPUTATION")
    print("=" * 60)

    config_manager = ConfigManager(args.config)
    config = config_manager.get_config()

    solver_time_limit = args.time_limit if args.time_limit is not None else getattr(config, 'solver_time_limit', 30)
    ratio_gap = getattr(config, 'solver_ratio_gap', None)

    print(f"  Config:      {args.config}")
    print(f"  Solver:      {getattr(config, 'solver_type', 'pulp')}")
    print(f"  Time limit:  {solver_time_limit}s")
    print(f"  Ratio gap:   {ratio_gap if ratio_gap is not None else '未指定（厳密最適）'}")
    if args.num_samples:
        print(f"  Pilot test:  {args.num_samples} samples only")

    for mode in args.modes:
        mode_dir = get_mode_dir(mode, config)
        if not mode_dir.exists():
            print(f"\n[{mode}] ディレクトリが存在しません。スキップ。")
            continue

        num_data = getattr(config, f'num_{mode}_data', 0)
        if num_data == 0:
            print(f"\n[{mode}] num_{mode}_data=0。スキップ。")
            continue

        effective_num = min(num_data, args.num_samples) if args.num_samples else num_data
        print(f"\n[{mode}] {effective_num}/{num_data} 件")

        compute_exact_solutions(
            config, mode, effective_num,
            time_limit=solver_time_limit,
            ratio_gap=ratio_gap,
            recompute=args.recompute,
        )

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == '__main__':
    main()
