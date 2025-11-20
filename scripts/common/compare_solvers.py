"""
全てのソルバータイプ（mip, pulp, SCIP）で厳密解を求め、計算時間と最大負荷率を比較するスクリプト

使い方:
    # デフォルトファイル（results/graph.gml, results/commodity_data.csv）を使用
    python3 scripts/common/compare_solvers.py

    # カスタムファイルを指定
    python3 scripts/common/compare_solvers.py <graph_file> <commodity_file>

    # 例: 訓練データの最初のサンプルを使用
    python3 scripts/common/compare_solvers.py results/graph.gml data/train_data/commodity_file/0/commodity_data_0.csv

出力:
    - 各ソルバーの計算時間
    - 各ソルバーの最大負荷率
    - 最大負荷率が異なる場合は警告を表示
    - 計算時間の比較（最速/最遅のソルバー、速度比）
"""
import sys
import os
from pathlib import Path

# プロジェクトルートのパスを追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.common.data_management.exact_solution import SolveExactSolution
import time

# coloramaがあれば使用、なければプレーンテキスト
try:
    from colorama import Fore, Style, init
    init(autoreset=True)
    HAS_COLOR = True
except ImportError:
    # coloramaがない場合のダミークラス
    class Fore:
        GREEN = ""
        RED = ""
        YELLOW = ""
    class Style:
        RESET_ALL = ""
    HAS_COLOR = False

def compare_solvers(graph_file, commodity_file):
    """
    全てのソルバータイプで最適化問題を解き、結果を比較する

    Args:
        graph_file: グラフファイルのパス (.gml)
        commodity_file: 需要データファイルのパス (.csv)

    Returns:
        dict: 各ソルバーの結果を含む辞書
    """
    solver_types = ['mip', 'pulp', 'SCIP']
    results = {}

    print(f"\n{'='*80}")
    print(f"グラフファイル: {graph_file}")
    print(f"需要ファイル: {commodity_file}")
    print(f"{'='*80}\n")

    for solver_type in solver_types:
        print(f"ソルバー '{solver_type}' で計算中...", end='', flush=True)

        try:
            # ソルバーのインスタンスを作成
            solver = SolveExactSolution(
                solver_type=solver_type,
                comodity_file_name=commodity_file,
                graph_file_name=graph_file
            )

            # 最適化問題を解く
            start_time = time.time()
            solution_matrix, r_kakai, max_load_ratio, elapsed_time = solver.solve_exact_solution_to_env()
            total_time = time.time() - start_time

            # 結果を保存
            results[solver_type] = {
                'max_load_ratio': max_load_ratio,
                'elapsed_time': elapsed_time,
                'total_time': total_time,
                'status': 'success',
                'error': None
            }

            print(f" {Fore.GREEN}✓ 完了{Style.RESET_ALL} ({elapsed_time:.4f}秒)")

        except ImportError as e:
            # PySCIPOptがインストールされていない場合など
            results[solver_type] = {
                'max_load_ratio': None,
                'elapsed_time': None,
                'total_time': None,
                'status': 'error',
                'error': f'ImportError: {str(e)}'
            }
            print(f" {Fore.YELLOW}⚠ スキップ{Style.RESET_ALL} (依存パッケージなし)")

        except Exception as e:
            # その他のエラー
            results[solver_type] = {
                'max_load_ratio': None,
                'elapsed_time': None,
                'total_time': None,
                'status': 'error',
                'error': str(e)
            }
            print(f" {Fore.RED}✗ エラー{Style.RESET_ALL}: {str(e)}")

    return results

def print_results(results):
    """
    結果を整形して表示する

    Args:
        results: 各ソルバーの結果を含む辞書
    """
    print(f"\n{'='*80}")
    print("結果の比較")
    print(f"{'='*80}\n")

    # 成功したソルバーのみを取得
    successful_results = {k: v for k, v in results.items() if v['status'] == 'success'}

    if not successful_results:
        print(f"{Fore.RED}全てのソルバーでエラーが発生しました。{Style.RESET_ALL}")
        return

    # テーブルを表示（シンプルなフォーマット）
    print(f"{'ソルバー':<12} {'最大負荷率':<18} {'計算時間':<15} {'合計時間':<15} {'ステータス':<10}")
    print("-" * 80)

    for solver_type, result in results.items():
        if result['status'] == 'success':
            status_text = f"{Fore.GREEN}成功{Style.RESET_ALL}" if HAS_COLOR else "成功"
            print(f"{solver_type:<12} {result['max_load_ratio']:<18.10f} "
                  f"{result['elapsed_time']:<15.4f} {result['total_time']:<15.4f} {status_text}")
        else:
            status_text = f"{Fore.YELLOW}エラー{Style.RESET_ALL}" if HAS_COLOR else "エラー"
            print(f"{solver_type:<12} {'-':<18} {'-':<15} {'-':<15} {status_text}")

    # 最大負荷率の比較
    if len(successful_results) >= 2:
        print(f"\n{'='*80}")
        print("最大負荷率の差異チェック")
        print(f"{'='*80}\n")

        load_ratios = {k: v['max_load_ratio'] for k, v in successful_results.items()}
        reference_value = next(iter(load_ratios.values()))
        tolerance = 1e-6  # 許容誤差

        all_match = True
        for solver_type, load_ratio in load_ratios.items():
            diff = abs(load_ratio - reference_value)
            if diff > tolerance:
                all_match = False
                print(f"{Fore.RED}⚠ 警告: ソルバー '{solver_type}' の最大負荷率が他と異なります{Style.RESET_ALL}")
                print(f"   値: {load_ratio:.10f}, 差分: {diff:.2e}")

        if all_match:
            print(f"{Fore.GREEN}✓ 全てのソルバーで最大負荷率が一致しています{Style.RESET_ALL}")
            print(f"   最大負荷率: {reference_value:.10f}")

    # 計算時間の比較
    if len(successful_results) >= 2:
        print(f"\n{'='*80}")
        print("計算時間の比較")
        print(f"{'='*80}\n")

        elapsed_times = {k: v['elapsed_time'] for k, v in successful_results.items()}
        fastest_solver = min(elapsed_times, key=elapsed_times.get)
        slowest_solver = max(elapsed_times, key=elapsed_times.get)

        print(f"最速: {Fore.GREEN}{fastest_solver}{Style.RESET_ALL} ({elapsed_times[fastest_solver]:.4f}秒)")
        print(f"最遅: {Fore.RED}{slowest_solver}{Style.RESET_ALL} ({elapsed_times[slowest_solver]:.4f}秒)")

        if elapsed_times[slowest_solver] > 0:
            speedup = elapsed_times[slowest_solver] / elapsed_times[fastest_solver]
            print(f"速度比: {speedup:.2f}x")

    # エラーがあった場合の詳細を表示
    error_results = {k: v for k, v in results.items() if v['status'] == 'error'}
    if error_results:
        print(f"\n{'='*80}")
        print("エラー詳細")
        print(f"{'='*80}\n")

        for solver_type, result in error_results.items():
            print(f"{Fore.YELLOW}{solver_type}:{Style.RESET_ALL} {result['error']}")

def main():
    """
    メイン関数
    """
    # デフォルトのファイルパス
    default_graph_file = str(project_root / 'results' / 'graph.gml')
    default_commodity_file = str(project_root / 'results' / 'commodity_data.csv')

    # コマンドライン引数からファイルパスを取得
    if len(sys.argv) >= 3:
        graph_file = sys.argv[1]
        commodity_file = sys.argv[2]
    else:
        graph_file = default_graph_file
        commodity_file = default_commodity_file
        print(f"デフォルトファイルを使用します:")
        print(f"  グラフファイル: {graph_file}")
        print(f"  需要ファイル: {commodity_file}")

    # ファイルの存在確認
    if not os.path.exists(graph_file):
        print(f"{Fore.RED}エラー: グラフファイルが見つかりません: {graph_file}{Style.RESET_ALL}")
        sys.exit(1)

    if not os.path.exists(commodity_file):
        print(f"{Fore.RED}エラー: 需要ファイルが見つかりません: {commodity_file}{Style.RESET_ALL}")
        sys.exit(1)

    # ソルバーを比較
    results = compare_solvers(graph_file, commodity_file)

    # 結果を表示
    print_results(results)

if __name__ == '__main__':
    main()
