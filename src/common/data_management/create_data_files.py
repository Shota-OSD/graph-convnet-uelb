import os
import csv
import networkx as nx
from .data_maker import DataMaker
from src.common.solvers.exact_ilp import SolveExactSolution
from src.common.solvers.ksp_ilp import KspIlpSolver
from src.common.config.paths import get_mode_dir, get_graph_file, get_commodity_file, get_node_flow_file, get_ksp_ilp_solution_file, BUCKET_SIZE
import torch


def _count_completed_data(mode_dir, num_data):
    """完了済みデータの連続インデックス数を返す。

    exact_solution.csv の行数と、各インデックスの 3 ファイル
    (graph, commodity, node_flow) の存在を確認し、
    連続して揃っている最大インデックス+1 を返す。
    """
    exact_file = mode_dir / "exact_solution.csv"
    if not exact_file.exists():
        return 0

    # exact_solution.csv の行数を取得
    try:
        with open(exact_file, 'r') as f:
            csv_rows = list(csv.reader(f))
        num_csv_rows = len(csv_rows)
    except Exception:
        return 0

    if num_csv_rows == 0:
        return 0

    # 各インデックスの 3 ファイルが揃っているか確認
    completed = 0
    for i in range(min(num_csv_rows, num_data)):
        bucket = i - (i % BUCKET_SIZE)
        graph_path = mode_dir / "graph_file" / str(bucket) / f"graph_{i}.gml"
        commodity_path = mode_dir / "commodity_file" / str(bucket) / f"commodity_data_{i}.csv"
        node_flow_path = mode_dir / "node_flow_file" / str(bucket) / f"node_flow_{i}.csv"
        if graph_path.exists() and commodity_path.exists() and node_flow_path.exists():
            completed = i + 1
        else:
            break

    return completed


def _cleanup_incomplete(mode_dir, completed, num_data):
    """completed 以降の不完全データを削除し、exact_solution.csv を切り詰める。"""
    exact_file = mode_dir / "exact_solution.csv"

    # exact_solution.csv を completed 行に切り詰め
    if exact_file.exists():
        try:
            with open(exact_file, 'r') as f:
                rows = list(csv.reader(f))
            with open(exact_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(rows[:completed])
        except Exception:
            pass

    # completed 以降の個別ファイルを削除
    for i in range(completed, num_data):
        bucket = i - (i % BUCKET_SIZE)
        for subdir, prefix, suffix in [
            ("graph_file", "graph_", ".gml"),
            ("commodity_file", "commodity_data_", ".csv"),
            ("node_flow_file", "node_flow_", ".csv"),
        ]:
            path = mode_dir / subdir / str(bucket) / f"{prefix}{i}{suffix}"
            if path.exists():
                path.unlink()


def create_data_files(config, data_mode="test", num_samples=None, skip_exact=False):
    """
    指定されたモードのデータファイル（グラフ、品種、厳密解など）を生成し保存する関数。

    Args:
        config: 設定オブジェクト（`num_{data_mode}_data` や `solver_type` を含む必要あり）。
        data_mode (str): データモード ("train", "val", "test")。デフォルトは "test"。
        num_samples (int, optional): 生成するサンプル数を上書き（パイロットテスト用）。
        skip_exact (bool): Trueの場合、厳密解の計算をスキップ。CLIフラグまたはconfigで制御。
    """
    # CLI フラグ or config で skip_exact を決定
    skip_exact = skip_exact or getattr(config, 'skip_exact', False)
    num_data = getattr(config, f'num_{data_mode}_data')
    if num_samples is not None:
        num_data = min(num_data, num_samples)
    solver_type = config.solver_type
    solver_time_limit = getattr(config, 'solver_time_limit', 30)
    solver_ratio_gap = getattr(config, 'solver_ratio_gap', None)
    require_optimal = getattr(config, 'require_optimal', True)
    Maker = DataMaker(config)

    mode_dir = get_mode_dir(data_mode, config)
    mode_dir.mkdir(parents=True, exist_ok=True)

    exact_file_name = str(mode_dir / "exact_solution.csv")
    K = getattr(config, 'K', None)
    ksp_ilp_time_limit = getattr(config, 'ksp_ilp_time_limit', 300)
    ksp_ilp_ratio_gap = getattr(config, 'ksp_ilp_ratio_gap', None)
    ksp_ilp_solver = None
    ksp_ilp_file = None
    if K is not None:
        ksp_ilp_solver = KspIlpSolver(solver_name='CBC', time_limit=ksp_ilp_time_limit, ratio_gap=ksp_ilp_ratio_gap)
        ksp_ilp_file = get_ksp_ilp_solution_file(data_mode, config, K)
    infinit_loop_count = 0
    incorrect_value_count = 0
    non_optimal_count = 0

    # 再開ロジック: 完了済みデータ数を確認
    start_index = _count_completed_data(mode_dir, num_data)

    if start_index >= num_data:
        print(f"All {num_data} {data_mode} data already completed. Skipping.")
        return

    if start_index > 0:
        print(f"Resuming {data_mode} data generation from index {start_index}/{num_data}")
        _cleanup_incomplete(mode_dir, start_index, num_data)
    else:
        if os.path.exists(exact_file_name):
            os.remove(exact_file_name)

    for i in range(start_index, num_data):
        data = i
        if data % BUCKET_SIZE == 0:
            print(f"{data} data was created.")

        # ディレクトリ番号の定義
        file_number = data - (data % BUCKET_SIZE)

        # ディレクトリ作成
        directories = ["graph_file", "commodity_file", "node_flow_file"]
        #directories = ["graph_file", "commodity_file", "edge_file", "node_flow_file", "edge_flow_file"]
        for directory in directories:
            (mode_dir / directory / str(file_number)).mkdir(parents=True, exist_ok=True)

        # ファイル名の定義
        graph_file_name = str(mode_dir / "graph_file" / str(file_number) / f"graph_{data}.gml")
        commodity_file_name = str(mode_dir / "commodity_file" / str(file_number) / f"commodity_data_{data}.csv")
        node_flow_file_name = str(mode_dir / "node_flow_file" / str(file_number) / f"node_flow_{data}.csv")
        #edge_file_name = str(mode_dir / "edge_file" / str(file_number) / f"edge_numbering_{data}.csv")
        #edge_flow_file_name = str(mode_dir / "edge_flow_file" / str(file_number) / f"edge_flow_{data}.csv")

        # グラフ作成
        G = Maker.create_graph()

        # 品種作成
        commodity_list = Maker.generate_commodity()

        # グラフ保存
        nx.write_gml(G, graph_file_name)

        # 品種保存
        with open(commodity_file_name, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(commodity_list)

        if skip_exact:
            # KSP-ILP 逐次計算
            if ksp_ilp_solver is not None:
                result = ksp_ilp_solver.solve_from_files(data, data_mode, config, K)
                with open(ksp_ilp_file, 'a', newline='') as f:
                    csv.writer(f).writerow([result.alpha, result.elapsed_time, result.mip_gap])
                if data % BUCKET_SIZE == 0:
                    gap_str = f", Gap={result.mip_gap:.4f}" if result.mip_gap is not None else ""
                    print(f"  KSP-ILP: {data}/{num_data} (MLU={result.alpha:.6f}{gap_str})")
            continue

        # 作成したデータが適切でない場合のやり直し
        while True:
            # 厳密解の計算
            try:
                E = SolveExactSolution(solver_type, commodity_file_name, graph_file_name)
                flow_var_kakai, edge_list, objective_value, elapsed_time, is_optimal, exact_mip_gap = E.solve_exact_solution_to_env(time_limit=solver_time_limit, ratio_gap=solver_ratio_gap)
                node_flow_matrix, edge_flow_matrix, infinit_loop = E.generate_flow_matrices(flow_var_kakai)
            except Exception as e:
                print(f"Error in exact solution calculation for data {data}: {e}")
                infinit_loop = True
                is_optimal = False
                objective_value = 1.0  # エラーの場合は1.0として扱う
                exact_mip_gap = None

            # 厳密解が1以上、最適性未証明、またはフローが正しく導けなかった場合のやり直し
            if infinit_loop:
                infinit_loop_count += 1
            elif objective_value >= 1.0:
                incorrect_value_count += 1
            elif require_optimal and not is_optimal:
                non_optimal_count += 1
            else:
                break

            # やり直し: グラフと品種を再生成
            G = Maker.create_graph()
            commodity_list = Maker.generate_commodity()
            nx.write_gml(G, graph_file_name)
            with open(commodity_file_name, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(commodity_list)

        # 厳密解保存
        with open(exact_file_name, 'a', newline='') as f:
            out = csv.writer(f)
            out.writerow([objective_value, elapsed_time, exact_mip_gap])

        # 厳密解ノードフロー保存
        with open(node_flow_file_name, 'w', newline='') as file:
            writer = csv.writer(file)
            for row in node_flow_matrix:
                writer.writerow(row)

        # KSP-ILP 逐次計算
        if ksp_ilp_solver is not None:
            result = ksp_ilp_solver.solve_from_files(data, data_mode, config, K)
            with open(ksp_ilp_file, 'a', newline='') as f:
                csv.writer(f).writerow([result.alpha, result.elapsed_time, result.mip_gap])
            if data % BUCKET_SIZE == 0:
                gap_str = f", Gap={result.mip_gap:.4f}" if result.mip_gap is not None else ""
                print(f"  KSP-ILP: {data}/{num_data} (MLU={result.alpha:.6f}{gap_str})")

        """必要ないので一旦スキップ
        # 厳密解エッジフロー保存
        with open(edge_flow_file_name, 'w', newline='') as file:
            writer = csv.writer(file)
            for row in edge_flow_matrix:
                writer.writerow(row)
        
        # エッジ保存
        with open(edge_file_name, 'w', newline='') as file:
            writer = csv.writer(file)
            for item in edge_list:
                writer.writerow([item[0], item[1][0], item[1][1]])
        """
    
    print(f"Data generation completed: {num_data} data created.")
    if not skip_exact:
        print(f"Infinit loops: {infinit_loop_count}, Incorrect values: {incorrect_value_count}, Non-optimal (discarded): {non_optimal_count} (time_limit={solver_time_limit}s)")


def compute_ksp_ilp_solutions(config, data_mode, num_data, K, time_limit=30):
    """既存データに対して KSP-ILP を事前計算し CSV に保存する.

    exact_solution.csv と同じ形式 [load_factor, elapsed_time] で
    ksp_ilp_K{K}_solution.csv に書き出す。
    既に完了済みの行はスキップする。

    Args:
        config: 設定オブジェクト
        data_mode: 'train', 'val', 'test'
        num_data: データ数
        K: KSP 候補パス数
        time_limit: ILP ソルバーの制限時間 [秒]
    """
    ksp_ilp_file = get_ksp_ilp_solution_file(data_mode, config, K)

    # 完了済み行数を確認（再開サポート）
    start_index = 0
    if ksp_ilp_file.exists():
        try:
            with open(ksp_ilp_file, 'r') as f:
                start_index = sum(1 for _ in csv.reader(f))
        except Exception:
            start_index = 0

    if start_index >= num_data:
        print(f"KSP-ILP (K={K}) for {data_mode}: all {num_data} already computed. Skipping.")
        return

    if start_index > 0:
        print(f"KSP-ILP (K={K}) for {data_mode}: resuming from index {start_index}/{num_data}")

    print(f"Computing KSP-ILP solutions (K={K}) for {data_mode} data...")
    solver = KspIlpSolver(solver_name='CBC', time_limit=time_limit)

    for i in range(start_index, num_data):
        result = solver.solve_from_files(i, data_mode, config, K)
        with open(ksp_ilp_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([result.alpha, result.elapsed_time, result.mip_gap])
        if i % BUCKET_SIZE == 0:
            gap_str = f", Gap={result.mip_gap:.4f}" if result.mip_gap is not None else ""
            print(f"  KSP-ILP: {i}/{num_data} computed (MLU={result.alpha:.6f}{gap_str})")

    print(f"KSP-ILP (K={K}) computation completed: {num_data} solutions saved to {ksp_ilp_file}")