"""KSP-LP (PILP: Path-based Integer Linear Programming) ソルバー.

KSP 候補経路内での最適 MLU を求める。
全パス ILP と異なり変数が K×C と小規模なため、大規模グラフでも高速に解ける。
"""

import time
import pulp
import networkx as nx
from typing import List

from src.common.graph.k_shortest_path import KShortestPathFinder


class SolveKspLpSolution:
    """KSP候補経路内でのPILP（Path-based ILP）を解く"""

    def __init__(self, graph: nx.Graph, commodity_list: List[List[int]],
                 K: int = 10, solver_time_limit=None):
        """
        Args:
            graph: NetworkXグラフ（edge属性に'capacity'）
            commodity_list: [[source, sink, demand], ...]
            K: 候補経路数（デフォルト10）
            solver_time_limit: ソルバー制限時間（秒）。None の場合は制限なし
        """
        self.graph = graph
        self.commodity_list = commodity_list
        self.K = K
        self.solver_time_limit = solver_time_limit

    def solve(self) -> dict:
        """KSP-LP を解く.

        Returns:
            dict: {
                'objective_value': float,   # KSP-LP 最適 MLU
                'elapsed_time': float,      # PILP ソルバー実行時間（秒）
                'is_optimal': bool,         # 最適性が証明されたか
                'selected_paths': list,     # 各品種が選択したパスのインデックス
                'ksp_elapsed_time': float,  # KSP 候補生成にかかった時間
            }
        """
        # 1. KSP 候補経路を生成
        ksp_finder = KShortestPathFinder()
        ksp_start = time.time()
        all_ksp = ksp_finder.search_k_shortest_paths(
            self.graph, self.commodity_list, self.K
        )
        ksp_elapsed = time.time() - ksp_start

        # 2. PILP を定式化
        problem = pulp.LpProblem('KSP_LP', pulp.LpMinimize)

        # MLU 変数
        alpha = pulp.LpVariable('alpha', lowBound=0, upBound=1, cat='Continuous')
        problem += (alpha, "Objective")

        # x_{k,p} 変数: 品種 k が候補パス p を使うか (0-1)
        num_commodities = len(self.commodity_list)
        x = {}
        for k in range(num_commodities):
            num_paths = len(all_ksp[k])
            for p in range(num_paths):
                x[k, p] = pulp.LpVariable(f'x_{k}_{p}', cat=pulp.LpBinary)

        # 制約1: 各品種は候補から1本選択
        for k in range(num_commodities):
            num_paths = len(all_ksp[k])
            problem += (
                pulp.lpSum(x[k, p] for p in range(num_paths)) == 1,
                f"select_one_{k}"
            )

        # エッジ → 容量のマッピングを構築
        capacity = nx.get_edge_attributes(self.graph, 'capacity')

        # 制約2: リンク容量制約
        # 各エッジについて、そのエッジを通るパスの需要合計 <= alpha * capacity
        # まず各エッジを使う (k, p) の組を事前計算
        edge_usage = {}  # edge -> [(k, p, demand), ...]
        for k in range(num_commodities):
            demand = self.commodity_list[k][2]
            for p, path in enumerate(all_ksp[k]):
                for i in range(len(path) - 1):
                    edge = (path[i], path[i + 1])
                    if edge not in edge_usage:
                        edge_usage[edge] = []
                    edge_usage[edge].append((k, p, demand))

        for edge, usages in edge_usage.items():
            cap = capacity.get(edge)
            if cap is None:
                continue
            problem += (
                pulp.lpSum(d * x[k, p] for k, p, d in usages) <= alpha * cap,
                f"cap_{edge[0]}_{edge[1]}"
            )

        # 3. 求解
        solver_kwargs = {'msg': False}
        if self.solver_time_limit is not None:
            solver_kwargs['timeLimit'] = self.solver_time_limit

        solve_start = time.time()
        status = problem.solve(pulp.PULP_CBC_CMD(**solver_kwargs))
        solve_elapsed = time.time() - solve_start

        is_optimal = pulp.LpStatus[status] == 'Optimal'
        if self.solver_time_limit is not None:
            is_optimal = is_optimal and (solve_elapsed < self.solver_time_limit - 1.0)

        objective_value = alpha.value() if alpha.value() is not None else 1.0

        # 選択されたパスのインデックスを取得
        selected_paths = []
        for k in range(num_commodities):
            selected = 0
            for p in range(len(all_ksp[k])):
                if pulp.value(x[k, p]) == 1:
                    selected = p
                    break
            selected_paths.append(selected)

        return {
            'objective_value': objective_value,
            'elapsed_time': solve_elapsed,
            'is_optimal': is_optimal,
            'selected_paths': selected_paths,
            'ksp_elapsed_time': ksp_elapsed,
        }
