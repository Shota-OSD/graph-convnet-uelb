"""KSP-ILP Solver for UELB.

KSP 候補パス内で ILP を厳密に解く。既存の KSP-Iterative が greedy で
パスを選ぶのに対し、KSP-ILP は ILP で最適なパス割当を求める。

定式化:
    min α  (= MLU)
    s.t.
      Σ_p x_{k,p} = 1            ∀k        (各コモディティは1本のパスを選択)
      Σ_{k,p: e∈p} d_k · x_{k,p} ≤ α · c_e  ∀e  (容量制約)
      x_{k,p} ∈ {0, 1}
"""

import csv
import os
import re
import tempfile
import time
from dataclasses import dataclass
from typing import Any, List, Optional

import networkx as nx
import pulp

from src.common.config.paths import get_graph_file, get_commodity_file
from src.common.graph.k_shortest_path import KShortestPathFinder


def parse_cbc_log_for_bound(log_path: str) -> Optional[float]:
    """CBC ログファイルから best bound (Best possible) をパースする."""
    try:
        with open(log_path) as f:
            log = f.read()
        # "Best possible: 0.43000000" のパターン
        m = re.search(r'Best possible:\s+([-\d.eE+]+)', log)
        if m:
            return float(m.group(1))
    except Exception:
        pass
    return None


@dataclass
class KspIlpResult:
    """KSP-ILP ソルバーの結果."""
    status: str           # 'Optimal', 'Infeasible', etc.
    alpha: float          # 最適 MLU 値
    grouping: List[List[int]]  # 選択されたパス (コモディティ毎のノードリスト)
    elapsed_time: float
    is_optimal: bool
    mip_gap: Optional[float] = None  # MIP Gap (0.0 = 最適解証明済み)


class KspIlpSolver:
    """KSP 候補パス内で ILP を解いて最適パス割当を求めるソルバー."""

    def __init__(self, solver_name: str = 'CBC', time_limit: int = 300, ratio_gap: Optional[float] = None):
        self.solver_name = solver_name
        self.time_limit = time_limit
        self.ratio_gap = ratio_gap  # 例: 0.05 = 5% Gap で停止

    def _resolve_solver(self, log_path: Optional[str] = None):
        """PuLP ソルバーインスタンスを返す (HiGHS → CBC フォールバック)."""
        # CBC は ratioGap を options リストで渡す
        cbc_options = [f'ratioGap {self.ratio_gap}'] if self.ratio_gap is not None else []
        if self.solver_name == 'HiGHS':
            try:
                solver = pulp.HiGHS_CMD(msg=False, timeLimit=self.time_limit, logPath=log_path)
                if solver.available():
                    return solver
            except Exception:
                pass
            print("Warning: HiGHS not available, falling back to CBC")
        return pulp.PULP_CBC_CMD(msg=False, timeLimit=self.time_limit, logPath=log_path, options=cbc_options)

    def solve(
        self,
        G: nx.DiGraph,
        commodity_list: List[List[int]],
        allcommodity_ksps: List[List[List[int]]],
    ) -> KspIlpResult:
        """KSP-ILP を解く.

        Args:
            G: NetworkX 有向グラフ (capacity 属性付き)
            commodity_list: [[source, target, demand], ...]
            allcommodity_ksps: 各コモディティの K 最短パスリスト

        Returns:
            KspIlpResult
        """
        num_commodities = len(commodity_list)
        edges = list(G.edges())
        capacity = nx.get_edge_attributes(G, 'capacity')

        # --- 前処理: 各パスが使うエッジ集合 ---
        # path_edges[k][p] = set of (u, v)
        path_edges = []
        for k in range(num_commodities):
            k_edges = []
            for path in allcommodity_ksps[k]:
                edge_set = set()
                for i in range(len(path) - 1):
                    edge_set.add((path[i], path[i + 1]))
                k_edges.append(edge_set)
            path_edges.append(k_edges)

        # --- PuLP 問題構築 ---
        prob = pulp.LpProblem('KSP_ILP_UELB', pulp.LpMinimize)

        # 変数
        alpha = pulp.LpVariable('alpha', 0, 1, cat='Continuous')
        x = {}
        for k in range(num_commodities):
            for p in range(len(allcommodity_ksps[k])):
                x[k, p] = pulp.LpVariable(f'x_{k}_{p}', cat=pulp.LpBinary)

        # 目的関数
        prob += alpha

        # 制約1: 各コモディティは1本のパスを選択
        for k in range(num_commodities):
            prob += (
                pulp.lpSum(x[k, p] for p in range(len(allcommodity_ksps[k]))) == 1,
                f"one_path_{k}",
            )

        # 制約2: 容量制約 (各エッジ)
        for e_idx, (u, v) in enumerate(edges):
            edge_tuple = (u, v)
            c_e = capacity[edge_tuple]
            # このエッジを使う (k, p) の組を集める
            terms = []
            for k in range(num_commodities):
                d_k = commodity_list[k][2]
                for p in range(len(allcommodity_ksps[k])):
                    if edge_tuple in path_edges[k][p]:
                        terms.append(d_k * x[k, p])
            if terms:
                prob += (
                    pulp.lpSum(terms) <= alpha * c_e,
                    f"capacity_{e_idx}",
                )

        # --- ソルバー実行 ---
        log_path = tempfile.mktemp(suffix='.log')
        solver = self._resolve_solver(log_path=log_path)
        start = time.time()
        status = prob.solve(solver)
        elapsed_time = time.time() - start

        status_str = pulp.LpStatus[status]
        is_optimal = (status_str == 'Optimal') and (elapsed_time < self.time_limit - 1.0)

        # --- MIP Gap 計算 ---
        alpha_val = pulp.value(alpha) if status_str == 'Optimal' else float('inf')
        mip_gap = None
        if is_optimal:
            mip_gap = 0.0
        elif alpha_val != float('inf') and alpha_val > 0:
            best_bound = parse_cbc_log_for_bound(log_path)
            if best_bound is not None:
                mip_gap = (alpha_val - best_bound) / alpha_val
        try:
            os.unlink(log_path)
        except OSError:
            pass
        grouping = []
        for k in range(num_commodities):
            selected = False
            for p in range(len(allcommodity_ksps[k])):
                val = pulp.value(x[k, p])
                if val is not None and val > 0.5:
                    grouping.append(allcommodity_ksps[k][p])
                    selected = True
                    break
            if not selected:
                # フォールバック: 最短パスを選択
                grouping.append(allcommodity_ksps[k][0] if allcommodity_ksps[k] else [])

        return KspIlpResult(
            status=status_str,
            alpha=alpha_val,
            grouping=grouping,
            elapsed_time=elapsed_time,
            is_optimal=is_optimal,
            mip_gap=mip_gap,
        )

    def solve_from_files(
        self,
        data_idx: int,
        mode: str,
        config: Any,
        K: int = 10,
    ) -> KspIlpResult:
        """ファイルからデータを読み込んで KSP-ILP を解く便利メソッド.

        Args:
            data_idx: データインデックス
            mode: 'train', 'val', 'test'
            config: ConfigManager 等の設定オブジェクト
            K: KSP の候補パス数

        Returns:
            KspIlpResult
        """
        # グラフ読込
        graph_file = str(get_graph_file(mode, data_idx, config))
        G = nx.read_gml(graph_file, destringizer=int)

        # コモディティ読込
        commodity_file = str(get_commodity_file(mode, data_idx, config))
        commodity_list = []
        with open(commodity_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 3:
                    commodity_list.append([int(row[0]), int(row[1]), int(row[2])])

        # KSP 候補パス生成
        ksp_finder = KShortestPathFinder()
        allcommodity_ksps = ksp_finder.search_k_shortest_paths(G, commodity_list, K)

        return self.solve(G, commodity_list, allcommodity_ksps)
