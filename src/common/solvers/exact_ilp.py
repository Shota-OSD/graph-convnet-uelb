"""Exact ILP solver for UELB.

変数 x_{l,e} (フロー×エッジ) + フロー保存則制約で厳密解を求める。
対応ソルバー: PuLP+CBC, MIP+CBC
"""

from mip import Model, minimize, xsum, BINARY
import time
import csv
import networkx as nx
import numpy as np
import pulp
import torch

from src.common.graph.flow import Flow


class SolveExactSolution:
    def __init__(self, solver_type, comodity_file_name, graph_file_name):
        self.solver_type = solver_type
        self.comodity_file_name = comodity_file_name
        self.G = nx.read_gml(graph_file_name, destringizer=int)
        self.G.all_flows = list()

        with open(self.comodity_file_name, newline='') as f:
            self.commodity = csv.reader(f)
            self.commodity = [row for row in self.commodity]
            self.commodity = [[int(item) for item in row] for row in self.commodity]

        self.r_kakai = list(enumerate(self.G.edges()))
        self.commodity_count = 0
        self.tuples = []
        self.capacity = nx.get_edge_attributes(self.G, 'capacity')

        while len(self.tuples) < len(self.commodity):
            s = self.commodity[self.commodity_count][0]
            t = self.commodity[self.commodity_count][1]
            demand = self.commodity[self.commodity_count][2]
            self.tuples.append((s, t))
            f = Flow(self.G, self.commodity_count, s, t, demand)
            self.G.all_flows.append(f)
            self.commodity_count += 1

    def solve_exact_solution_to_env(self, time_limit=30):
        if self.solver_type == 'mip':
            return self._solve_mip(time_limit)
        if self.solver_type == 'pulp':
            return self._solve_pulp(time_limit)
        raise ValueError(f"Unknown solver_type: {self.solver_type}")

    def _solve_mip(self, time_limit):
        """MIP+CBC ソルバー."""
        UELB_kakai = Model('UELB_kakai')

        L_kakai = UELB_kakai.add_var('L_kakai', lb=0, ub=1)
        flow_var_kakai = []
        for l in range(len(self.G.all_flows)):
            x_kakai = [UELB_kakai.add_var('x{}_{}'.format(l, m), var_type=BINARY) for m, (i, j) in self.r_kakai]
            flow_var_kakai.append(x_kakai)

        UELB_kakai.objective = minimize(L_kakai)
        UELB_kakai += (-L_kakai) >= -1

        for e in range(len(self.G.edges())):
            UELB_kakai += 0 <= L_kakai - ((xsum([(flow_var_kakai[l.get_id()][e]) * (l.get_demand()) for l in self.G.all_flows])) / self.capacity[self.r_kakai[e][1]])

        for l in self.G.all_flows:
            UELB_kakai += xsum([flow_var_kakai[l.get_id()][e] * (l.get_demand()) for e in range(len(self.G.edges())) if self.r_kakai[e][1][0] == l.get_s()]) == l.get_demand()
            UELB_kakai += xsum([flow_var_kakai[l.get_id()][e] * (l.get_demand()) for e in range(len(self.G.edges())) if self.r_kakai[e][1][1] == l.get_s()]) == 0
            UELB_kakai += xsum([flow_var_kakai[l.get_id()][e] * (l.get_demand()) for e in range(len(self.G.edges())) if self.r_kakai[e][1][0] == l.get_t()]) == 0
            UELB_kakai += xsum([flow_var_kakai[l.get_id()][e] * (l.get_demand()) for e in range(len(self.G.edges())) if self.r_kakai[e][1][1] == l.get_t()]) == l.get_demand()

        for l in self.G.all_flows:
            for v in self.G.nodes():
                if v != l.get_s() and v != l.get_t():
                    UELB_kakai += xsum([(flow_var_kakai[l.get_id()][e]) * (l.get_demand()) for e in range(len(self.G.edges())) if self.r_kakai[e][1][0] == v]) \
                        == xsum([(flow_var_kakai[l.get_id()][e]) * (l.get_demand()) for e in range(len(self.G.edges())) if self.r_kakai[e][1][1] == v])

        start = time.time()
        UELB_kakai.optimize()
        elapsed_time = time.time() - start

        return UELB_kakai.objective_value, elapsed_time

    def _solve_pulp(self, time_limit):
        """PuLP+CBC ソルバー."""
        UELB_problem = pulp.LpProblem('UELB', pulp.LpMinimize)

        L = pulp.LpVariable('L', 0, 1, 'Continuous')
        flow_var_kakai = []
        for l in range(len(self.G.all_flows)):
            e_01 = [pulp.LpVariable('x{}_{}'.format(l, m), cat=pulp.LpBinary) for m, (i, j) in self.r_kakai]
            flow_var_kakai.append(e_01)

        UELB_problem += (L, "Objective")
        UELB_problem += (-L) >= -1

        for e in range(len(self.G.edges())):
            UELB_problem += 0 <= L - ((sum([(flow_var_kakai[l.get_id()][e]) * (l.get_demand()) for l in self.G.all_flows])) / self.capacity[self.r_kakai[e][1]])

        for l in self.G.all_flows:
            UELB_problem += sum([flow_var_kakai[l.get_id()][e] * (l.get_demand()) for e in range(len(self.G.edges())) if self.r_kakai[e][1][0] == l.get_s()]) == l.get_demand()
            UELB_problem += sum([flow_var_kakai[l.get_id()][e] * (l.get_demand()) for e in range(len(self.G.edges())) if self.r_kakai[e][1][1] == l.get_s()]) == 0
            UELB_problem += sum([flow_var_kakai[l.get_id()][e] * (l.get_demand()) for e in range(len(self.G.edges())) if self.r_kakai[e][1][0] == l.get_t()]) == 0
            UELB_problem += sum([flow_var_kakai[l.get_id()][e] * (l.get_demand()) for e in range(len(self.G.edges())) if self.r_kakai[e][1][1] == l.get_t()]) == l.get_demand()

        for l in self.G.all_flows:
            for v in self.G.nodes():
                if v != l.get_s() and v != l.get_t():
                    UELB_problem += sum([(flow_var_kakai[l.get_id()][e]) * (l.get_demand()) for e in range(len(self.G.edges())) if self.r_kakai[e][1][0] == v]) \
                        == sum([(flow_var_kakai[l.get_id()][e]) * (l.get_demand()) for e in range(len(self.G.edges())) if self.r_kakai[e][1][1] == v])

        start = time.time()
        status = UELB_problem.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit))
        elapsed_time = time.time() - start

        is_optimal = (pulp.LpStatus[status] == 'Optimal') and (elapsed_time < time_limit - 1.0)
        self.flow_var_kakai = flow_var_kakai
        return flow_var_kakai, self.r_kakai, L.value(), elapsed_time, is_optimal

    def generate_edges_target(self):
        num_flows = len(self.G.all_flows)
        num_nodes = len(self.G.nodes())
        num_edges = len(self.G.edges())
        exact_edges_matrix = torch.zeros((num_nodes, num_nodes, num_flows), dtype=int)

        for l in range(num_flows):
            for e in range(num_edges):
                if pulp.value(self.flow_var_kakai[l][e]) == 1:
                    node_from = self.r_kakai[e][1][0]
                    node_to = self.r_kakai[e][1][1]
                    exact_edges_matrix[node_from][node_to][l] = 1

        return exact_edges_matrix

    def generate_flow_matrices(self, flow_var_kakai):
        """フローがどのノードやエッジをどの順序で通過するかを計算する.

        Args:
            flow_var_kakai: 各フローがエッジを通過するかどうかを示すバイナリ変数のリスト

        Returns:
            node_flow_matrix: 各フローのノード通過順序を示す行列
            edge_flow_matrix: 各フローのエッジ通過順序を示す行列
            infinit_loop: 探索が正常に終了したかどうかを示すフラグ
        """
        num_flows = len(self.G.all_flows)
        num_nodes = len(self.G.nodes())
        num_edges = len(self.G.edges())
        node_flow_matrix = np.zeros((num_flows, num_nodes), dtype=int)
        edge_flow_matrix = np.zeros((num_flows, num_edges), dtype=int)
        infinit_loop = False

        for l in range(num_flows):
            current_node_order = 1
            current_edge_order = 1
            current_node = self.G.all_flows[l].get_s()
            node_flow_matrix[l][current_node] = current_node_order
            current_node_order += 1

            max_steps = num_edges * 3
            steps = 0

            while current_node != self.G.all_flows[l].get_t():
                steps += 1
                if steps > max_steps:
                    infinit_loop = True
                    return node_flow_matrix, edge_flow_matrix, infinit_loop

                for e in range(num_edges):
                    node_from = self.r_kakai[e][1][0]
                    node_to = self.r_kakai[e][1][1]

                    if node_from == current_node and pulp.value(flow_var_kakai[l][e]) == 1:
                        edge_flow_matrix[l][e] = current_edge_order
                        current_edge_order += 1

                        if node_flow_matrix[l][node_to] == 0:
                            node_flow_matrix[l][node_to] = current_node_order
                            current_node_order += 1

                        current_node = node_to
                        break

        infinit_loop = False
        return node_flow_matrix, edge_flow_matrix, infinit_loop
