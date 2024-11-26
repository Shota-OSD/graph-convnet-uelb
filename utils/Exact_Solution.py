#LP-ksps_envで使う厳密解を求めるためのプログラムファイル
from mip import *
import time
import csv
import networkx as nx
import matplotlib.pyplot as plt
from utils.flow import Flow
import pulp
import torch
# from pyscipopt import Model as SCIPModel, quicksum

class SolveExactSolution():
    def __init__(self, solver_type, comodity_file_name, graph_file_name):
        """
        ・graph.gml内
            node [
                id 0
                label "0"
            ] (id==label)
            edge [
                source 0
                target 1
                capacity 3334
            ] (max=9538, min=1624)
        """

        self.solver_type = solver_type
        self.comodity_file_name = comodity_file_name
        self.G = nx.read_gml(graph_file_name,destringizer=int)
        self.G.all_flows = list()

        """   
        ・commodity_data.csv内
            source, sink, demand 
            39,29,429
            5,28,259

        ・self.commodity = [
            [source, sink, demand]
            [39, 29, 429], 
            [5, 28, 259]
        ]
        """

        with open(self.comodity_file_name,newline='') as f: # 品種の読み込み
            self.commodity=csv.reader(f)
            self.commodity=[row for row in self.commodity]
            self.commodity=[[int(item) for item in row]for row in self.commodity]

        """
        self.r_kakai = [
            (id, (source, sink))
            (0, (39, 29)),
            (1, (5, 28)),
            (2, (0, 39))
        ]

        self.capacity =
        {
            (source, sink): capacity
            (39, 29): 100,
            (5, 28): 150,
            (0, 39): 200
        }
        """
        self.r_kakai = list(enumerate(self.G.edges()))
        self.commodity_count = 0
        self.tuples = []
        self.capacity = nx.get_edge_attributes(self.G, 'capacity')

        while(len(self.tuples)<len(self.commodity)): 
            s = self.commodity[self.commodity_count][0] # source
            t = self.commodity[self.commodity_count][1] # sink
            demand = self.commodity[self.commodity_count][2] # demand
            self.tuples.append((s,t))
            f = Flow(self.G,self.commodity_count,s,t,demand) #　定義されたクラス
            self.G.all_flows.append(f) # all_flowsはリスト
            self.commodity_count +=1
        
    def solve_exact_solution_to_env(self):
        if (self.solver_type == 'mip'): # mip+CBC
            # 問題の定義(全体の下界)
            UELB_kakai = Model('UELB_kakai') # モデルの名前

            L_kakai = UELB_kakai.add_var('L_kakai',lb = 0, ub = 1)
            flow_var_kakai = []
            for l in range(len(self.G.all_flows)):
                x_kakai = [UELB_kakai.add_var('x{}_{}'.format(l,m), var_type = BINARY) for m, (i,j) in self.r_kakai]#enumerate関数のmとエッジ(i,j)
                flow_var_kakai.append(x_kakai) #品種エル(l)に対して全ての辺のIDを表す番号mがついている。中身は0-1

            UELB_kakai.objective = minimize(L_kakai) # 目的関数

            UELB_kakai += (-L_kakai) >= -1 # 負荷率1以下
            
            # print("容量制限")
            for e in range(len(self.G.edges())): #容量制限
                UELB_kakai += 0 <= L_kakai - ((xsum([(flow_var_kakai[l.get_id()][e])*(l.get_demand()) for l in self.G.all_flows])) / self.capacity[self.r_kakai[e][1]])
            # print("フロー保存則1")
            for l in self.G.all_flows: #フロー保存則
                UELB_kakai += xsum([flow_var_kakai[l.get_id()][e]*(l.get_demand()) for e in range(len(self.G.edges())) if self.r_kakai[e][1][0] == l.get_s()]) == l.get_demand()
                UELB_kakai += xsum([flow_var_kakai[l.get_id()][e]*(l.get_demand()) for e in range(len(self.G.edges())) if self.r_kakai[e][1][1] == l.get_s()]) == 0
                UELB_kakai += xsum([flow_var_kakai[l.get_id()][e]*(l.get_demand()) for e in range(len(self.G.edges())) if self.r_kakai[e][1][0] == l.get_t()]) == 0
                UELB_kakai += xsum([flow_var_kakai[l.get_id()][e]*(l.get_demand()) for e in range(len(self.G.edges())) if self.r_kakai[e][1][1] == l.get_t()]) == l.get_demand()
            # print("フロー保存則2")
            for l in self.G.all_flows: #フロー保存則
                for v in self.G.nodes():
                    if(v != l.get_s() and v != l.get_t()):
                        UELB_kakai += xsum([(flow_var_kakai[l.get_id()][e])*(l.get_demand()) for e in range(len(self.G.edges())) if self.r_kakai[e][1][0] == v])\
                        ==xsum([(flow_var_kakai[l.get_id()][e])*(l.get_demand()) for e in range(len(self.G.edges())) if self.r_kakai[e][1][1] == v])

            #線形計画問題を解く
            start = time.time()
            UELB_kakai.optimize()
            elapsed_time = time.time()-start

            with open(self.exact_file_name, 'a', newline='') as f:
                out = csv.writer(f)
                out.writerow([UELB_kakai.objective_value, elapsed_time]) 
            return UELB_kakai.objective_value,elapsed_time
        
        if (self.solver_type == 'pulp'): # pulp+CBC
            UELB_problem = pulp.LpProblem('UELB', pulp.LpMinimize) # モデルの名前
            """
            ・LpProblem: 線形計画問題を定義するための pulp のクラス。
            ・'UELB' : 問題の名前で、自由に決められる。
            ・pulp.LpMinimize : 問題が「最小化問題」であることを指定。負荷率 L を最小化する目的を表す。

            ・self.r_kakai = [(0, ('A', 'B')), (1, ('B', 'C')), (2, ('C', 'D'))]

            ・フロー1(l=0):
                x0_0: エッジ A-B を通るかどうか(0 または 1)
                x0_1: エッジ B-C を通るかどうか(0 または 1)
                x0_2: エッジ C-D を通るかどうか(0 または 1)
            ・フロー2(l=1):
                x1_0: エッジ A-B を通るかどうか(0 または 1)
                x1_1: エッジ B-C を通るかどうか(0 または 1)
                x1_2: エッジ C-D を通るかどうか(0 または 1)

            上記の変数を格納 # 0/1のバイナリ変数は本問題の条件で決定
            flow_var_kakai = [
                [x0_0, x0_1, x0_2],  # フロー1に対応する0-1変数
                [x1_0, x1_1, x1_2]   # フロー2に対応する0-1変数
            ]
            """
            L = pulp.LpVariable('L', 0, 1, 'Continuous') # 最大負荷率　Continuous：連続値 Integer:整数値 Binary:２値変数
            flow_var_kakai = []
            for l in range(len(self.G.all_flows)): # 0,1変数
                e_01 = [pulp.LpVariable('x{}_{}'.format(l,m), cat=pulp.LpBinary) for m, (i,j) in self.r_kakai]#enumerate関数のmとエッジ(i,j)
                flow_var_kakai.append(e_01) #品種エル(l)に対して全ての辺のIDを表す番号mがついている。中身は0-1

            UELB_problem += ( L , "Objective" ) # 目的関数値　「負荷率 L を最小化する」という目標を示す
            
            UELB_problem += (-L) >= -1 # 負荷率1以下

            # print("容量制限")
            for e in range(len(self.G.edges())): # 容量制限
                UELB_problem += 0 <= L - ((sum([(flow_var_kakai[l.get_id()][e])*(l.get_demand()) for l in self.G.all_flows])) / self.capacity[self.r_kakai[e][1]])

            # print("フロー保存則1")
            for l in self.G.all_flows: #フロー保存則
                UELB_problem += sum([flow_var_kakai[l.get_id()][e]*(l.get_demand()) for e in range(len(self.G.edges())) if self.r_kakai[e][1][0] == l.get_s()]) == l.get_demand()
                UELB_problem += sum([flow_var_kakai[l.get_id()][e]*(l.get_demand()) for e in range(len(self.G.edges())) if self.r_kakai[e][1][1] == l.get_s()]) == 0
                UELB_problem += sum([flow_var_kakai[l.get_id()][e]*(l.get_demand()) for e in range(len(self.G.edges())) if self.r_kakai[e][1][0] == l.get_t()]) == 0
                UELB_problem += sum([flow_var_kakai[l.get_id()][e]*(l.get_demand()) for e in range(len(self.G.edges())) if self.r_kakai[e][1][1] == l.get_t()]) == l.get_demand()
            
            # print("フロー保存則2")
            for l in self.G.all_flows: #フロー保存則
                for v in self.G.nodes():
                    if(v != l.get_s() and v != l.get_t()):
                        UELB_problem += sum([(flow_var_kakai[l.get_id()][e])*(l.get_demand()) for e in range(len(self.G.edges())) if self.r_kakai[e][1][0] == v])\
                        ==sum([(flow_var_kakai[l.get_id()][e])*(l.get_demand()) for e in range(len(self.G.edges())) if self.r_kakai[e][1][1] == v])

            start = time.time()
            status = UELB_problem.solve(pulp.PULP_CBC_CMD(msg=False)) # 線形計画問題を解く
            elapsed_time = time.time()-start

            # print(status)
            # print(UELB_problem) # 制約式を全て出してくれる    
            self.flow_var_kakai = flow_var_kakai
            return flow_var_kakai, self.r_kakai, L.value(), elapsed_time

        if (self.solver_type == 'SCIP'): # PySCIPOpt+SCIP

            # SCIP Modelの作成
            model = SCIPModel("UELB_problem_SCIP")

            # 変数の定義
            L = model.addVar(vtype="C", name="L", lb=0, ub=1)
    
            flow_var_kakai = []
            for l in range(len(self.G.all_flows)):
                e_01 = [model.addVar('x{}_{}'.format(l, m), vtype='B') for m, (i, j) in enumerate(self.r_kakai)]
                flow_var_kakai.append(e_01)

            model.setObjective(L, "minimize")

            model.addCons((L) <= 1)# 負荷率1以下

            for e in range(len(self.G.edges())): # 容量制限
                model.addCons(L - (quicksum([(flow_var_kakai[l.get_id()][e])*(l.get_demand()) for l in self.G.all_flows]) / self.capacity[self.r_kakai[e][1]]) >= 0)

            for l in self.G.all_flows: #フロー保存則
                model.addCons( quicksum([flow_var_kakai[l.get_id()][e]*(l.get_demand()) for e in range(len(self.G.edges())) if self.r_kakai[e][1][0] == l.get_s()]) == l.get_demand() )
                model.addCons( quicksum([flow_var_kakai[l.get_id()][e]*(l.get_demand()) for e in range(len(self.G.edges())) if self.r_kakai[e][1][1] == l.get_s()]) == 0 )
                model.addCons( quicksum([flow_var_kakai[l.get_id()][e]*(l.get_demand()) for e in range(len(self.G.edges())) if self.r_kakai[e][1][0] == l.get_t()]) == 0 )
                model.addCons( quicksum([flow_var_kakai[l.get_id()][e]*(l.get_demand()) for e in range(len(self.G.edges())) if self.r_kakai[e][1][1] == l.get_t()]) == l.get_demand() )
            
            for l in self.G.all_flows: #フロー保存則
                for v in self.G.nodes():
                    if(v != l.get_s() and v != l.get_t()):
                        model.addCons( quicksum([(flow_var_kakai[l.get_id()][e])*(l.get_demand()) for e in range(len(self.G.edges())) if self.r_kakai[e][1][0] == v])\
                        ==quicksum([(flow_var_kakai[l.get_id()][e])*(l.get_demand()) for e in range(len(self.G.edges())) if self.r_kakai[e][1][1] == v]) )

            # print("start optimize")
            start = time.time()
            model.optimize()
            elapsed_time = time.time()-start
            
            with open(self.exact_file_name, 'a', newline='') as f:
                out = csv.writer(f)
                out.writerow([model.getObjVal(),elapsed_time]) 
            # nx.draw(self.G, with_labels=True)
            # plt.show()
            return model.getObjVal(),elapsed_time
        
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
        """
        この関数を使うときはsolve_exact_solution_to_envと併用すること
        フローがどのノードやエッジをどの順序で通過するかを計算し、
        node_flow_matrix および edge_flow_matrix にその順序を記録する関数。

        Args:
            flow_var_kakai: 各フローがエッジを通過するかどうかを示すバイナリ変数のリスト
            num_flows: フローの数
            num_edges: エッジの数
            r_kakai: エッジ情報 (ノードペアを含む)
            all_flows: 全フローの情報 (始点・終点など)
            node_flow_matrix: ノードの通過順序を格納する行列
            edge_flow_matrix: エッジの通過順序を格納する行列

        Returns:
            node_flow_matrix: 各フローのノード通過順序を示す行列
            edge_flow_matrix: 各フローのエッジ通過順序を示す行列
            infinit_loop: 探索が正常に終了したかどうかを示すフラグ
        """
        # 必要なデータを準備
        num_flows = len(self.G.all_flows)
        num_nodes = len(self.G.nodes())
        num_edges = len(self.G.edges())
        node_flow_matrix = np.zeros((num_flows, num_nodes), dtype=int)
        edge_flow_matrix = np.zeros((num_flows, num_edges), dtype=int)
        infinit_loop = False

        # フローの順番を行列に反映
        for l in range(num_flows):
            current_node_order = 1  # フローのノード通過順序 (1から始める)
            current_edge_order = 1  # フローのエッジ通過順序 (1から始める)

            # 始点ノードを取得
            current_node = self.G.all_flows[l].get_s()

            # 始点ノードの順番を記録
            node_flow_matrix[l][current_node] = current_node_order
            current_node_order += 1

            # 最大探索回数（エッジ数の3倍）
            max_steps = num_edges * 3
            steps = 0

            # フローを追跡
            while current_node != self.G.all_flows[l].get_t():  # 終点に達するまでループ
                steps += 1
                if steps > max_steps:
                    # 探索回数がエッジ数の3倍を超えたら無限ループと見なす
                    infinit_loop = True
                    return node_flow_matrix, edge_flow_matrix, infinit_loop

                # 現在のノードから出るエッジを探索
                for e in range(num_edges):
                    node_from = self.r_kakai[e][1][0]
                    node_to = self.r_kakai[e][1][1]

                    if node_from == current_node and pulp.value(flow_var_kakai[l][e]) == 1:  # 次に進むエッジを発見
                        # エッジの順序を記録
                        edge_flow_matrix[l][e] = current_edge_order
                        current_edge_order += 1

                        # 次のノードがまだ通過していなければ、その順番を記録
                        if node_flow_matrix[l][node_to] == 0:
                            node_flow_matrix[l][node_to] = current_node_order
                            current_node_order += 1

                        # 次のノードに移動
                        current_node = node_to
                        break

        # 探索が正常に完了した場合はinfinit_loopをFalseに設定
        infinit_loop = False
        return node_flow_matrix, edge_flow_matrix, infinit_loop
