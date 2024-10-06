#LP-ksps_envで使う厳密解を求めるためのプログラムファイル
from mip import *
import time
import csv
import networkx as nx
import matplotlib.pyplot as plt
from flow import Flow
import pulp
# from pyscipopt import Model as SCIPModel, quicksum

class Solve_exact_solution():
    def __init__(self, episode, solver_type, exact_file_name):
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

        self.episode = episode
        self.solver_type = solver_type
        self.exact_file_name = exact_file_name
        self.G = nx.read_gml("/Users/osadashouta/Desktop/Research/RL-KSPs/value/graph.gml",destringizer=int) # グラフの定義
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

        with open('/Users/osadashouta/Desktop/Research/RL-KSPs/value/commodity_data.csv',newline='') as f: # 品種の読み込み
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

        # 出力するファイルのパス
        edge_numbering_file = "edge_numbering_file.csv"

        # CSVファイルに書き込み
        with open(edge_numbering_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            # 各タプルをCSVフォーマットに変換して書き込み
            for item in self.r_kakai:
                writer.writerow([item[0], item[1][0], item[1][1]])

        print(f"データが {edge_numbering_file} に保存されました。")

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

            print("start optimize")
            #線形計画問題を解く
            start = time.time()
            UELB_kakai.optimize()
            elapsed_time = time.time()-start

            with open(self.exact_file_name, 'a', newline='') as f:
                out = csv.writer(f)
                out.writerow([self.episode, UELB_kakai.objective_value, elapsed_time]) 
            return UELB_kakai.objective_value,elapsed_time
        
        if (self.solver_type == 'pulp'): # pulp+CBC
            UELB_problem = pulp.LpProblem('UELB', pulp.LpMinimize) # モデルの名前
            """
            ・LpProblem: 線形計画問題を定義するための pulp のクラス。
            ・'UELB' : 問題の名前で、自由に決められる。
            ・pulp.LpMinimize : 問題が「最小化問題」であることを指定。負荷率 L を最小化する目的を表す。

            ・self.r_kakai = [(0, ('A', 'B')), (1, ('B', 'C')), (2, ('C', 'D'))]

            ・フロー1（l=0）:
                x0_0: エッジ A-B を通るかどうか（0 または 1）
                x0_1: エッジ B-C を通るかどうか（0 または 1）
                x0_2: エッジ C-D を通るかどうか（0 または 1）
            ・フロー2（l=1）:
                x1_0: エッジ A-B を通るかどうか（0 または 1）
                x1_1: エッジ B-C を通るかどうか（0 または 1）
                x1_2: エッジ C-D を通るかどうか（0 または 1）

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

            print("start optimize")
            start = time.time()
            status = UELB_problem.solve(pulp.PULP_CBC_CMD(msg=False)) # 線形計画問題を解く
            elapsed_time = time.time()-start

            # print(status)
            # print(UELB_problem) # 制約式を全て出してくれる 

            # flow_var_kakai のバイナリ変数の値を CSV ファイルに保存する
            with open('exact_solution_flow.csv', mode='w', newline='') as file:
                writer = csv.writer(file)
                
                # 各フローに対してループ
                for l in range(len(flow_var_kakai)):  # フローのループ
                    row = []
                    for e in range(len(flow_var_kakai[l])):  # 各フローに対するエッジのループ
                        var_value = pulp.value(flow_var_kakai[l][e])  # 変数の値を取得
                        row.append(var_value)  # 値を行に追加
                    writer.writerow(row)  # 行を CSV ファイルに書き込む
            
            with open(self.exact_file_name, 'a', newline='') as f:
                out = csv.writer(f)
                out.writerow([self.episode, L.value(),elapsed_time]) 
            return L.value(),elapsed_time

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
                out.writerow([self.episode, model.getObjVal(),elapsed_time]) 
            # nx.draw(self.G, with_labels=True)
            # plt.show()
            return model.getObjVal(),elapsed_time