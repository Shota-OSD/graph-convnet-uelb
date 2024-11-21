import os
import csv
import networkx as nx

def create_data_files(config, data_mode="test"):
    """
    指定されたモードのデータファイル（グラフ、品種、厳密解など）を生成し保存する関数。

    Args:
        config: 設定オブジェクト（`num_{data_mode}_data` や `solver_type` を含む必要あり）。
        data_mode (str): データモード ("train", "val", "test")。デフォルトは "test"。
    """
    num_data = getattr(config, f'num_{data_mode}_data')
    solver_type = config.solver_type
    Maker = DataMaker(config)
    exact_file_name = f"./data/{data_mode}_data/exact_solution.csv"
    infinit_loop_count = 0
    incorrect_value_count = 0

    for data in range(num_data):
        if data % 10 == 0:
            print(f"{data} data was created.")

        # ディレクトリ番号の定義
        file_number = data - (data % 10)
        
        # ディレクトリ作成
        directories = ["graph_file", "commodity_file", "node_flow_file"]
        #directories = ["graph_file", "commodity_file", "edge_file", "node_flow_file", "edge_flow_file"]
        for directory in directories:
            path = f"./data/{data_mode}_data/{directory}/{file_number}"
            os.makedirs(path, exist_ok=True)

        # ファイル名の定義
        graph_file_name = f"./data/{data_mode}_data/graph_file/{file_number}/graph_{data}.gml"
        commodity_file_name = f"./data/{data_mode}_data/commodity_file/{file_number}/commodity_data_{data}.csv"
        node_flow_file_name = f"./data/{data_mode}_data/node_flow_file/{file_number}/node_flow_{data}.csv"
        #edge_file_name = f"./data/{data_mode}_data/edge_file/{file_number}/edge_numbering_{data}.csv"
        #edge_flow_file_name = f"./data/{data_mode}_data/edge_flow_file/{file_number}/edge_flow_{data}.csv"

        # 作成したデータが適切でない場合のやり直し
        while True:
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

            # 厳密解の計算
            E = SolveExactSolution(solver_type, commodity_file_name, graph_file_name)
            flow_var_kakai, edge_list, objective_value, elapsed_time = E.solve_exact_solution_to_env()
            node_flow_matrix, edge_flow_matrix, infinit_loop = E.generate_flow_matrices(flow_var_kakai)
            
            # 厳密解が1以上、または厳密解のフローが正しく導けなかった場合のやり直し
            if infinit_loop:
                infinit_loop_count += 1
            elif objective_value >= 1.0:
                incorrect_value_count += 1
            else:
                break
        
        # 厳密解保存
        with open(exact_file_name, 'a', newline='') as f:
            out = csv.writer(f)
            out.writerow([objective_value, elapsed_time]) 

        # 厳密解ノードフロー保存
        with open(node_flow_file_name, 'w', newline='') as file:
            writer = csv.writer(file)
            for row in node_flow_matrix:
                writer.writerow(row)
        
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
    print(f"Infinit loops: {infinit_loop_count}, Incorrect values: {incorrect_value_count}")
