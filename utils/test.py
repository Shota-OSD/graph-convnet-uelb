import numpy as np
import csv

# Edgeファイルの読み込みとエッジIDマッピング
edge_dict = {}
with open('./data/edge_file/0/edge_numbering_file_0.csv', 'r') as edge_file:
    reader = csv.reader(edge_file)
    for row in reader:
        edge_id, source, target = int(row[0]), int(row[1]), int(row[2])
        edge_dict[(source, target)] = edge_id
        edge_dict[(target, source)] = edge_id  # 無向グラフの場合、逆も対応

# Flowファイルの読み込み
flows = []
with open('./data/exact_flow/0/exact_flow_0.csv', 'r') as flow_file:
    reader = csv.reader(flow_file)
    for row in reader:
        flows.append([int(x) for x in row])

# 品種数とノード数
num_flows = len(flows)
num_nodes = len(flows[0])

# エッジ数（edge_fileの長さ）
num_edges = len(edge_dict)

# エッジ行列の作成（縦: 品種, 横: エッジ）
flow_matrix = np.zeros((num_flows, num_edges), dtype=int)

# Flowからエッジへの変換
for flow_idx, flow in enumerate(flows):
    # 各品種のフロー（ノード間の移動をエッジに変換）
    for i in range(1, num_nodes):
        if flow[i-1] != 0 and flow[i] != 0:  # 0は無視
            source, target = flow[i-1], flow[i]
            if (source, target) in edge_dict:
                edge_id = edge_dict[(source, target)]
                flow_matrix[flow_idx, edge_id] = i  # i番目の順序として保存

# 結果を表示
print(flow_matrix)

# 結果をCSVファイルとして保存
with open('flow_edge_matrix.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(flow_matrix)
