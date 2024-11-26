import time
import numpy as np
from sklearn.utils import shuffle
import networkx as nx
import csv
import torch

class DotDict(dict):
    """Wrapper around in-built dict class to access members through the dot operation.
    d = DotDict(name="Alice", age=25)
    print(d.name)  # "Alice" と表示される
    print(d.age)   # 25 と表示される

    # 通常の辞書としても機能
    print(d['name'])  # "Alice"
    print(d['age'])   # 25
    """

    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self


class DatasetReader(object):
    """Iterator that reads UELB dataset files and yields mini-batches.
    """

    def __init__(self, num_data, batch_size, mode):
        """
        Args:
            num_nodes: Number of nodes
            num_data: Number of data that will be generated
            batch_size: Batch size
        """
        self.num_data = num_data
        self.batch_size = batch_size
        self.mode = mode
        self.max_iter = (self.num_data // self.batch_size)

    def __iter__(self):
        """
        self.max_iter = 5 で、self.batch_size = 10
            1回目のループ: start_idx = 0, end_idx = 10
            2回目のループ: start_idx = 10, end_idx = 20
            3回目のループ: start_idx = 20, end_idx = 30
            4回目のループ: start_idx = 30, end_idx = 40
            5回目のループ: start_idx = 40, end_idx = 50
        """
        for batch in range(self.max_iter):
            start_idx = batch * self.batch_size
            end_idx = (batch + 1) * self.batch_size
            yield self.process_batch(start_idx, end_idx)

    def process_batch(self, start_idx, end_idx):
        """Helper function to convert raw lines into a mini-batch as a DotDict.
        """
        # define file path
        
        batch_edges = []            # Adjacency matrix　
        batch_edges_capacity = []   # Capacity of edges
        batch_edges_target = []     # Binary classification targets (0/1)
        batch_nodes = []            # Node feature (Source or target of comodities?)
        batch_nodes_target = []     # Multi-class classification targets (`num_nodes` classes)
        batch_commodities = []
        batch_load_factor = []

        exact_solution_file = f'./data/{self.mode}_data/exact_solution.csv'

        for i in range(start_idx, end_idx):
            # define file path
            graph_file = f'./data/{self.mode}_data/graph_file/{i-(i%10)}/graph_{i}.gml'
            commodity_file = f'./data/{self.mode}_data/commodity_file/{i-(i%10)}/commodity_data_{i}.csv'
            node_flow_file = f'./data/{self.mode}_data/node_flow_file/{i-(i%10)}/node_flow_{i}.csv'
            edge_file = f'./data/{self.mode}_data/edge_file/{i-(i%10)}/edge_numbering_{i}.csv'
            edge_flow_file = f'./data/{self.mode}_data/edge_flow_file/{i-(i%10)}/edge_flow_{i}.csv'
            
            """ Make a graph """
            G = nx.read_gml(graph_file,destringizer=int)
            num_nodes = len(G.nodes)
            
            """ Make commodities """
            commodity_list = []
            with open(commodity_file, 'r') as commodity_data:
                reader = csv.reader(commodity_data)
                for row in reader:
                    source = int(row[0])
                    target = int(row[1])
                    demand = int(row[2])
                    commodity_list.append([source, target, demand])
            num_commodities = len(commodity_list)

            """ Get adjacency matrix """
            adj_matrix = nx.adjacency_matrix(G)
            adj_matrix_np = adj_matrix.toarray()
            
            """ Get capacity matrix """
            capacity_matrix = np.zeros((num_nodes, num_nodes))
            for edge in G.edges(data=True):
                source = edge[0]
                target = edge[1]
                capacity = edge[2].get('capacity', 0)  # エッジにcapacityがない場合は0を使う
                capacity_matrix[source, target] = capacity
            
            """ Make node features """
            nodes = np.ones((num_nodes, num_commodities), dtype=int)
            # commodity_listを参照してsourceとtargetに基づきnodesを更新
            for commodity_index, (source, target, demand) in enumerate(commodity_list):
                nodes[source, commodity_index] = demand  # sourceの場合はdemandを設定
                nodes[target, commodity_index] = -demand  # targetの場合は負のdemandを設定
            
            """ Read node target """
            nodes_target = []
            with open(node_flow_file, newline='') as csvfile:
                csvreader = csv.reader(csvfile, delimiter=',')
                for row in csvreader:
                    nodes_target.append([int(element) for element in row])
                    
            """ Read edge target """
            edges_target = np.zeros((num_nodes, num_nodes, num_commodities), dtype=int)
                        
            for flow_idx in range(num_commodities):
                # node_orderをテンソルに変換
                node_order = torch.tensor(nodes_target[flow_idx], dtype=torch.int)
                valid_nodes = (node_order > 0).nonzero(as_tuple=True)[0]
                sorted_nodes = valid_nodes[node_order[valid_nodes].argsort()]
                for i in range(len(sorted_nodes) - 1):
                    src = sorted_nodes[i].item()  # 出発ノード
                    tgt = sorted_nodes[i + 1].item()  # 到着ノード
                    edges_target[src, tgt, flow_idx] = 1  # エッジを通過しているとマーク

            
            # Concatenate the data
            batch_edges.append(adj_matrix_np)
            batch_edges_capacity.append(capacity_matrix)
            batch_edges_target.append(edges_target)
            batch_nodes.append(nodes)
            batch_nodes_target.append(nodes_target)
            batch_commodities.append(commodity_list)
        
        # From list to tensors as a DotDict
        batch_load_factor = []
        with open(exact_solution_file, 'r') as exact_solution:
            reader = csv.reader(exact_solution)
            for i, row in enumerate(reader):
                if start_idx <= i < end_idx:
                    batch_load_factor.append(float(row[0]))

        batch = DotDict()
        batch.edges = np.stack(batch_edges, axis=0)                 # 隣接行列 (batch_size, num_nodes, num_nodes)
        batch.edges_capacity = np.stack(batch_edges_capacity, axis=0) # 容量の行列 (batch_size, num_nodes, num_nodes)　
        batch.edges_target = np.stack(batch_edges_target, axis=0)   # 各品種が何番目にエッジを通るか(batch_size, num_nodes, num_nodes, num_commodities)
        batch.nodes = np.stack(batch_nodes, axis=0)                 # ノード情報(各品種のs,t, demandが含まれている) (batch_size, num_nodes, num_commodities)
        batch.nodes_target = np.stack(batch_nodes_target, axis=0)   # 各品種が何番目にノードを通るか(batch_size, num_commodities, num_nodes)
        batch.commodities = np.stack(batch_commodities, axis=0)    # 各品種のdemand (batch_size, num_commodities, 3)
        batch.load_factor = np.stack(batch_load_factor, axis=0)    # 最大負荷率 (batch_size)
        # 厳密解の最大負荷率と計算時間を保存
        return batch
