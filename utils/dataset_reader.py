import time
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.utils import shuffle
from utils.data_maker import DataMaker


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

    def __init__(self, config):
        """
        Args:
            num_nodes: Number of nodes
            num_data: Number of data that will be generated
            batch_size: Batch size
        """
        self.batch_size = config.batch_size
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
        batch_load_factor = []

        for i in range(start_idx, end_idx):
            # define file path
            graph_file = f'./data/graph_file/{i%10}/graph_{i}.gml'
            commodity_file = f'./data/commodity_file/{i%10}/commodity_{i}.csv'
            edge_file = f'./data/edge_file/{i%10}/edge_numbering_file_{i}.csv'
            node_flow_file = f'./data/node_flow_file/{i%10}/node_flow_{i}.csv'
            edge_flow_file = f'./data/edge_flow_file/{i%10}/edge_flow_{i}.csv'
            exact_solution_file = f'./data/exact_solution.csv'
            
            # Make a graph
            G = nx.read_gml(graph_file,destringizer=int)
            
            # Make commodities
            commodity_list = Maker.generate_commodity()
            
            # Get adjacency matrix
            adj_matrix = nx.adjacency_matrix(G)
            adj_matrix_np = adj_matrix.toarray()
            
            # Get capacity matrix
            num_nodes = len(G.nodes)
            capacity_matrix = np.zeros((num_nodes, num_nodes))
            for edge in G.edges(data=True):
                source = edge[0]
                target = edge[1]
                capacity = edge[2].get('capacity', 0)  # エッジにcapacityがない場合は0を使う
                capacity_matrix[source, target] = capacity
            
            # Make node features
            
            
            # Read node target
            nodes_target = []
            with open(node_flow_file, newline='') as csvfile:
                csvreader = csv.reader(csvfile, delimiter=',')
                for row in csvreader:
                    nodes_target.append([int(element) for element in row])
                    
            # Read edge target
            edges_target = np.zeros((num_nodes, num_nodes, flow), dtype=int)
            
            edges = []
            with open(edge_file_path, 'r') as edge_file:
                reader = csv.reader(edge_file)
                for row in reader:
                    edges.append([int(row[0]), int(row[1]), int(row[2])])  # (id, source, target)として保存
            edges = np.array(edges)  # NumPy配列に変換
            
            edge_flow_matrix = []
            with open(edge_flow_file_path, 'r') as flow_file:
                reader = csv.reader(edge_flow_file)
                for row in reader:
                    edge_flow_matrix.append([int(value) for value in row])  # 各行を整数として保存
            edge_flow_matrix = np.array(flow_matrix)  # NumPy配列に変換
            
            flow = flow_matrix.shape[0]  # 流量数
            
            # edgesとflow_matrixを元にedges_targetを更新
            for flow_index in range(flow):
                for edge_index, edge in enumerate(edges):
                    if flow_matrix[flow_index, edge_index] > 0:  # flow_matrixの値が1以上
                        source = edge[1]  # source
                        target = edge[2]  # target
                        edges_target[source, target, flow_index] = 1  # エッジの存在を1に設定
                        edges_target[target, source, flow_index] = 1  # 無向グラフの場合は対称性を保つ

            # Don't add final connection for tour/cycle
            tour_nodes = [int(node) - 1 for node in line[line.index('output') + 1:-1]][:-1]
            
            # Compute node and edge representation of tour + tour_len
            tour_len = 0
            nodes_target = np.zeros(self.num_nodes)
            edges_target = np.zeros((self.num_nodes, self.num_nodes))
            for idx in range(len(tour_nodes) - 1):
                i = tour_nodes[idx]
                j = tour_nodes[idx + 1]
                nodes_target[i] = idx  # node targets: ordering of nodes in tour
                edges_target[i][j] = 1
                edges_target[j][i] = 1
                tour_len += W_val[i][j]
            
            # Add final connection of tour in edge target
            nodes_target[j] = len(tour_nodes) - 1
            edges_target[j][tour_nodes[0]] = 1
            edges_target[tour_nodes[0]][j] = 1
            tour_len += W_val[j][tour_nodes[0]]
            
            # Concatenate the data
            batch_edges.append(adj_matrix_np)
            batch_edges_values.append(W_val)
            batch_edges_target.append(edges_target)
            batch_nodes.append(nodes)
            batch_nodes_target.append(nodes_target)
            batch_nodes_coord.append(nodes_coord)
            batch_tour_nodes.append(tour_nodes)
            batch_tour_len.append(tour_len)
        
        # From list to tensors as a DotDict
        batch = DotDict()
        batch.edges = np.stack(batch_edges, axis=0)                 # 隣接行列 (batch_size, num_nodes, num_nodes)
        batch.edges_capacity = np.stack(batch_edges_values, axis=0) # 容量の行列 (batch_size, num_nodes, num_nodes)　
        batch.edges_target = np.stack(batch_edges_target, axis=0)   # 各品種が何番目にエッジを通るか(batch_size, num_edges, num_commodities)
        batch.nodes = np.stack(batch_nodes, axis=0)                 # 各品種のs,t (batch_size, num_nodes, num_commodities)
        batch.nodes_target = np.stack(batch_nodes_target, axis=0)   # 各品種が何番目にノードを通るか(batch_size, num_nodes, num_commodities)
        batch.Load_factor = np.stack(batch_tour_len, axis=0)        # 最大負荷率 (batch_size)
        # 厳密解の最大負荷率と計算時間を保存
        return batch
