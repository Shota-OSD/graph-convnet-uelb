import torch
import torch.nn.functional as F

import numpy as np
import random


def tour_nodes_to_W(nodes):
    """Helper function to convert ordered list of tour nodes to edge adjacency matrix.
    """
    W = np.zeros((len(nodes), len(nodes)))
    for idx in range(len(nodes) - 1):
        i = int(nodes[idx])
        j = int(nodes[idx + 1])
        W[i][j] = 1
        W[j][i] = 1
    # Add final connection of tour in edge target
    W[j][int(nodes[0])] = 1
    W[int(nodes[0])][j] = 1
    return W


def tour_nodes_to_tour_len(nodes, W_values):
    """Helper function to calculate tour length from ordered list of tour nodes.
    """
    tour_len = 0
    for idx in range(len(nodes) - 1):
        i = nodes[idx]
        j = nodes[idx + 1]
        tour_len += W_values[i][j]
    # Add final connection of tour in edge target
    tour_len += W_values[j][nodes[0]]
    return tour_len


def W_to_tour_len(W, W_values):
    """Helper function to calculate tour length from edge adjacency matrix.
    """
    tour_len = 0
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            if W[i][j] == 1:
                tour_len += W_values[i][j]
    tour_len /= 2  # Divide by 2 because adjacency matrices are symmetric
    return tour_len


def is_valid_tour(nodes, num_nodes):
    """Sanity check: tour visits all nodes given.
    """
    return sorted(nodes) == [i for i in range(num_nodes)]


def mean_load_factor(x_edges_capacity, y_pred_edges, x_edges, batch_commodities):
    """
    Computes mean load factor for given batch prediction as edge adjacency matrices (for PyTorch tensors).

    Args:
        x_edges_capacity: Edge capacity matrix (batch_size, num_nodes, num_nodes)
        y_pred_edges: Edge predictions (batch_size, num_nodes, num_nodes, num_comodities)
        x_edges: Adjacency matrix (batch_size, num_nodes, num_nodes)
        batch_commodities: Commodity information (batch_size, num_commodities, info(3))

    Returns:
        mean_tour_len: Mean tour length over batch
    """
    # Make Binery output from y_pred
    y = (y_pred_edges > 0.5).float()  # B x V x V x C

    # Compute load factor
    capacity_matrix = x_edges_capacity.unsqueeze(-1).long() # B x V x V x 1
    exact_edged_pred_matrix = (y * x_edges.unsqueeze(-1)) # B x V x V x C (binary)
    # Get demand and reshape for broadcasting
    batch_demand = batch_commodities[:, :, 2] # B x C
    batch_demand = batch_demand.unsqueeze(1).unsqueeze(2)  # (B x 1 x 1 x C)
    
    # Compute load factor per batch
    load_matrix = exact_edged_pred_matrix * batch_demand  # (B x V x V x C)
    load_factor_matrix = load_matrix / (capacity_matrix.float() + 1e-10)  # 微小値を追加してゼロ除算を防ぐ
  # (B x V x V x C)
    load_factor_matrix = load_factor_matrix.sum(dim=3) # (B x V x V)
    max_load_factor_per_batch = load_factor_matrix.max(dim=1)[0].max(dim=1)[0]  # (B)
    # Compute mean load factor over batch
    mean_maximum_load_factor = max_load_factor_per_batch.mean()  # scalar
    
    return mean_maximum_load_factor


def mean_feasible_load_factor(num_batch, num_flow, num_node, pred_paths, edges_capacity, commodities):
    """
    Computes mean load factor for given batch prediction as edge adjacency matrices (for PyTorch tensors).

    Args:
        pred_paths: Nodes that are orderd by path (batch_size, num_flow, num_node_in_each_path)
        bs_nodes: Node orderings (batch_size, num_flow, num_nodes)
        x_edges_capacity: Edge capacity matrix (batch_size, num_nodes, num_nodes)
        batch_commodities: Commodity information (batch_size, num_commodities, info(3))

    Returns:
        mean_tour_len: Mean tour length over batch
    """
    
    bs_edges = torch.zeros((num_batch, num_node, num_node, num_flow), dtype=torch.int32)

    # 改行なしでテンソル全体を出力するオプション
    torch.set_printoptions(linewidth=200)

    # comvert node order to edge
    for batch_idx, batch in enumerate(pred_paths):
        for path_idx, path in enumerate(batch):
            # obtain the path from the node order
            for node_idx in range(len(path) - 1):
                bs_edges[batch_idx, path[node_idx], path[node_idx + 1], path_idx] = 1
    
    # multiple edge capacity by demand
    demands = commodities[:, :, 2].view(num_batch, 1, 1, num_flow)
    bs_edges_demand = bs_edges * demands
    
    # sum the demand on the edges
      # 改行なしでテンソル全体を出力するオプション
    torch.set_printoptions(linewidth=200)

    bs_edges_summed = bs_edges_demand.sum(dim=-1)
    #print("bs_edges_summed: \n", bs_edges_summed[0])
    
    # compute the load factor
    load_factors = torch.where(
        edges_capacity == 0,               # 条件: edges_capacity が 0 の場合
        torch.tensor(0.0),                 # true の場合の値: 0 を設定
        bs_edges_summed.float() / edges_capacity.float()  # false の場合の値: 通常の割り算
    )
    
    # compute the maximum load factor
    max_values_per_batch = load_factors.max(dim=1).values.max(dim=1).values
    #print("max_values_per_batch: ", max_values_per_batch)
    
    # compute the mean maximum load factor
    result = max_values_per_batch.float().mean()
    
    return result

def compute_load_factor(edges_target, edges_capacity, commodities):
    """
    Compute the load factor for a given solution.
    edges_target: (batch_size, num_nodes, num_nodes, num_commodities)
    edges_capacity: (batch_size, num_nodes, num_nodes)
    commodities: (batch_size, num_commodities, 3)
    """
    # Compute the demand on each edge
    demands = commodities[:, :, 2].view(-1, 1, 1, commodities.size(1))
    print("demands: ", demands)
    edges_demand = edges_target * demands
    
    # Sum the demand on the edges
    edges_summed = edges_demand.sum(dim=-1)
    
    # Compute the load factor
    load_factor = torch.where(
        edges_capacity == 0,               # 条件: edges_capacity が 0 の場合
        torch.tensor(0.0),                 # true の場合の値: 0 を設定
        edges_summed.float() / edges_capacity.float()  # false の場合の値: 通常の割り算
    )
    max_values_per_batch = load_factor.max(dim=1).values.max(dim=1).values
    #print("max_values_per_batch: ", max_values_per_batch)
    
    return max_values_per_batch



def mean_tour_len_nodes(x_edges_values, bs_nodes):
    """
    Computes mean tour length for given batch prediction as node ordering after beamsearch (for Pytorch tensors).

    Args:
        x_edges_values: Edge values (distance) matrix (batch_size, num_nodes, num_nodes)
        bs_nodes: Node orderings (batch_size, num_nodes)

    Returns:
        mean_tour_len: Mean tour length over batch
    """
    y = bs_nodes.cpu().numpy()
    W_val = x_edges_values.cpu().numpy()
    running_tour_len = 0
    for batch_idx in range(y.shape[0]):
        for y_idx in range(y[batch_idx].shape[0] - 1):
            i = y[batch_idx][y_idx]
            j = y[batch_idx][y_idx + 1]
            running_tour_len += W_val[batch_idx][i][j]
        running_tour_len += W_val[batch_idx][j][0]  # Add final connection to tour/cycle
    return running_tour_len / y.shape[0]


def get_max_k(dataset, max_iter=1000):
    """
    Given a TSP dataset, compute the maximum value of k for which the k'th nearest neighbor 
    of a node is connected to it in the groundtruth TSP tour.
    
    For each node in all instances, compute the value of k for the next node in the tour, 
    and take the max of all ks.
    """ 
    ks = []
    for _ in range(max_iter):
        batch = next(iter(dataset))
        for idx in range(batch.edges.shape[0]):
            for row in range(dataset.num_nodes):
                # Compute indices of current node's neighbors in the TSP solution
                connections = np.where(batch.edges_target[idx][row]==1)[0]
                # Compute sorted list of indices of nearest neighbors (ascending order)
                sorted_neighbors = np.argsort(batch.edges_values[idx][row], axis=-1)  
                for conn_idx in connections:
                    ks.append(np.where(sorted_neighbors==conn_idx)[0][0])
    # print("Ks array counts: ", np.unique(ks, return_counts=True))
    # print(f"Mean: {np.mean(ks)}, StdDev: {np.std(ks)}")
    return int(np.max(ks))

def generate_commodity(G, demand_l, demand_h, commodity): # 品種の定義(numpy使用)
    determin_st = []
    commodity_list = []
    for i in range(commodity): # commodity generate
        commodity_dict = {}
        s , t = tuple(random.sample(G.nodes, 2)) # source，sink定義
        demand = random.randint(demand_l, demand_h) # demand設定
        tentative_st = [s,t]
        while True:
            if tentative_st in determin_st:
                s , t = tuple(random.sample(G.nodes, 2)) # source，sink再定義
                tentative_st = [s,t]
            else:
                break
        determin_st.append(tentative_st) # commodity決定
        commodity_dict["id"] = i
        commodity_dict["source"] = s
        commodity_dict["sink"] = t
        commodity_dict["demand"] = demand

        commodity_list.append([s,t,demand])
    commodity_list.sort(key=lambda x: -x[2]) # demand大きいものから降順

    return commodity_list