import torch
import torch.nn.functional as F

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from ..graph.graph_utils import *


def plot_uelb(p, edges, edges_targets, title="default"):
    """
    Helper function to plot the UELB graph with different colors for each commodity.
    
    Args:
        p: Matplotlib figure/subplot
        edges: Edge adjacency matrix (num_nodes * num_nodes)
        edges_targets: Edge matrices for each commodity (num_nodes * num_nodes * num_commodities)
        title: Title of figure/subplot
    
    Returns:
        p: Updated figure/subplot
    """

    def _edges_to_node_pairs(W):
        """Helper function to convert edge matrix into pairs of adjacent nodes."""
        pairs = []
        for r in range(len(W)):
            for c in range(len(W)):
                if W[r][c] == 1:
                    pairs.append((r, c))
        return pairs

    # ノード座標をランダムに配置（実際のデータがあればそれを使う）
    num_nodes = edges.shape[0]
    pos = {i: (np.random.rand(), np.random.rand()) for i in range(num_nodes)}
    
    # グラフオブジェクトの作成
    G = nx.from_numpy_array(edges)

    # ノードの描画
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=100)

    # 品種ごとのエッジを描画（色を変える）
    num_commodities = edges_targets.shape[2]
    colors = plt.cm.rainbow(np.linspace(0, 1, num_commodities))  # 品種ごとに色を設定

    # 品種が流れているエッジの描画
    for commodity_index in range(num_commodities):
        # 各品種の隣接行列を取得
        W_target = edges_targets[:, :, commodity_index]
        # 隣接行列からエッジのペアを取得
        target_pairs = _edges_to_node_pairs(W_target)
        # エッジを描画
        nx.draw_networkx_edges(G, pos, edgelist=target_pairs, edge_color=[colors[commodity_index]], width=2, alpha=0.7)

    # 流れていないエッジの描画
    all_edges = _edges_to_node_pairs(edges)  # すべてのエッジを取得
    flowing_edges = set(tuple(pair) for commodity_index in range(num_commodities)
                         for pair in _edges_to_node_pairs(edges_targets[:, :, commodity_index]))
    
    # 流れていないエッジを取得
    non_flowing_edges = [edge for edge in all_edges if edge not in flowing_edges]
    nx.draw_networkx_edges(G, pos, edgelist=non_flowing_edges, edge_color='black', width=0.5, alpha=0.5)

    # タイトルを設定
    p.set_title(title)
    return p



def plot_tsp_heatmap(p, x_coord, W_val, W_pred, title="default"):
    """
    Helper function to plot predicted TSP tours with edge strength denoting confidence of prediction.
    
    Args:
        p: Matplotlib figure/subplot
        x_coord: Coordinates of nodes
        W_val: Edge values (distance) matrix
        W_pred: Edge predictions matrix
        title: Title of figure/subplot
    
    Returns:
        p: Updated figure/subplot
    
    """

    def _edges_to_node_pairs(W):
        """Helper function to convert edge matrix into pairs of adjacent nodes.
        """
        pairs = []
        edge_preds = []
        for r in range(len(W)):
            for c in range(len(W)):
                if W[r][c] > 0.25:
                    pairs.append((r, c))
                    edge_preds.append(W[r][c])
        return pairs, edge_preds
        
    G = nx.from_numpy_array(W_val)
    pos = dict(zip(range(len(x_coord)), x_coord.tolist()))
    node_pairs, edge_color = _edges_to_node_pairs(W_pred)
    node_color = ['g'] + ['b'] * (len(x_coord) - 1)  # Green for 0th node, blue for others
    nx.draw_networkx_nodes(G, pos, node_color=node_color, node_size=50)
    nx.draw_networkx_edges(G, pos, edgelist=node_pairs, edge_color=edge_color, edge_cmap=plt.cm.Reds, width=0.75)
    p.set_title(title)
    return p


def plot_predictions(x_nodes_coord, x_edges, x_edges_values, y_edges, y_pred_edges, num_plots=3):
    """
    Plots groundtruth TSP tour vs. predicted tours (without beamsearch).
    
    Args:
        x_nodes_coord: Input node coordinates (batch_size, num_nodes, node_dim)
        x_edges: Input edge adjacency matrix (batch_size, num_nodes, num_nodes)
        x_edges_values: Input edge distance matrix (batch_size, num_nodes, num_nodes)
        y_edges: Groundtruth labels for edges (batch_size, num_nodes, num_nodes)
        y_pred_edges: Predictions for edges (batch_size, num_nodes, num_nodes)
        num_plots: Number of figures to plot
    
    """
    y = F.softmax(y_pred_edges, dim=3)  # B x V x V x voc_edges
    y_bins = y.argmax(dim=3)  # Binary predictions: B x V x V
    y_probs = y[:,:,:,1]  # Prediction probabilities: B x V x V
    for f_idx, idx in enumerate(np.random.choice(len(y), num_plots, replace=False)):
        f = plt.figure(f_idx, figsize=(10, 5))
        x_coord = x_nodes_coord[idx].cpu().numpy()
        W = x_edges[idx].cpu().numpy()
        W_val = x_edges_values[idx].cpu().numpy()
        W_target = y_edges[idx].cpu().numpy()
        W_sol_bins = y_bins[idx].cpu().numpy()
        W_sol_probs = y_probs[idx].cpu().numpy()
        plt1 = f.add_subplot(121)
        plot_tsp(plt1, x_coord, W, W_val, W_target, 'Groundtruth: {:.3f}'.format(W_to_tour_len(W_target, W_val)))
        plt2 = f.add_subplot(122)
        plot_tsp_heatmap(plt2, x_coord, W_val, W_sol_probs, 'Prediction Heatmap')
        plt.show()


def plot_predictions_beamsearch(x_nodes_coord, x_edges, x_edges_values, y_edges, y_pred_edges, bs_nodes, num_plots=3):
    """
    Plots groundtruth TSP tour vs. predicted tours (with beamsearch).
    
    Args:
        x_nodes_coord: Input node coordinates (batch_size, num_nodes, node_dim)
        x_edges: Input edge adjacency matrix (batch_size, num_nodes, num_nodes)
        x_edges_values: Input edge distance matrix (batch_size, num_nodes, num_nodes)
        y_edges: Groundtruth labels for edges (batch_size, num_nodes, num_nodes)
        y_pred_edges: Predictions for edges (batch_size, num_nodes, num_nodes)
        bs_nodes: Predicted node ordering in TSP tours after beamsearch (batch_size, num_nodes)
        num_plots: Number of figures to plot
    
    """
    y = F.softmax(y_pred_edges, dim=3)  # B x V x V x voc_edges
    y_bins = y.argmax(dim=3)  # Binary predictions: B x V x V
    y_probs = y[:,:,:,1]  # Prediction probabilities: B x V x V
    for f_idx, idx in enumerate(np.random.choice(len(y), num_plots, replace=False)):
        f = plt.figure(f_idx, figsize=(15, 5))
        x_coord = x_nodes_coord[idx].cpu().numpy()
        W = x_edges[idx].cpu().numpy()
        W_val = x_edges_values[idx].cpu().numpy()
        W_target = y_edges[idx].cpu().numpy()
        W_sol_bins = y_bins[idx].cpu().numpy()
        W_sol_probs = y_probs[idx].cpu().numpy()
        W_bs = tour_nodes_to_W(bs_nodes[idx].cpu().numpy())
        plt1 = f.add_subplot(131)
        plot_tsp(plt1, x_coord, W, W_val, W_target, 'Groundtruth: {:.3f}'.format(W_to_tour_len(W_target, W_val)))
        plt2 = f.add_subplot(132)
        plot_tsp_heatmap(plt2, x_coord, W_val, W_sol_probs, 'Prediction Heatmap')
        plt3 = f.add_subplot(133)
        plot_tsp(plt3, x_coord, W, W_val, W_bs, 'Beamsearch: {:.3f}'.format(W_to_tour_len(W_bs, W_val)))
        plt.show()
