import numpy as np
from typing import List


def compute_edge_usage(
    assignment: List[List[int]],
    commodity_list: List[List[int]],
    num_nodes: int,
) -> np.ndarray:
    """
    現在のパス割当からエッジ使用量行列を計算する。

    Args:
        assignment: 各コモディティのパス [C][path_length]
        commodity_list: [[src, dst, demand], ...] [C]
        num_nodes: グラフのノード数

    Returns:
        edge_usage: [V, V] - 各エッジの総使用量
    """
    usage = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    for c, path in enumerate(assignment):
        if len(path) < 2:
            continue
        demand = float(commodity_list[c][2])
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            usage[u, v] += demand
    return usage


def compute_load_factor(
    edge_usage: np.ndarray,
    edge_capacity: np.ndarray,
) -> float:
    """
    最大負荷率を計算する (max_edge(usage / capacity))。

    Args:
        edge_usage: エッジ使用量行列 [V, V]
        edge_capacity: エッジ容量行列 [V, V]

    Returns:
        load_factor: 最大負荷率
    """
    valid_mask = edge_capacity > 1e-8
    if not valid_mask.any():
        return 0.0
    ratios = np.where(valid_mask, edge_usage / (edge_capacity + 1e-8), 0.0)
    return float(np.max(ratios))
