import torch
import networkx as nx
import numpy as np
from typing import List, Dict, Tuple, Optional
from torch import Tensor

from .path_pool_manager import PathPoolManager
from src.gnn_ils.utils.load_utils import compute_load_factor, compute_edge_usage


class ILSEnvironment:
    """
    ILS 改善ループ環境 (batch_size=1)。

    状態空間:
        x_nodes:           [1, V, C] - torch.long
        x_commodities:     [1, C, 3] - torch.float
        x_edges_capacity:  [1, V, V] - torch.float
        x_edges_usage:     [1, V, V] - torch.float (毎ステップ更新)
        current_assignment: List[List[int]] - 現在のパス割当 [C][path_length]
        path_pool:         List[List[List[int]]] - 候補パスプール [C][P_c][path_length]

    行動空間:
        Level1: コモディティ選択 Discrete(C)
        Level2: パス選択 Discrete(max_candidate_paths)
    """

    def __init__(self, config: dict):
        """
        Args:
            config:
                - num_nodes, num_commodities
                - max_iterations (default: 50)
                - no_improve_patience (default: 10)
                - perturbation_prob (default: 0.1)
                - K, max_disjoint, max_candidate_paths
                - reward_mode: 'shared' or 'decomposed' (default: 'shared')
        """
        self.num_nodes = config['num_nodes']
        self.num_commodities = config['num_commodities']
        self.max_iterations = config.get('max_iterations', 50)
        self.no_improve_patience = config.get('no_improve_patience', 10)
        self.perturbation_prob = config.get('perturbation_prob', 0.1)
        self.max_candidate_paths = config.get('max_candidate_paths', 15)
        self.reward_mode = config.get('reward_mode', 'shared')

        self.path_pool_manager = PathPoolManager(config)

        # 状態変数
        self.G: Optional[nx.Graph] = None
        self.commodity_list: List[List[int]] = []
        self.path_pool: List[List[List[int]]] = []
        self.current_assignment: List[List[int]] = []
        self.x_nodes: Optional[Tensor] = None
        self.x_commodities: Optional[Tensor] = None
        self.x_edges_capacity: Optional[Tensor] = None
        self.x_edges_usage: Optional[Tensor] = None
        self.current_load_factor: float = float('inf')
        self.iteration: int = 0
        self.no_improve_count: int = 0
        self.capacity_np: Optional[np.ndarray] = None

    def reset(
        self,
        G: nx.Graph,
        commodity_list: List[List[int]],
        x_nodes: Tensor,           # [1, V, C] - torch.long
        x_commodities: Tensor,     # [1, C, 3] - torch.float
        x_edges_capacity: Tensor,  # [1, V, V] - torch.float
    ) -> Dict:
        """
        エピソードの初期化。

        1. PathPoolManager でパスプール構築
        2. 容量制約を無視した最短パス (Dijkstra) で初期割当
        3. 初期負荷を計算

        Returns:
            state 辞書
        """
        self.G = G
        self.commodity_list = commodity_list
        self.x_nodes = x_nodes
        self.x_commodities = x_commodities
        self.x_edges_capacity = x_edges_capacity
        self.capacity_np = x_edges_capacity[0].cpu().numpy()

        # パスプール構築
        self.path_pool = self.path_pool_manager.build_path_pool(G, commodity_list)

        # 初期割当: 容量無視の最短パス (到達保証)
        self.current_assignment = self._compute_initial_assignment()

        # 初期エッジ使用量と負荷
        usage_np = compute_edge_usage(self.current_assignment, commodity_list, self.num_nodes)
        self.x_edges_usage = torch.tensor(
            usage_np, dtype=torch.float32
        ).unsqueeze(0)  # [1, V, V]
        self.current_load_factor = compute_load_factor(usage_np, self.capacity_np)

        # カウンタ初期化
        self.iteration = 0
        self.no_improve_count = 0

        return self._build_state()

    def _compute_initial_assignment(self) -> List[List[int]]:
        """Dijkstra で初期パス割当 (容量無視)。全コモディティ到達を保証する。"""
        assignment = []
        for commodity in self.commodity_list:
            src, dst = int(commodity[0]), int(commodity[1])
            try:
                path = list(nx.dijkstra_path(self.G, src, dst, weight='weight'))
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                # パスが存在しない場合はパスプールの先頭を使用
                pool = self.path_pool[len(assignment)] if len(assignment) < len(self.path_pool) else []
                path = pool[0] if pool else [src]
            assignment.append(path)
        return assignment

    def step(
        self,
        selected_commodity: int,
        selected_path_idx: int,
    ) -> Tuple[Dict, float, bool, Dict]:
        """
        1コモディティ1パス交換を実行し、負荷を再計算する。

        Args:
            selected_commodity: 交換対象コモディティのインデックス
            selected_path_idx:  path_pool[selected_commodity] 内のパスインデックス

        Returns:
            state:  更新後の状態辞書
            reward: 改善ベース報酬
            done:   終了条件
            info:   メトリクス辞書
        """
        old_load_factor = self.current_load_factor

        # パス交換 (インデックス境界チェック)
        paths = self.path_pool[selected_commodity]
        new_path_idx = min(selected_path_idx, len(paths) - 1)
        self.current_assignment[selected_commodity] = paths[new_path_idx]

        # エッジ使用量と負荷を再計算
        usage_np = compute_edge_usage(
            self.current_assignment, self.commodity_list, self.num_nodes
        )
        new_load_factor = compute_load_factor(usage_np, self.capacity_np)

        self.x_edges_usage = torch.tensor(
            usage_np, dtype=torch.float32
        ).unsqueeze(0)
        self.current_load_factor = new_load_factor

        # カウンタ更新
        self.iteration += 1
        if new_load_factor < old_load_factor - 1e-8:
            self.no_improve_count = 0
        else:
            self.no_improve_count += 1

        reward = self._compute_reward(old_load_factor, new_load_factor)
        done = self._check_done()

        info = {
            'load_factor': new_load_factor,
            'old_load_factor': old_load_factor,
            'iteration': self.iteration,
            'no_improve_count': self.no_improve_count,
        }

        return self._build_state(), reward, done, info

    def _compute_reward(self, old_load_factor: float, new_load_factor: float) -> float:
        """
        報酬計算: 改善時は正、悪化時は負。

        shared モード: reward = -(new_lf - old_lf)
        """
        return -(new_load_factor - old_load_factor)

    def _check_done(self) -> bool:
        """終了条件: max_iterations 到達 or no_improve_patience 超過。"""
        return (
            self.iteration >= self.max_iterations
            or self.no_improve_count >= self.no_improve_patience
        )

    def get_commodity_mask(self) -> Tensor:
        """
        交換可能なコモディティのマスクを生成する。

        条件: 候補パスが1本以上あり、現在のパスと異なるパスが存在する。

        Returns:
            mask: [1, C] - bool
        """
        mask = torch.zeros(1, self.num_commodities, dtype=torch.bool)
        for c in range(self.num_commodities):
            paths = self.path_pool[c]
            if len(paths) <= 1:
                continue
            current = tuple(self.current_assignment[c])
            has_alternative = any(tuple(p) != current for p in paths)
            if has_alternative:
                mask[0, c] = True
        return mask

    def get_path_mask(self, commodity_idx: int) -> Tensor:
        """
        指定コモディティの有効パスマスクを生成する。

        Returns:
            mask: [1, max_candidate_paths] - bool
        """
        paths = self.path_pool[commodity_idx]
        num_valid = min(len(paths), self.max_candidate_paths)
        mask = torch.zeros(1, self.max_candidate_paths, dtype=torch.bool)
        mask[0, :num_valid] = True
        return mask

    def _build_state(self) -> Dict:
        """現在の状態辞書を構築する。"""
        return {
            'x_nodes': self.x_nodes,
            'x_commodities': self.x_commodities,
            'x_edges_capacity': self.x_edges_capacity,
            'x_edges_usage': self.x_edges_usage,
            'current_assignment': self.current_assignment,
            'path_pool': self.path_pool,
            'commodity_mask': self.get_commodity_mask(),
            'load_factor': self.current_load_factor,
        }
