import torch
import torch.nn as nn
from typing import List, Tuple, Optional
from torch import Tensor

from .gnn_ils_encoder import GNNILSEncoder
from .commodity_selector import CommoditySelector
from .path_selector import PathSelector
from .value_head import ILSValueHead


class GNNILSModel(nn.Module):
    """
    GNN-ILS Actor-Critic Model。

    Architecture:
        GNNILSEncoder (shared)
            ├── CommoditySelector (Level1 Actor)
            ├── PathSelector      (Level2 Actor)
            └── ILSValueHead      (Critic)
    """

    def __init__(self, config: dict, dtypeFloat=torch.float32, dtypeLong=torch.long):
        """
        Args:
            config:
                - num_nodes, num_commodities, hidden_dim, num_layers
                - aggregation, dropout_rate
                - max_candidate_paths (default: 15)
                - commodity_selector_mlp_layers (default: 2)
                - path_selector_mlp_layers (default: 2)
                - value_head_mlp_layers (default: 3)
                - use_graph_embedding_value (default: False)
        """
        super().__init__()
        self.dtypeFloat = dtypeFloat
        self.dtypeLong = dtypeLong
        self.demand_max = config.get('demand_higher', 500)
        self.path_aggregation = config.get('path_aggregation', 'mean')

        self.encoder = GNNILSEncoder(config)

        num_commodities = config['num_commodities']
        num_nodes = config['num_nodes']
        hidden_dim = config['hidden_dim']
        max_paths = config.get('max_candidate_paths', 15)

        self.commodity_selector = CommoditySelector(
            hidden_dim=hidden_dim,
            num_commodities=num_commodities,
            mlp_layers=config.get('commodity_selector_mlp_layers', 2),
        )
        self.path_selector = PathSelector(
            hidden_dim=hidden_dim,
            max_paths=max_paths,
            mlp_layers=config.get('path_selector_mlp_layers', 2),
            path_aggregation=self.path_aggregation,
        )
        self.value_head = ILSValueHead(
            hidden_dim=hidden_dim,
            num_nodes=num_nodes,
            num_commodities=num_commodities,
            mlp_layers=config.get('value_head_mlp_layers', 3),
            use_graph_embedding=config.get('use_graph_embedding_value', False),
        )

    def encode(
        self,
        x_nodes: Tensor,           # [B, V, C] - torch.long
        x_commodities: Tensor,     # [B, C, 3] - torch.float
        x_edges_capacity: Tensor,  # [B, V, V] - torch.float
        x_edges_usage: Tensor,     # [B, V, V] - torch.float
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Returns:
            node_features:   [B, V, C, H]
            edge_features:   [B, V, V, C, H]
            graph_embedding: [B, H]
        """
        return self.encoder(x_nodes, x_commodities, x_edges_capacity, x_edges_usage)

    def select_commodity(
        self,
        edge_features: Tensor,                        # [B, V, V, C, H]
        current_assignment: List[List[List[int]]],     # [B][C][path_length]
        demands: Tensor,                               # [B, C]
        commodity_mask: Tensor,                        # [B, C]
        deterministic: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Level1: コモディティ選択。

        Returns:
            selected_commodity: [B] - int
            log_prob:           [B] - 選択されたコモディティの対数確率
            entropy:            [B]
        """
        path_commodity_features = self._build_commodity_path_features(
            edge_features, current_assignment, demands
        )
        action_probs, log_probs_all, entropy = self.commodity_selector(
            path_commodity_features, commodity_mask
        )

        if deterministic:
            selected = action_probs.argmax(dim=-1)
        else:
            dist = torch.distributions.Categorical(probs=action_probs)
            selected = dist.sample()

        log_prob = log_probs_all.gather(1, selected.unsqueeze(1)).squeeze(1)  # [B]
        return selected, log_prob, entropy

    def select_path(
        self,
        edge_features: Tensor,                   # [B, V, V, C, H]
        selected_commodity: Tensor,              # [B]
        candidate_paths: List[List[List[int]]],  # [B][P_c][path_length]
        current_paths: List[List[int]],          # [B][path_length] (選択コモディティの現パス)
        demands: Tensor,                         # [B] (生の demand、正規化はモデル内で行う)
        path_mask: Tensor,                       # [B, max_paths]
        deterministic: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Level2: パス選択。

        Returns:
            selected_path_idx: [B] - int
            log_prob:          [B]
            entropy:           [B]
        """
        demands_norm = demands / self.demand_max
        action_probs, log_probs_all, entropy = self.path_selector(
            edge_features, selected_commodity, candidate_paths, current_paths, demands_norm, path_mask
        )

        if deterministic:
            selected = action_probs.argmax(dim=-1)
        else:
            dist = torch.distributions.Categorical(probs=action_probs)
            selected = dist.sample()

        log_prob = log_probs_all.gather(1, selected.unsqueeze(1)).squeeze(1)  # [B]
        return selected, log_prob, entropy

    def get_value(
        self,
        node_features: Tensor,          # [B, V, C, H]
        graph_embedding: Optional[Tensor] = None,  # [B, H]
    ) -> Tensor:
        """
        Critic: 状態価値 V(s)。

        Returns:
            state_value: [B]
        """
        return self.value_head(node_features, graph_embedding)

    def _build_commodity_path_features(
        self,
        edge_features: Tensor,                     # [B, V, V, C, H]
        current_assignment: List[List[List[int]]],  # [B][C][path_length]
        demands: Tensor,                            # [B, C]
    ) -> Tensor:
        """
        各コモディティの現パス上 edge_features を集約し、
        正規化 demand と結合して [B, C, H+1] を返す。
        """
        B, V, _, C, H = edge_features.shape
        device = edge_features.device

        path_feats = torch.zeros(B, C, H, device=device)
        for b in range(B):
            for c in range(C):
                path = current_assignment[b][c]
                if len(path) < 2:
                    continue
                edge_list = []
                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    edge_list.append(edge_features[b, u, v, c, :])
                stacked = torch.stack(edge_list)  # [num_edges, H]
                if self.path_aggregation == 'max':
                    path_feats[b, c] = stacked.max(dim=0).values
                else:
                    path_feats[b, c] = stacked.mean(dim=0)

        # demand 正規化: [B, C] → [B, C, 1]
        demand_norm = (demands / self.demand_max).unsqueeze(-1)

        return torch.cat([path_feats, demand_norm], dim=-1)  # [B, C, H+1]
