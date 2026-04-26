import torch
import torch.nn as nn
from typing import Tuple
from torch import Tensor

from src.gcn.models.gcn_layers import ResidualGatedGCNLayer


class GNNILSEncoder(nn.Module):
    """
    GNN Encoder for GNN-ILS.

    HybridGNNEncoder と同一構造だが、ILS 固有の入力に対応:
    - x_edges_usage は ILS ループ中に毎ステップ更新される (必須入力)

    内部構成:
        nodes_embedding:         nn.Embedding(C*3, H//2)
        commodities_embedding:   nn.Linear(1, H//2, bias=False)
        edge_capacity_embedding: nn.Linear(1, H//2, bias=False)
        edge_usage_embedding:    nn.Linear(1, H//2, bias=False)
        gcn_layers:              ModuleList[ResidualGatedGCNLayer] * num_layers
    """

    def __init__(self, config: dict):
        """
        Args:
            config:
                - num_nodes: int
                - num_commodities: int
                - hidden_dim: int (default: 128)
                - num_layers: int (default: 8)
                - aggregation: str (default: 'mean')
                - dropout_rate: float (default: 0.3)
        """
        super().__init__()

        self.num_nodes = config['num_nodes']
        self.num_commodities = config['num_commodities']
        self.hidden_dim = config['hidden_dim']
        self.num_layers = config['num_layers']
        self.aggregation = config.get('aggregation', 'mean')
        self.dropout_rate = config.get('dropout_rate', 0.3)

        voc_nodes_in = self.num_commodities * 3
        self.nodes_embedding = nn.Embedding(voc_nodes_in, self.hidden_dim // 2)
        self.commodities_embedding = nn.Linear(1, self.hidden_dim // 2, bias=False)
        self.edge_capacity_embedding = nn.Linear(1, self.hidden_dim // 2, bias=False)
        self.edge_usage_embedding = nn.Linear(1, self.hidden_dim // 2, bias=False)

        self.gcn_layers = nn.ModuleList([
            ResidualGatedGCNLayer(self.hidden_dim, self.aggregation)
            for _ in range(self.num_layers)
        ])
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(
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
        if x_nodes.dtype != torch.long:
            raise TypeError(f"x_nodes must be torch.long, got {x_nodes.dtype}")

        # Commodity demand 抽出
        if len(x_commodities.shape) == 3:
            x_commodities_demand = x_commodities[:, :, 2]  # [B, C]
        else:
            x_commodities_demand = x_commodities

        # ノードへ展開 [B, C] -> [B, V, C]
        x_commodities_expanded = x_commodities_demand.unsqueeze(1).expand(-1, self.num_nodes, -1)

        # Node embedding: [B, V, C] -> [B, V, C, H//2]
        x_embedded = self.nodes_embedding(x_nodes)

        # Commodity embedding: [B, V, C, 1] -> [B, V, C, H//2]
        c_embedded = self.commodities_embedding(x_commodities_expanded.unsqueeze(-1))

        # ノード次元で集約してコモディティ次元へ展開
        x_aggregate = x_embedded.mean(dim=2).unsqueeze(2).expand(-1, -1, self.num_commodities, -1)

        # Node features concat: [B, V, C, H]
        x = torch.cat((x_aggregate, c_embedded), dim=3)

        # Edge features: capacity + usage を [B, V, V, C] に展開
        cap_exp = x_edges_capacity.unsqueeze(-1).expand(-1, -1, -1, self.num_commodities)
        use_exp = x_edges_usage.unsqueeze(-1).expand(-1, -1, -1, self.num_commodities)

        e_capacity = self.edge_capacity_embedding(cap_exp.unsqueeze(4))  # [B, V, V, C, H//2]
        e_usage = self.edge_usage_embedding(use_exp.unsqueeze(4))        # [B, V, V, C, H//2]

        e = torch.cat((e_capacity, e_usage), dim=4)  # [B, V, V, C, H]

        # GCN layers
        for layer_idx, gcn_layer in enumerate(self.gcn_layers):
            x, e = gcn_layer(x, e)
            if layer_idx < self.num_layers - 1:
                x = self.dropout(x)
                e = self.dropout(e)

        # Global graph embedding: [B, V, C, H] -> [B, H]
        graph_embedding = x.mean(dim=[1, 2])

        return x, e, graph_embedding
