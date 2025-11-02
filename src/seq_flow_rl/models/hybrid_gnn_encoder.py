"""
HybridGNNEncoder: GNN Encoder with dynamic edge usage tracking.

This encoder extends the ResidualGatedGCN with support for dynamic edge usage states,
enabling per-commodity or per-step GNN updates during sequential rollout.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from torch import Tensor

from src.gcn.models.gcn_layers import ResidualGatedGCNLayer
from src.common.types import validate_encoder_input_types


class HybridGNNEncoder(nn.Module):
    """
    Hybrid GNN Encoder for SeqFlowRL.

    Key features:
    - Reuses Residual Gated GCN layers from existing GCN implementation
    - Adds dynamic edge usage embedding (capacity + current usage)
    - Supports both static (capacity only) and dynamic (capacity + usage) modes

    Design decision #1: Shared encoder for both Actor and Critic (confirmed)
    """

    def __init__(self, config):
        """
        Args:
            config: Configuration dictionary containing:
                - num_nodes: Number of nodes in the graph
                - num_commodities: Number of commodities
                - hidden_dim: Hidden dimension for GNN layers
                - num_layers: Number of GNN layers
                - aggregation: Aggregation method ('mean', 'sum', etc.)
                - dropout_rate: Dropout rate (default: 0.3)
        """
        super().__init__()

        self.num_nodes = config['num_nodes']
        self.num_commodities = config['num_commodities']
        self.hidden_dim = config['hidden_dim']
        self.num_layers = config['num_layers']
        self.aggregation = config.get('aggregation', 'mean')
        self.dropout_rate = config.get('dropout_rate', 0.3)

        # Node embedding: commodity information (src, dst, demand)
        # voc_nodes_in = num_commodities * 3 (for each commodity: is_src, is_dst, demand)
        self.voc_nodes_in = self.num_commodities * 3
        self.nodes_embedding = nn.Embedding(self.voc_nodes_in, self.hidden_dim // 2)

        # Commodity embedding: demand values
        self.commodities_embedding = nn.Linear(1, self.hidden_dim // 2, bias=False)

        # Edge capacity embedding
        self.edge_capacity_embedding = nn.Linear(1, self.hidden_dim // 2, bias=False)

        # Edge usage embedding (dynamic state)
        self.edge_usage_embedding = nn.Linear(1, self.hidden_dim // 2, bias=False)

        # Residual Gated GCN layers (reuse from existing implementation)
        gcn_layers = []
        for _ in range(self.num_layers):
            gcn_layers.append(ResidualGatedGCNLayer(self.hidden_dim, self.aggregation))
        self.gcn_layers = nn.ModuleList(gcn_layers)

        # Dropout for regularization
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(
        self,
        x_nodes: Tensor,  # [B, V, C] - torch.long (for embedding)
        x_commodities: Tensor,  # [B, C, 3] or [B, C] - torch.float
        x_edges_capacity: Tensor,  # [B, V, V] - torch.float
        x_edges_usage: Optional[Tensor] = None  # [B, V, V] - torch.float (optional)
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass through the GNN encoder.

        Args:
            x_nodes: Node features [B, V, C] - torch.long (0=none, 1=source, 2=target)
                     CRITICAL: Must be torch.long for nn.Embedding layer
            x_commodities: Commodity data [B, C, 3] or [B, C] - torch.float
                          If [B, C, 3], contains (source, target, demand)
            x_edges_capacity: Edge capacity matrix [B, V, V] - torch.float
            x_edges_usage: Edge usage matrix [B, V, V] - torch.float (optional, for dynamic updates)

        Returns:
            node_features: Node embeddings [B, V, C, H] - torch.float
            edge_features: Edge embeddings [B, V, V, C, H] - torch.float
            graph_embedding: Global graph embedding [B, H] - torch.float

        Raises:
            TypeError: If x_nodes is not torch.long
        """
        # Type validation (critical for embedding layer)
        if x_nodes.dtype != torch.long:
            raise TypeError(
                f"x_nodes must be torch.long for embedding layer, got {x_nodes.dtype}. "
                f"Use .long() to convert."
            )

        # Optional: Full validation of all inputs
        encoder_input = {
            'x_nodes': x_nodes,
            'x_commodities': x_commodities,
            'x_edges_capacity': x_edges_capacity,
            'x_edges_usage': x_edges_usage,
        }
        validate_encoder_input_types(encoder_input, strict=False)  # Warning only
        batch_size = x_nodes.shape[0]

        # Handle x_commodities shape: extract demand if it contains [src, dst, demand]
        if len(x_commodities.shape) == 3:
            x_commodities_demand = x_commodities[:, :, 2]  # Extract demand column
        else:
            x_commodities_demand = x_commodities

        # Expand commodities to match node dimensions
        # [B, C] -> [B, V, C]
        x_commodities_expanded = x_commodities_demand.unsqueeze(1).expand(-1, self.num_nodes, -1)

        # Node embedding: Embed node commodity flags
        # x_nodes: [B, V, C] -> embedded: [B, V, C, H//2]
        x_embedded = self.nodes_embedding(x_nodes)

        # Commodity embedding: Embed demand values
        # [B, V, C] -> [B, V, C, 1] -> [B, V, C, H//2]
        c_embedded = self.commodities_embedding(x_commodities_expanded.unsqueeze(-1))

        # Aggregate node embeddings across commodities and expand back
        # [B, V, C, H//2] -> [B, V, H//2] -> [B, V, C, H//2]
        x_aggregate = x_embedded.mean(dim=2).unsqueeze(2).expand(-1, -1, self.num_commodities, -1)

        # Concatenate aggregated node features with commodity embeddings
        # [B, V, C, H//2] + [B, V, C, H//2] -> [B, V, C, H]
        x = torch.cat((x_aggregate, c_embedded), dim=3)

        # Edge embedding: Capacity + Usage
        # Expand edge features to include commodity dimension
        # [B, V, V] -> [B, V, V, C]
        x_edges_capacity_expanded = x_edges_capacity.unsqueeze(-1).expand(-1, -1, -1, self.num_commodities)

        if x_edges_usage is not None:
            # Dynamic mode: Use both capacity and current usage
            x_edges_usage_expanded = x_edges_usage.unsqueeze(-1).expand(-1, -1, -1, self.num_commodities)

            # Embed capacity and usage separately
            # [B, V, V, C] -> [B, V, V, C, 1] -> [B, V, V, C, H//2]
            e_capacity = self.edge_capacity_embedding(x_edges_capacity_expanded.unsqueeze(4))
            e_usage = self.edge_usage_embedding(x_edges_usage_expanded.unsqueeze(4))

            # Concatenate capacity and usage embeddings
            # [B, V, V, C, H//2] + [B, V, V, C, H//2] -> [B, V, V, C, H]
            e = torch.cat((e_capacity, e_usage), dim=4)
        else:
            # Static mode: Use capacity only, duplicate to match hidden_dim
            # [B, V, V, C] -> [B, V, V, C, 1] -> [B, V, V, C, H//2]
            e_capacity = self.edge_capacity_embedding(x_edges_capacity_expanded.unsqueeze(4))
            # Duplicate to match hidden_dim
            e = torch.cat((e_capacity, e_capacity), dim=4)

        # Apply GCN layers
        for layer_idx, gcn_layer in enumerate(self.gcn_layers):
            x, e = gcn_layer(x, e)
            # Apply dropout except for the last layer
            if layer_idx < self.num_layers - 1:
                x = self.dropout(x)
                e = self.dropout(e)

        # x shape: [B, V, C, H]
        # e shape: [B, V, V, C, H]

        # Compute global graph embedding for Critic
        # Average over nodes and commodities: [B, V, C, H] -> [B, H]
        graph_embedding = x.mean(dim=[1, 2])

        return x, e, graph_embedding
