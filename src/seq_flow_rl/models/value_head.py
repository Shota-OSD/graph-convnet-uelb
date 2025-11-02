"""
ValueHead: Critic network for state value estimation.

Implements global value function that considers the entire network state.
"""

import torch
import torch.nn as nn

from src.gcn.models.gcn_layers import MLP


class ValueHead(nn.Module):
    """
    Value Head for SeqFlowRL Critic.

    Design decision #2-B: Global Value (confirmed)
    - Takes the entire graph state (all nodes × all commodities)
    - Outputs a single scalar value V(s)
    - Aligns with the objective: minimizing global load factor
    """

    def __init__(self, hidden_dim, num_nodes, num_commodities, mlp_layers=3):
        """
        Args:
            hidden_dim: Hidden dimension from GNN encoder
            num_nodes: Number of nodes in the graph
            num_commodities: Number of commodities
            mlp_layers: Number of MLP layers (default: 3)
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.num_commodities = num_commodities
        self.mlp_layers = mlp_layers

        # Global value: Flatten all node × commodity features
        # Input dimension: num_nodes * num_commodities * hidden_dim
        input_dim = num_nodes * num_commodities * hidden_dim

        # Value MLP: Gradually reduce dimensions: input_dim -> 512 -> 256 -> 128 -> 1
        # This prevents excessive parameters when input_dim is large
        if mlp_layers == 4:
            hidden_dims = [512, 256, 128]
        elif mlp_layers == 3:
            hidden_dims = [512, 256]
        elif mlp_layers == 2:
            hidden_dims = [256]
        else:
            hidden_dims = []

        self.value_mlp = MLP(
            input_dim,
            1,  # Single scalar value output
            num_layers=mlp_layers,
            hidden_dims=hidden_dims
        )

    def forward(self, node_features):
        """
        Compute state value V(s) from node features.

        Args:
            node_features: Node embeddings [B, V, C, H]

        Returns:
            state_value: Scalar value for each state [B, 1] or [B]
        """
        batch_size = node_features.shape[0]

        # Flatten all node and commodity features: [B, V, C, H] -> [B, V*C*H]
        flattened_features = node_features.reshape(batch_size, -1)

        # Predict state value: [B, V*C*H] -> [B, 1]
        state_value = self.value_mlp(flattened_features)

        # Squeeze to [B] for easier loss computation
        state_value = state_value.squeeze(-1)

        return state_value

    def forward_graph_embedding(self, graph_embedding):
        """
        Alternative forward using pre-computed graph embedding.

        This is more efficient if the encoder already provides a global embedding.

        Args:
            graph_embedding: Global graph embedding [B, H]

        Returns:
            state_value: Scalar value for each state [B]
        """
        # For future optimization: Use graph_embedding instead of full node_features
        # This would require a smaller value_mlp with input_dim = hidden_dim
        raise NotImplementedError(
            "This method is for future optimization. "
            "Currently using full node features for global value computation."
        )


class PerCommodityValueHead(nn.Module):
    """
    Alternative Value Head: Per-Commodity Value (experimental).

    This is NOT used in the confirmed design but provided for comparison experiments.
    Design decision #2-B: Global Value is confirmed, this is for experiments only.
    """

    def __init__(self, hidden_dim, num_nodes, num_commodities, mlp_layers=3):
        """
        Args:
            hidden_dim: Hidden dimension from GNN encoder
            num_nodes: Number of nodes in the graph
            num_commodities: Number of commodities
            mlp_layers: Number of MLP layers
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.num_commodities = num_commodities

        # Per-commodity value: Each commodity gets its own value
        # Input: node features for a specific commodity
        input_dim = num_nodes * hidden_dim

        # Separate MLP for each commodity (not parameter efficient)
        # Gradually reduce dimensions for each commodity
        if mlp_layers == 3:
            hidden_dims = [256, 128]
        elif mlp_layers == 2:
            hidden_dims = [128]
        else:
            hidden_dims = []

        self.commodity_value_mlps = nn.ModuleList([
            MLP(input_dim, 1, num_layers=mlp_layers, hidden_dims=hidden_dims)
            for _ in range(num_commodities)
        ])

    def forward(self, node_features):
        """
        Compute per-commodity values and aggregate to global value.

        Args:
            node_features: Node embeddings [B, V, C, H]

        Returns:
            state_value: Aggregated scalar value [B]
            commodity_values: Per-commodity values [B, C] (for analysis)
        """
        batch_size = node_features.shape[0]
        commodity_values = []

        for c in range(self.num_commodities):
            # Extract features for commodity c: [B, V, H]
            commodity_feat = node_features[:, :, c, :]
            # Flatten: [B, V*H]
            commodity_feat_flat = commodity_feat.reshape(batch_size, -1)
            # Predict value for commodity c: [B, 1]
            value_c = self.commodity_value_mlps[c](commodity_feat_flat)
            commodity_values.append(value_c)

        # Stack per-commodity values: [B, C]
        commodity_values = torch.cat(commodity_values, dim=1)

        # Aggregate to global value (sum or mean)
        state_value = commodity_values.sum(dim=1)

        return state_value, commodity_values
