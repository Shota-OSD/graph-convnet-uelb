import torch
import torch.nn.functional as F
import torch.nn as nn

class BatchNormNode(nn.Module):
    """Batch normalization for node features."""

    def __init__(self, hidden_dim):
        super(BatchNormNode, self).__init__()
        self.batch_norm = nn.BatchNorm2d(hidden_dim, track_running_stats=False)

    def forward(self, x):
        """
        Args:
            x: Node features (batch_size, num_nodes, num_commodities, hidden_dim)
        Returns:
            x_bn: Node features after batch normalization (batch_size, num_nodes, hidden_dim)
        """
        x_trans = x.transpose(1, 3)
        x_trans_bn = self.batch_norm(x_trans)
        x_bn = x_trans_bn.transpose(1, 3)
        return x_bn


class BatchNormEdge(nn.Module):
    """Batch normalization for edge features."""

    def __init__(self, hidden_dim):
        super(BatchNormEdge, self).__init__()
        self.batch_norm = nn.BatchNorm3d(hidden_dim, track_running_stats=False)

    def forward(self, e):
        """
        Args:
            e: Edge features (batch_size, num_nodes, num_nodes, num_commodities, hidden_dim)
        Returns:
            e_bn: Edge features after batch normalization (batch_size, num_nodes, num_nodes, hidden_dim)
        """
        e_trans = e.transpose(1, 4)
        e_trans_bn = self.batch_norm(e_trans)
        e_bn = e_trans_bn.transpose(1, 4)

        return e_bn


class NodeFeatures(nn.Module):
    """Convnet features for nodes."""

    def __init__(self, hidden_dim, aggregation="mean"):
        super(NodeFeatures, self).__init__()
        self.aggregation = aggregation
        self.U = nn.Linear(hidden_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, edge_gate):
        """
        Args:
            x: Node features (batch_size, num_nodes, hidden_dim)
            edge_gate: Edge gate values (batch_size, num_nodes, num_nodes, hidden_dim)
        Returns:
            x_new: Convolved node features (batch_size, num_nodes, hidden_dim)
        """
        Ux = self.U(x)
        Vx = self.V(x).unsqueeze(1)
        gateVx = edge_gate * Vx
        if self.aggregation == "mean":
            x_new = Ux + torch.sum(gateVx, dim=2) / (1e-20 + torch.sum(edge_gate, dim=2))
        elif self.aggregation == "sum":
            x_new = Ux + torch.sum(gateVx, dim=2)
        return x_new


class EdgeFeatures(nn.Module):
    """Convnet features for edges."""

    def __init__(self, hidden_dim):
        super(EdgeFeatures, self).__init__()
        self.U = nn.Linear(hidden_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, e):
        """
        Args:
            x: Node features (batch_size, num_nodes, num_commodities, hidden_dim)
            e: Edge features (batch_size, num_nodes, num_nodes, num_commodities, hidden_dim)
        Returns:
            e_new: Convolved edge features (batch_size, num_nodes, num_nodes, hidden_dim)
        """
        Ue = self.U(e)
        Vx = self.V(x)
        Wx = Vx.unsqueeze(1)  # Extend Vx from "B x V x H" to "B x V x 1 x H"
        Vx = Vx.unsqueeze(2)  # extend Vx from "B x V x H" to "B x 1 x V x H"
        e_new = Ue + Vx + Wx
        return e_new


class ResidualGatedGCNLayer(nn.Module):
    """Convnet layer with gating and residual connection."""

    def __init__(self, hidden_dim, aggregation="sum", dropout_rate=0.3):
        super().__init__()
        self.node_feat = NodeFeatures(hidden_dim, aggregation)
        self.edge_feat = EdgeFeatures(hidden_dim)
        self.bn_node = BatchNormNode(hidden_dim)
        self.bn_edge = BatchNormEdge(hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)
    def forward(self, x, e):
        """
        Args:
            x: Node features (batch_size, num_nodes, num_commodities, hidden_dim)
            e: Edge features (batch_size, num_nodes, num_nodes, num_commodities, hidden_dim)
        Returns:
            x_new: Convolved node features (batch_size, num_nodes, num_commodities, hidden_dim)
            e_new: Convolved edge features (batch_size, num_nodes, num_nodes, num_commodities, hidden_dim)
        """
        e_in = e
        x_in = x
        e_tmp = self.edge_feat(x_in, e_in)
        edge_gate = torch.sigmoid(e_tmp)
        x_tmp = self.node_feat(x_in, edge_gate)
        e_tmp = self.bn_edge(e_tmp)
        x_tmp = self.bn_node(x_tmp)
        e_new = self.relu(e_tmp) + e_in
        x_new = self.relu(x_tmp) + x_in
        # Apply dropout
        e_new = self.dropout(e_new)
        x_new = self.dropout(x_new)
        return x_new, e_new


class MLP(nn.Module):
    """Multi-layer Perceptron for output prediction."""

    def __init__(self, input_dim, output_dim, num_layers=2, hidden_dims=None, dropout_rate=0.2):
        """
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            num_layers: Number of layers (default: 2)
            hidden_dims: List of hidden dimensions for intermediate layers.
                        If None, uses input_dim for all intermediate layers (backward compatible).
                        If provided, should have length (num_layers - 1).
            dropout_rate: Dropout rate (default: 0.2)
        """
        super().__init__()

        # Backward compatibility: if hidden_dims not provided, use input_dim for all layers
        if hidden_dims is None:
            hidden_dims = [input_dim] * (num_layers - 1)

        # Validate hidden_dims length
        if len(hidden_dims) != num_layers - 1:
            raise ValueError(f"hidden_dims length ({len(hidden_dims)}) must be num_layers - 1 ({num_layers - 1})")

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        self.layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, output_dim)

    def forward(self, x):
        """
        Args:
            x: Input features (batch_size, input_dim)
        Returns:
            y: Output predictions (batch_size, output_dim)
        """
        x = self.layers(x)
        y = self.output_layer(x)
        return y