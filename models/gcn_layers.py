import torch
import torch.nn.functional as F
import torch.nn as nn

class BatchNormNode(nn.Module):
    """Batch normalization for node features."""

    def __init__(self, hidden_dim):
        super(BatchNormNode, self).__init__()
        self.batch_norm = nn.BatchNorm1d(hidden_dim, track_running_stats=False)

    def forward(self, x):
        """
        Args:
            x: Node features (batch_size, num_nodes, hidden_dim)
        Returns:
            x_bn: Node features after batch normalization (batch_size, num_nodes, hidden_dim)
        """
        x_trans = x.transpose(1, 2)
        print(f'BatchNormNode - x_trans contiguous: {x_trans.is_contiguous()}')  # Check contiguity
        x_trans_bn = self.batch_norm(x_trans)
        x_bn = x_trans_bn.transpose(1, 2)
        print(f'BatchNormNode - x_bn contiguous: {x_bn.is_contiguous()}')  # Check contiguity
        return x_bn


class BatchNormEdge(nn.Module):
    """Batch normalization for edge features."""

    def __init__(self, hidden_dim):
        super(BatchNormEdge, self).__init__()
        self.batch_norm = nn.BatchNorm2d(hidden_dim, track_running_stats=False)

    def forward(self, e):
        """
        Args:
            e: Edge features (batch_size, num_nodes, num_nodes, hidden_dim)
        Returns:
            e_bn: Edge features after batch normalization (batch_size, num_nodes, num_nodes, hidden_dim)
        """
        e_trans = e.transpose(1, 3)
        print(f'BatchNormEdge - e_trans contiguous: {e_trans.is_contiguous()}')  # Check contiguity
        e_trans_bn = self.batch_norm(e_trans)
        e_bn = e_trans_bn.transpose(1, 3)
        print(f'BatchNormEdge - e_bn contiguous: {e_bn.is_contiguous()}')  # Check contiguity
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
        print(f'NodeFeatures - Ux contiguous: {Ux.is_contiguous()}')  # Check contiguity
        print(f'NodeFeatures - Vx contiguous: {Vx.is_contiguous()}')  # Check contiguity
        print(f'NodeFeatures - gateVx contiguous: {gateVx.is_contiguous()}')  # Check contiguity
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
            x: Node features (batch_size, num_nodes, hidden_dim)
            e: Edge features (batch_size, num_nodes, num_nodes, hidden_dim)
        Returns:
            e_new: Convolved edge features (batch_size, num_nodes, num_nodes, hidden_dim)
        """
        Ue = self.U(e)
        Vx = self.V(x).unsqueeze(1)
        Vx = Vx.unsqueeze(2)
        print(f'EdgeFeatures - Ue contiguous: {Ue.is_contiguous()}')  # Check contiguity
        print(f'EdgeFeatures - Vx contiguous: {Vx.is_contiguous()}')  # Check contiguity
        e_new = Ue + Vx + Vx
        return e_new


class ResidualGatedGCNLayer(nn.Module):
    """Convnet layer with gating and residual connection."""

    def __init__(self, hidden_dim, aggregation="sum"):
        super(ResidualGatedGCNLayer, self).__init__()
        self.node_feat = NodeFeatures(hidden_dim, aggregation)
        self.edge_feat = EdgeFeatures(hidden_dim)
        self.bn_node = BatchNormNode(hidden_dim)
        self.bn_edge = BatchNormEdge(hidden_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, e):
        """
        Args:
            x: Node features (batch_size, num_nodes, hidden_dim)
            e: Edge features (batch_size, num_nodes, num_nodes, hidden_dim)
        Returns:
            x_new: Convolved node features (batch_size, num_nodes, hidden_dim)
            e_new: Convolved edge features (batch_size, num_nodes, num_nodes, hidden_dim)
        """
        e_in = e
        x_in = x
        e_tmp = self.edge_feat(x_in, e_in)
        edge_gate = torch.sigmoid(e_tmp)
        x_tmp = self.node_feat(x_in, edge_gate)
        e_tmp = self.bn_edge(e_tmp)
        x_tmp = self.bn_node(x_tmp)
        print(f'ResidualGatedGCNLayer - e_tmp contiguous: {e_tmp.is_contiguous()}')  # Check contiguity
        print(f'ResidualGatedGCNLayer - x_tmp contiguous: {x_tmp.is_contiguous()}')  # Check contiguity
        e_new = self.relu(e_tmp) + e_in
        x_new = self.relu(x_tmp) + x_in
        return x_new, e_new


class MLP(nn.Module):
    """Multi-layer Perceptron for output prediction."""

    def __init__(self, hidden_dim, output_dim, num_layers=2):
        super(MLP, self).__init__()
        layers = [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 1)]
        self.layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Args:
            x: Input features (batch_size, hidden_dim)
        Returns:
            y: Output predictions (batch_size, output_dim)
        """
        for layer in self.layers:
            x = layer(x)
            print(f'MLP - x after layer contiguous: {x.is_contiguous()}')  # Check contiguity
            x = self.relu(x)
        y = self.output_layer(x)
        print(f'MLP - y contiguous: {y.is_contiguous()}')  # Check contiguity
        return y
