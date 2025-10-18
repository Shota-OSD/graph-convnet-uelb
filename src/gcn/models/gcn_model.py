import torch
import torch.nn.functional as F
import torch.nn as nn

from .gcn_layers import ResidualGatedGCNLayer, MLP
from .model_utils import *

class ResidualGatedGCNModel(nn.Module):
    """Residual Gated GCN Model for outputting predictions as edge adjacency matrices.

    References:
        Paper: https://arxiv.org/pdf/1711.07553v2.pdf
        Code: https://github.com/xbresson/spatial_graph_convnets
    """

    def __init__(self, config, dtypeFloat, dtypeLong):
        super().__init__()
        self.dtypeFloat = dtypeFloat
        self.dtypeLong = dtypeLong
        # Define net parameters
        self.num_nodes = config.num_nodes
        self.num_commodities = config.num_commodities
        self.voc_nodes_in = config.num_commodities * 3
        self.voc_nodes_out = config['num_nodes']  # config['voc_nodes_out']
        self.voc_edges_in = config['voc_edges_in']
        self.voc_edges_out = config['voc_edges_out']
        self.hidden_dim = config['hidden_dim']
        self.num_layers = config['num_layers']
        self.mlp_layers = config['mlp_layers']
        self.aggregation = config['aggregation']
        self.dropout_rate = config.get('dropout_rate', 0.2)
        # Node and edge embedding layers/lookups
        self.nodes_embedding = nn.Embedding(self.voc_nodes_in, self.hidden_dim // 2)
        self.commodities_embedding = nn.Linear(1, self.hidden_dim // 2, bias=False)
        self.edges_values_embedding = nn.Linear(1, self.hidden_dim, bias=False)
        # Define GCN Layers
        gcn_layers = []
        for layer in range(self.num_layers):
            gcn_layers.append(ResidualGatedGCNLayer(self.hidden_dim, self.aggregation))
        self.gcn_layers = nn.ModuleList(gcn_layers)
        # Define MLP classifiers
        self.mlp_edges = MLP(self.hidden_dim, self.voc_edges_out, self.mlp_layers)
        # self.mlp_nodes = MLP(self.hidden_dim, self.voc_nodes_out, self.mlp_layers)

    def forward(self, x_edges, x_commodities, x_edges_capacity, x_nodes, y_edges=None, edge_cw=None, compute_loss=True):
        """
        Forward pass through the GCN model.

        Args:
            x_edges: Input edge adjacency matrix (batch_size, num_nodes, num_nodes)
            x_commodities: Input edge adjacency matrix (batch_size, num_commodities)
            x_edges_capacity: Input edge capacity matrix (batch_size, num_nodes, num_nodes)
            x_nodes: Input node with commodity information (batch_size, num_nodes, num_commodities)
            y_edges: Targets for edges (batch_size, num_edges, num_commodities) - optional
            edge_cw: Class weights for edges loss - optional
            compute_loss: Whether to compute loss (requires y_edges and edge_cw)

        Returns:
            y_pred_edges: Predictions for edges (batch_size, num_nodes, num_nodes, num_commodities)
            loss: Value of loss function (None if compute_loss=False)

        # Nomalize node and edge capacity
        x_edges_capacity_min = x_edges_capacity.min()
        x_edges_capacity_max = x_edges_capacity.max()
        normalized_x_edges_capacity = (x_edges_capacity - x_edges_capacity_min) / (x_edges_capacity_max - x_edges_capacity_min)
        """
        # Features embedding
        x_edges_capacity_expanded = x_edges_capacity.unsqueeze(-1).expand(-1, -1, -1, self.num_commodities)
        x_commodities_expanded = x_commodities.unsqueeze(1).expand(-1, self.num_nodes, -1)

        x_embedded = self.nodes_embedding(x_nodes)
        c = self.commodities_embedding(x_commodities_expanded.unsqueeze(-1))
        e = self.edges_values_embedding(x_edges_capacity_expanded.unsqueeze(4))

        x_aggregate = x_embedded.mean(dim=2).unsqueeze(2).expand(-1, -1, self.num_commodities, -1)

        x = torch.cat((x_aggregate, c), 3)
        # GCN layers
        for layer in range(self.num_layers):
            x, e = self.gcn_layers[layer](x.contiguous(), e.contiguous())  # B x V x C x H, B x V x V x C x H
        # MLP classifier
        y_pred_edges = self.mlp_edges(e)

        # Optionally compute loss
        loss = None
        if compute_loss:
            if y_edges is None or edge_cw is None:
                raise ValueError("y_edges and edge_cw required when compute_loss=True")

            edge_cw = torch.tensor(edge_cw, dtype=self.dtypeFloat)  # Convert to tensors
            # Move edge_cw to the same device as y_pred_edges
            edge_cw = edge_cw.to(y_pred_edges.device)
            loss = loss_edges(y_pred_edges, y_edges, edge_cw)

        return y_pred_edges, loss
