import torch
import torch.nn as nn
from torch import Tensor

from src.gcn.models.gcn_layers import MLP


class ILSValueHead(nn.Module):
    """
    Critic for GNN-ILS。

    グラフ全体の状態から単一スカラー V(s) を予測する。

    use_graph_embedding=False (デフォルト):
        node_features [B, V, C, H] を flatten → MLP(V*C*H, 1)
    use_graph_embedding=True:
        graph_embedding [B, H] → MLP(H, 1)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_nodes: int,
        num_commodities: int,
        mlp_layers: int = 3,
        use_graph_embedding: bool = False,
    ):
        super().__init__()
        self.use_graph_embedding = use_graph_embedding

        if use_graph_embedding:
            input_dim = hidden_dim
            if mlp_layers >= 3:
                hidden_dims = [256, 128][:mlp_layers - 1]
            elif mlp_layers == 2:
                hidden_dims = [128]
            else:
                hidden_dims = []
        else:
            input_dim = num_nodes * num_commodities * hidden_dim
            if mlp_layers >= 3:
                hidden_dims = [512, 256][:mlp_layers - 1]
            elif mlp_layers == 2:
                hidden_dims = [256]
            else:
                hidden_dims = []

        self.value_mlp = MLP(input_dim, 1, num_layers=mlp_layers, hidden_dims=hidden_dims)

    def forward(
        self,
        node_features: Tensor,          # [B, V, C, H]
        graph_embedding: Tensor = None, # [B, H] (use_graph_embedding=True の場合)
    ) -> Tensor:
        """
        Returns:
            state_value: [B]
        """
        if self.use_graph_embedding:
            state_value = self.value_mlp(graph_embedding)
        else:
            B = node_features.shape[0]
            flat = node_features.reshape(B, -1)
            state_value = self.value_mlp(flat)

        return state_value.squeeze(-1)  # [B]
