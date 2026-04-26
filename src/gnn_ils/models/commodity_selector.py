import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from torch import Tensor

from src.gcn.models.gcn_layers import MLP


class CommoditySelector(nn.Module):
    """
    Level1 Policy: コモディティ選択。

    各コモディティのノード集約特徴量 + グラフ埋め込みから
    交換対象コモディティを選択する。

    入力処理:
        node_features [B, V, C, H] → mean over V → [B, C, H]
        graph_embedding [B, H] → expand → [B, C, H]
        concat → [B, C, 2H] → MLP per commodity → scores [B, C]
        masked softmax → probabilities [B, C]
    """

    def __init__(self, hidden_dim: int, num_commodities: int, mlp_layers: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_commodities = num_commodities

        hidden_dims = [128] * (mlp_layers - 1)
        self.commodity_mlp = MLP(
            hidden_dim * 2,
            1,
            num_layers=mlp_layers,
            hidden_dims=hidden_dims,
        )

    def forward(
        self,
        node_features: Tensor,   # [B, V, C, H]
        graph_embedding: Tensor, # [B, H]
        commodity_mask: Tensor,  # [B, C] - bool (True = 交換可能)
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Returns:
            action_probs: [B, C]
            log_probs:    [B, C]
            entropy:      [B]
        """
        B, C = commodity_mask.shape

        # コモディティ毎のノード集約特徴量: [B, V, C, H] → [B, C, H]
        commodity_feats = node_features.mean(dim=1)  # [B, C, H]

        # グラフ埋め込みをコモディティ次元に展開: [B, H] → [B, C, H]
        graph_exp = graph_embedding.unsqueeze(1).expand(-1, C, -1)

        # 結合: [B, C, 2H]
        combined = torch.cat([commodity_feats, graph_exp], dim=-1)

        # MLP でスコア算出: [B*C, 2H] → [B*C, 1] → [B, C]
        scores = self.commodity_mlp(combined.view(B * C, -1)).view(B, C)

        # マスク適用 (交換不可コモディティを -inf に)
        scores = scores.masked_fill(~commodity_mask, float('-inf'))

        # 確率分布
        action_probs = F.softmax(scores, dim=-1)
        log_probs = F.log_softmax(scores, dim=-1)

        # エントロピー: -sum(p * log p) over valid actions
        # masked 位置では log_probs = -inf なので torch.where で 0 に置換し 0*(-inf)=NaN を防ぐ
        log_probs_safe = torch.where(commodity_mask, log_probs, torch.zeros_like(log_probs))
        entropy = -(action_probs * log_probs_safe).sum(dim=-1)  # [B]

        return action_probs, log_probs, entropy
