import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from torch import Tensor

from src.gcn.models.gcn_layers import MLP


class CommoditySelector(nn.Module):
    """
    Level1 Policy: コモディティ選択。

    各コモディティの現パス上エッジ特徴量の mean-pooling + demand から
    交換対象コモディティを選択する。

    入力処理:
        path_commodity_features [B, C, H+1] (呼び出し側で構築済み)
        → MLP per commodity → scores [B, C]
        masked softmax → probabilities [B, C]
    """

    def __init__(self, hidden_dim: int, num_commodities: int, mlp_layers: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_commodities = num_commodities

        hidden_dims = [128] * (mlp_layers - 1)
        self.commodity_mlp = MLP(
            hidden_dim + 1,
            1,
            num_layers=mlp_layers,
            hidden_dims=hidden_dims,
        )

    def forward(
        self,
        path_commodity_features: Tensor,  # [B, C, H+1]
        commodity_mask: Tensor,           # [B, C] - bool (True = 交換可能)
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Returns:
            action_probs: [B, C]
            log_probs:    [B, C]
            entropy:      [B]
        """
        B, C = commodity_mask.shape

        # MLP でスコア算出: [B*C, H+1] → [B*C, 1] → [B, C]
        scores = self.commodity_mlp(path_commodity_features.view(B * C, -1)).view(B, C)

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
