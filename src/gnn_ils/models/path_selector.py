import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from torch import Tensor

from src.gcn.models.gcn_layers import MLP


class PathSelector(nn.Module):
    """
    Level2 Policy: パス選択。

    選択されたコモディティに対し、候補パスプールから最適なパスを選択する。
    各候補パスについて concat(cand_feat [H], current_feat [H], demand_norm [1]) -> [2H+1]
    を MLP に通してスコアを計算する。

    入力処理:
        各候補パスについて: edge_features[b, u, v, c, :] を aggregation → [H]
        concat(cand_feat [H], current_feat [H], demand_norm [1]) → [2H+1] → MLP → スコア
        masked softmax → probabilities [B, max_paths]
    """

    def __init__(self, hidden_dim: int, max_paths: int, mlp_layers: int = 2,
                 path_aggregation: str = 'mean', dropout_rate: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_paths = max_paths
        self.path_aggregation = path_aggregation

        # 漏斗型: 入力次元(2H+1)に近い層から段階的に絞る
        if mlp_layers >= 3:
            hidden_dims = [hidden_dim * 2, 64][:mlp_layers - 1]
        elif mlp_layers == 2:
            # 従来は [128] (257→128→1) だが漏斗型原則に合わせて変更。後方互換なし
            hidden_dims = [128]
        else:
            hidden_dims = []
        self.path_score_mlp = MLP(
            hidden_dim * 2 + 1,
            1,
            num_layers=mlp_layers,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
        )

    def forward(
        self,
        edge_features: Tensor,                       # [B, V, V, C, H]
        selected_commodity: Tensor,                  # [B] - int
        candidate_paths: List[List[List[int]]],      # [B][P_c][path_length]
        current_paths: List[List[int]],              # [B][path_length] (選択コモディティの現パス)
        demands: Tensor,                             # [B] (正規化済み demand)
        path_mask: Tensor,                           # [B, max_paths] - bool
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Returns:
            action_probs: [B, max_paths]
            log_probs:    [B, max_paths]
            entropy:      [B]
        """
        B = edge_features.shape[0]
        neg_inf = torch.tensor(float('-inf'), device=edge_features.device)

        batch_scores = []
        for b in range(B):
            c_idx = selected_commodity[b].item()
            paths = candidate_paths[b]  # List[List[int]]
            current_feat = self._encode_path(edge_features, current_paths[b], c_idx, b)  # [H]
            demand_val = demands[b].unsqueeze(0)  # [1]

            path_scores = []
            for p_idx in range(self.max_paths):
                if p_idx < len(paths):
                    cand_feat = self._encode_path(edge_features, paths[p_idx], c_idx, b)  # [H]
                    inp = torch.cat([cand_feat, current_feat, demand_val], dim=0).unsqueeze(0)  # [1, 2H+1]
                    path_scores.append(self.path_score_mlp(inp).squeeze())
                else:
                    path_scores.append(neg_inf)
            batch_scores.append(torch.stack(path_scores))
        scores = torch.stack(batch_scores)  # [B, max_paths] - maintains grad_fn

        # パスマスク適用
        scores = scores.masked_fill(~path_mask, float('-inf'))

        action_probs = F.softmax(scores, dim=-1)
        log_probs = F.log_softmax(scores, dim=-1)

        # masked 位置では log_probs = -inf なので torch.where で 0 に置換し 0*(-inf)=NaN を防ぐ
        log_probs_safe = torch.where(path_mask, log_probs, torch.zeros_like(log_probs))
        entropy = -(action_probs * log_probs_safe).sum(dim=-1)  # [B]

        return action_probs, log_probs, entropy

    def _encode_path(
        self,
        edge_features: Tensor,  # [B, V, V, C, H]
        path: List[int],
        commodity_idx: int,
        batch_idx: int,
    ) -> Tensor:
        """
        パスをエッジ特徴量の mean-pooling で表現する。

        path 上の各エッジ (u, v) の edge_features[b, u, v, c, :] を
        mean-pooling してパス全体の特徴量 [H] を得る。
        """
        if len(path) < 2:
            return torch.zeros(self.hidden_dim, device=edge_features.device)

        edge_feats = []
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            edge_feats.append(edge_features[batch_idx, u, v, commodity_idx, :])

        stacked = torch.stack(edge_feats)  # [num_edges, H]
        if self.path_aggregation == 'max':
            return stacked.max(dim=0).values
        return stacked.mean(dim=0)  # [H]
