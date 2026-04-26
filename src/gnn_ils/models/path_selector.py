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
    パス特徴量はパス上のエッジ特徴量の mean-pooling で表現する。

    入力処理:
        各候補パスについて: edge_features[b, u, v, c, :] を mean-pool → [H]
        path_feat [H] + graph_embedding [H] → concat [2H] → MLP → スコア
        masked softmax → probabilities [B, max_paths]
    """

    def __init__(self, hidden_dim: int, max_paths: int, mlp_layers: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_paths = max_paths

        hidden_dims = [128] * (mlp_layers - 1)
        self.path_score_mlp = MLP(
            hidden_dim * 2,
            1,
            num_layers=mlp_layers,
            hidden_dims=hidden_dims,
        )

    def forward(
        self,
        edge_features: Tensor,                       # [B, V, V, C, H]
        graph_embedding: Tensor,                     # [B, H]
        selected_commodity: Tensor,                  # [B] - int
        candidate_paths: List[List[List[int]]],      # [B][P_c][path_length]
        path_mask: Tensor,                           # [B, max_paths] - bool
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Returns:
            action_probs: [B, max_paths]
            log_probs:    [B, max_paths]
            entropy:      [B]
        """
        B = edge_features.shape[0]
        H = self.hidden_dim
        scores = torch.full((B, self.max_paths), float('-inf'), device=edge_features.device)

        for b in range(B):
            c_idx = selected_commodity[b].item()
            paths = candidate_paths[b]  # List[List[int]]
            g_emb = graph_embedding[b]  # [H]

            for p_idx, path in enumerate(paths):
                if p_idx >= self.max_paths:
                    break
                path_feat = self._encode_path(edge_features, path, c_idx, b)  # [H]
                inp = torch.cat([path_feat, g_emb], dim=0).unsqueeze(0)       # [1, 2H]
                scores[b, p_idx] = self.path_score_mlp(inp).squeeze()

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

        return torch.stack(edge_feats).mean(dim=0)  # [H]
