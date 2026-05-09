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

    def _batch_encode_paths(
        self,
        edge_features: Tensor,       # [B, V, V, C, H]
        paths_list: List[List[int]],  # flat list of paths (total N paths)
        commodity_indices: List[int], # commodity index for each path (length N)
        batch_indices: List[int],     # batch index for each path (length N)
        num_output: int,              # total number of output slots
        output_indices: List[int],    # which output slot each path maps to (length N)
    ) -> Tensor:
        """
        複数パスのエッジ特徴量を一括取得し集約する。

        Returns:
            path_features: [num_output, H]
        """
        H = self.hidden_dim
        device = edge_features.device

        if not paths_list:
            return torch.zeros(num_output, H, device=device)

        max_edges = max(len(p) - 1 for p in paths_list)
        if max_edges <= 0:
            return torch.zeros(num_output, H, device=device)

        N = len(paths_list)

        # インデックステンソル構築
        src_idx = torch.zeros(N, max_edges, dtype=torch.long, device=device)
        dst_idx = torch.zeros(N, max_edges, dtype=torch.long, device=device)
        mask = torch.zeros(N, max_edges, dtype=torch.bool, device=device)

        for i, path in enumerate(paths_list):
            n_edges = len(path) - 1
            for j in range(n_edges):
                src_idx[i, j] = path[j]
                dst_idx[i, j] = path[j + 1]
                mask[i, j] = True

        # バッチ・コモディティインデックス [N, max_edges]
        b_t = torch.tensor(batch_indices, dtype=torch.long, device=device).unsqueeze(1).expand(N, max_edges)
        c_t = torch.tensor(commodity_indices, dtype=torch.long, device=device).unsqueeze(1).expand(N, max_edges)

        # 一括 indexing [N, max_edges, H]
        all_feats = edge_features[b_t, src_idx, dst_idx, c_t, :]

        # マスク付き集約 [N, H]
        if self.path_aggregation == 'max':
            all_feats = all_feats.masked_fill(~mask.unsqueeze(-1), float('-inf'))
            encoded = all_feats.max(dim=1).values
            no_edges = ~mask.any(dim=1)
            encoded = encoded.masked_fill(no_edges.unsqueeze(-1), 0.0)
        else:
            all_feats = all_feats * mask.unsqueeze(-1).float()
            edge_counts = mask.sum(dim=1, keepdim=True).clamp(min=1).float()
            encoded = all_feats.sum(dim=1) / edge_counts

        # 出力スロットに配置 [num_output, H]
        result = torch.zeros(num_output, H, device=device)
        out_t = torch.tensor(output_indices, dtype=torch.long, device=device)
        result[out_t] = encoded

        return result

    def forward(
        self,
        edge_features: Tensor,                       # [B, V, V, C, H]
        selected_commodity: Tensor,                  # [B] - int
        candidate_paths: List[List[List[int]]],      # [B][P_c][path_length]
        current_paths: List[List[int]],              # [B][path_length]
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
        H = self.hidden_dim

        # --- 1. 現パスの一括エンコード [B, H] ---
        curr_paths_flat = []
        curr_c_indices = []
        curr_b_indices = []
        curr_out_indices = []
        for b in range(B):
            c_idx = selected_commodity[b].item()
            path = current_paths[b]
            if len(path) >= 2:
                curr_paths_flat.append(path)
                curr_c_indices.append(c_idx)
                curr_b_indices.append(b)
                curr_out_indices.append(b)

        current_feats = self._batch_encode_paths(
            edge_features, curr_paths_flat, curr_c_indices, curr_b_indices,
            num_output=B, output_indices=curr_out_indices
        )  # [B, H]

        # --- 2. 候補パスの一括エンコード [B * max_paths, H] ---
        cand_paths_flat = []
        cand_c_indices = []
        cand_b_indices = []
        cand_out_indices = []
        for b in range(B):
            c_idx = selected_commodity[b].item()
            paths = candidate_paths[b]
            for p_idx in range(min(len(paths), self.max_paths)):
                path = paths[p_idx]
                if len(path) >= 2:
                    cand_paths_flat.append(path)
                    cand_c_indices.append(c_idx)
                    cand_b_indices.append(b)
                    cand_out_indices.append(b * self.max_paths + p_idx)

        cand_feats = self._batch_encode_paths(
            edge_features, cand_paths_flat, cand_c_indices, cand_b_indices,
            num_output=B * self.max_paths, output_indices=cand_out_indices
        ).view(B, self.max_paths, H)  # [B, max_paths, H]

        # --- 3. MLP 一括 forward ---
        current_expanded = current_feats.unsqueeze(1).expand(B, self.max_paths, H)  # [B, max_paths, H]
        demand_expanded = demands.view(B, 1, 1).expand(B, self.max_paths, 1)  # [B, max_paths, 1]

        mlp_input = torch.cat([cand_feats, current_expanded, demand_expanded], dim=-1)  # [B, max_paths, 2H+1]
        mlp_input_flat = mlp_input.view(B * self.max_paths, 2 * H + 1)

        scores_flat = self.path_score_mlp(mlp_input_flat).squeeze(-1)  # [B * max_paths]
        scores = scores_flat.view(B, self.max_paths)  # [B, max_paths]

        # --- 4. マスク適用 + softmax ---
        scores = scores.masked_fill(~path_mask, float('-inf'))

        action_probs = F.softmax(scores, dim=-1)
        log_probs = F.log_softmax(scores, dim=-1)

        log_probs_safe = torch.where(path_mask, log_probs, torch.zeros_like(log_probs))
        entropy = -(action_probs * log_probs_safe).sum(dim=-1)  # [B]

        return action_probs, log_probs, entropy
