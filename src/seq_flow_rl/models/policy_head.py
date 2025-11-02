"""
PolicyHead: Actor network for action selection.

Supports both node-level and edge-level action spaces with unified interface.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.gcn.models.gcn_layers import MLP


class PolicyHead(nn.Module):
    """
    Policy Head for SeqFlowRL Actor.

    Supports two action space types:
    - 'node': Node-level actions (select next node given current node)
    - 'edge': Edge-level actions (score all edges from current node)

    Design decisions:
    - #2-A: Both action types implemented, 'node' is default (confirmed)
    - Unified output format [B, V] for both types
    """

    def __init__(self, hidden_dim, num_nodes, action_type='node', mlp_layers=2):
        """
        Args:
            hidden_dim: Hidden dimension from GNN encoder
            num_nodes: Number of nodes in the graph
            action_type: Type of action space ('node' or 'edge')
            mlp_layers: Number of MLP layers (default: 2)
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.action_type = action_type
        self.mlp_layers = mlp_layers

        if action_type == 'node':
            # Node-level: Next node selection
            # Input: concatenated context (current_node_feat + dst_node_feat)
            # Output: probability distribution over all nodes [num_nodes]
            # Gradually reduce dimensions: (hidden_dim*2) -> 256 -> 128 -> num_nodes
            self.action_mlp = MLP(
                hidden_dim * 2,  # current_node + dst_node features
                num_nodes,
                num_layers=mlp_layers,
                hidden_dims=[256, 128] if mlp_layers == 3 else [128] if mlp_layers == 2 else []
            )
        elif action_type == 'edge':
            # Edge-level: Edge score calculation
            # Input: edge features
            # Output: single score per edge
            # Gradually reduce dimensions: hidden_dim -> 64 -> 32 -> 1
            self.action_mlp = MLP(
                hidden_dim,  # edge features
                1,  # single score per edge
                num_layers=mlp_layers,
                hidden_dims=[64, 32] if mlp_layers == 3 else [64] if mlp_layers == 2 else []
            )
        else:
            raise ValueError(f"Invalid action_type: {action_type}. Must be 'node' or 'edge'.")

    def forward(self, node_features, edge_features, current_node, dst_node,
                commodity_idx, valid_edges_mask=None, reached_destination=False):
        """
        Compute action probabilities for next-hop selection.

        Args:
            node_features: Node embeddings [B, V, C, H]
            edge_features: Edge embeddings [B, V, V, C, H]
            current_node: Current node index [B] or int
            dst_node: Destination node index [B] or int
            commodity_idx: Commodity index [B] or int
            valid_edges_mask: Mask for valid edges [B, V] or None
            reached_destination: Whether already at destination (bool)

        Returns:
            action_probs: Probability distribution over nodes [B, V]
            log_probs: Log probabilities [B, V]
            entropy: Entropy of the distribution [B]
        """
        batch_size = node_features.shape[0]
        device = node_features.device

        # Safety check: If already reached destination, return uniform distribution
        # (RolloutEngine should handle this, but this is a safety mechanism)
        if reached_destination:
            action_probs = torch.ones(batch_size, self.num_nodes, device=device) / self.num_nodes
            log_probs = torch.log(action_probs + 1e-8)
            entropy = -torch.sum(action_probs * log_probs, dim=1)
            return action_probs, log_probs, entropy

        # Handle scalar vs batch indices
        if isinstance(current_node, int):
            current_node = torch.tensor([current_node] * batch_size, device=device)
        if isinstance(dst_node, int):
            dst_node = torch.tensor([dst_node] * batch_size, device=device)
        if isinstance(commodity_idx, int):
            commodity_idx = torch.tensor([commodity_idx] * batch_size, device=device)

        if self.action_type == 'node':
            # Node-level action space
            action_logits = self._forward_node_level(
                node_features, current_node, dst_node, commodity_idx
            )
        elif self.action_type == 'edge':
            # Edge-level action space
            action_logits = self._forward_edge_level(
                edge_features, current_node, commodity_idx
            )
        else:
            raise ValueError(f"Invalid action_type: {self.action_type}")

        # Apply valid edges mask (mask invalid edges, self-loops, visited nodes, etc.)
        if valid_edges_mask is not None:
            # Check if any batch has all actions masked (no valid actions)
            has_valid_action = valid_edges_mask.any(dim=-1)  # [B]

            # Mask invalid edges with large negative values
            action_logits = action_logits.masked_fill(~valid_edges_mask, float('-inf'))

            # For batches with no valid actions, create uniform distribution over all actions
            # (This is a fallback - ideally this should never happen in a well-designed environment)
            if not has_valid_action.all():
                # Set uniform logits for batches with no valid actions
                uniform_logits = torch.zeros_like(action_logits[0])
                action_logits = torch.where(
                    has_valid_action.unsqueeze(-1),
                    action_logits,
                    uniform_logits.unsqueeze(0)
                )

        # Compute probabilities
        action_probs = F.softmax(action_logits, dim=-1)
        log_probs = F.log_softmax(action_logits, dim=-1)

        # Handle potential NaN in probabilities (safety check)
        if torch.isnan(action_probs).any() or torch.isinf(action_probs).any():
            # Replace NaN/Inf with uniform distribution
            batch_size, num_actions = action_probs.shape
            action_probs = torch.where(
                torch.isnan(action_probs) | torch.isinf(action_probs),
                torch.full_like(action_probs, 1.0 / num_actions),
                action_probs
            )
            log_probs = torch.log(action_probs + 1e-10)

        # Compute entropy for exploration bonus
        # Entropy = -sum(p * log(p))
        # Clamp to avoid NaN from 0 * log(0)
        safe_log_probs = torch.where(
            action_probs > 0,
            log_probs,
            torch.zeros_like(log_probs)
        )
        entropy = -torch.sum(action_probs * safe_log_probs, dim=1)

        return action_probs, log_probs, entropy

    def _forward_node_level(self, node_features, current_node, dst_node, commodity_idx):
        """
        Node-level action space: Select next node.

        Strategy: Concatenate current node and destination node features,
        then predict distribution over all possible next nodes.

        Args:
            node_features: [B, V, C, H]
            current_node: [B]
            dst_node: [B]
            commodity_idx: [B]

        Returns:
            action_logits: [B, V]
        """
        batch_size = node_features.shape[0]

        # Extract current node features: [B, H]
        batch_indices = torch.arange(batch_size, device=node_features.device)
        current_node_feat = node_features[batch_indices, current_node, commodity_idx]

        # Extract destination node features: [B, H]
        dst_node_feat = node_features[batch_indices, dst_node, commodity_idx]

        # Concatenate current and destination features: [B, 2H]
        context = torch.cat([current_node_feat, dst_node_feat], dim=-1)

        # Predict next node distribution: [B, 2H] -> [B, V]
        action_logits = self.action_mlp(context)

        return action_logits

    def _forward_edge_level(self, edge_features, current_node, commodity_idx):
        """
        Edge-level action space: Score all outgoing edges.

        Strategy: Score each edge from current node, aggregate to node-level distribution.

        Args:
            edge_features: [B, V, V, C, H]
            current_node: [B]
            commodity_idx: [B]

        Returns:
            action_logits: [B, V] (aggregated node-level scores)
        """
        batch_size = edge_features.shape[0]

        # Extract outgoing edges from current node: [B, V, H]
        batch_indices = torch.arange(batch_size, device=edge_features.device)
        outgoing_edges = edge_features[batch_indices, current_node, :, commodity_idx]

        # Score each edge: [B, V, H] -> [B, V, 1] -> [B, V]
        edge_scores = self.action_mlp(outgoing_edges).squeeze(-1)

        return edge_scores

    def sample_action(self, action_probs, temperature=1.0, top_p=0.9):
        """
        Sample action from the policy distribution.

        Supports:
        - Temperature scaling for exploration control
        - Top-p (nucleus) sampling for diverse sampling

        Args:
            action_probs: Probability distribution [B, V]
            temperature: Temperature for scaling (default: 1.0)
            top_p: Nucleus sampling threshold (default: 0.9)

        Returns:
            sampled_actions: Sampled action indices [B]
        """
        # Clamp probabilities to avoid numerical issues
        action_probs = torch.clamp(action_probs, min=1e-10, max=1.0)

        # Renormalize to ensure sum = 1
        action_probs = action_probs / (action_probs.sum(dim=-1, keepdim=True) + 1e-10)

        if temperature != 1.0:
            # Apply temperature scaling to logits
            action_logits = torch.log(action_probs + 1e-10) / temperature
            action_probs = F.softmax(action_logits, dim=-1)

        if top_p < 1.0:
            # Top-p (nucleus) sampling
            sorted_probs, sorted_indices = torch.sort(action_probs, descending=True, dim=-1)
            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)

            # Create mask for top-p nucleus
            # Shift cumsum by one position to include the first element that exceeds top_p
            mask = cumsum_probs - sorted_probs <= top_p
            # Ensure at least one action is always included
            mask[:, 0] = True

            # Filter probabilities and renormalize
            filtered_probs = sorted_probs * mask.float()
            prob_sum = filtered_probs.sum(dim=-1, keepdim=True)

            # Safety check: if all probabilities are zero, use original distribution
            zero_mask = (prob_sum.squeeze(-1) == 0)
            if zero_mask.any():
                filtered_probs[zero_mask] = sorted_probs[zero_mask]
                prob_sum = filtered_probs.sum(dim=-1, keepdim=True)

            filtered_probs = filtered_probs / (prob_sum + 1e-10)

            # Clamp again to avoid numerical issues
            filtered_probs = torch.clamp(filtered_probs, min=1e-10, max=1.0)
            filtered_probs = filtered_probs / (filtered_probs.sum(dim=-1, keepdim=True) + 1e-10)

            # Sample from filtered distribution
            sampled_idx = torch.multinomial(filtered_probs, num_samples=1).squeeze(-1)
            sampled_actions = sorted_indices[torch.arange(sorted_indices.shape[0], device=sorted_indices.device), sampled_idx]
        else:
            # Standard sampling from full distribution
            sampled_actions = torch.multinomial(action_probs, num_samples=1).squeeze(-1)

        return sampled_actions
