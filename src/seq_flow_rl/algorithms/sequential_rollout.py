"""
Sequential Rollout Engine for SeqFlowRL.

Implements sequential path sampling with dynamic state updates and
per-commodity GNN re-encoding (Decision #3: Option 3-B confirmed).
"""

import torch
import torch.nn.functional as F
from collections import defaultdict

from ..utils.mask_utils import MaskGenerator


class SequentialRolloutEngine:
    """
    Sequential Rollout Engine for path generation.

    Key features:
    - Sequential commodity processing
    - Dynamic edge usage tracking
    - Per-commodity GNN updates (Decision #3-B confirmed)
    - Destination reached handling
    - Path loop prevention

    Design decision #3: Per-Commodity GNN update (confirmed)
    - After each commodity's path is determined, update edge usage
    - Re-encode graph state for next commodity
    """

    def __init__(self, model, config):
        """
        Args:
            model: SeqFlowRLModel instance
            config: Configuration dictionary containing:
                - gnn_update_frequency: 'never', 'per_commodity', or 'per_step'
                - sampling_temperature: Temperature for sampling (default: 1.0)
                - sampling_top_p: Nucleus sampling threshold (default: 0.9)
                - max_path_length: Maximum hops per commodity (default: 20)
        """
        self.model = model
        self.config = config

        # GNN update strategy (Decision #3)
        self.gnn_update_frequency = config.get('gnn_update_frequency', 'per_commodity')
        assert self.gnn_update_frequency in ['never', 'per_commodity', 'per_step'], \
            f"Invalid gnn_update_frequency: {self.gnn_update_frequency}"

        # Sampling parameters
        self.temperature = config.get('sampling_temperature', 1.0)
        self.top_p = config.get('sampling_top_p', 0.9)
        self.max_path_length = config.get('max_path_length', 20)

        # Device
        self.device = next(model.parameters()).device

    def rollout(self, batch_data, mode='train', deterministic=False):
        """
        Perform sequential rollout for all commodities in the batch.

        Args:
            batch_data: Dictionary containing:
                - x_nodes: Node features [B, V, C]
                - x_commodities: Commodity data [B, C, 3] (src, dst, demand)
                - x_edges_capacity: Edge capacity [B, V, V]
                - adjacency: Adjacency matrix [B, V, V] (optional)
            mode: 'train' or 'eval'
            deterministic: Use greedy action selection (default: False)

        Returns:
            rollout_results: Dictionary containing:
                - paths: List of paths per batch [B][C][path_length]
                - log_probs: Log probabilities [B, C]
                - entropies: Entropies [B, C]
                - state_values: State values [B, C]
                - total_log_prob: Total log prob per batch [B]
                - mean_entropy: Mean entropy per batch [B]
        """
        # Extract batch data
        x_nodes = batch_data['x_nodes'].to(self.device)
        x_commodities = batch_data['x_commodities'].to(self.device)
        x_edges_capacity = batch_data['x_edges_capacity'].to(self.device)

        batch_size = x_nodes.shape[0]
        num_nodes = x_nodes.shape[1]
        num_commodities = x_commodities.shape[1]

        # Initialize state
        state = self._initialize_state(x_nodes, x_commodities, x_edges_capacity)

        # Reachability check disabled: static reachability doesn't account for dynamic capacity constraints
        # Capacity-based masking is sufficient and more accurate for this problem
        reachability = None

        # Storage for results
        batch_paths = [[] for _ in range(batch_size)]
        batch_log_probs = torch.zeros(batch_size, num_commodities, device=self.device)
        batch_entropies = torch.zeros(batch_size, num_commodities, device=self.device)
        batch_state_values = torch.zeros(batch_size, num_commodities, device=self.device)

        # Initial GNN encoding (if gnn_update_frequency == 'never', this is the only encoding)
        if self.gnn_update_frequency == 'never':
            node_features, edge_features, _ = self.model.encoder(
                state['x_nodes'],
                state['x_commodities'],
                state['x_edges_capacity'],
                state['x_edges_usage']
            )

        # Sequential rollout: Process each commodity in order
        for c in range(num_commodities):
            # Per-commodity GNN update (Decision #3-B: confirmed)
            if self.gnn_update_frequency == 'per_commodity':
                # Re-encode with updated edge usage
                node_features, edge_features, _ = self.model.encoder(
                    state['x_nodes'],
                    state['x_commodities'],
                    state['x_edges_capacity'],
                    state['x_edges_usage']  # Updated usage from previous commodities
                )

            # Sample path for commodity c
            path, log_prob, entropy, state_value = self._sample_path(
                node_features,
                edge_features,
                state,
                commodity_idx=c,
                reachability=reachability,
                deterministic=deterministic
            )

            # Store results for this commodity
            for b in range(batch_size):
                batch_paths[b].append(path[b])
            batch_log_probs[:, c] = log_prob
            batch_entropies[:, c] = entropy
            batch_state_values[:, c] = state_value

            # Update edge usage with the sampled paths
            self._update_edge_usage(state, path, c)

        # Aggregate results
        rollout_results = {
            'paths': batch_paths,
            'log_probs': batch_log_probs,  # [B, C]
            'entropies': batch_entropies,  # [B, C]
            'state_values': batch_state_values,  # [B, C]
            'total_log_prob': batch_log_probs.sum(dim=1),  # [B]
            'mean_entropy': batch_entropies.mean(dim=1),  # [B]
            'edge_usage': state['x_edges_usage'],  # Final edge usage [B, V, V]
        }

        return rollout_results

    def _initialize_state(self, x_nodes, x_commodities, x_edges_capacity):
        """
        Initialize rollout state.

        Args:
            x_nodes: Node features [B, V, C]
            x_commodities: Commodity data [B, C, 3]
            x_edges_capacity: Edge capacity [B, V, V]

        Returns:
            state: Dictionary containing graph state
        """
        batch_size = x_nodes.shape[0]
        num_nodes = x_nodes.shape[1]

        state = {
            'x_nodes': x_nodes,
            'x_commodities': x_commodities,
            'x_edges_capacity': x_edges_capacity,
            # Initialize edge usage to zero (will be updated during rollout)
            'x_edges_usage': torch.zeros_like(x_edges_capacity),
        }

        return state

    def _sample_path(self, node_features, edge_features, state, commodity_idx,
                     reachability, deterministic=False):
        """
        Sample a single path for one commodity across all batches.

        Args:
            node_features: Node embeddings [B, V, C, H]
            edge_features: Edge embeddings [B, V, V, C, H]
            state: Current state dictionary
            commodity_idx: Commodity index
            reachability: Reachability matrix [B, V, V]
            deterministic: Use greedy selection

        Returns:
            paths: List of paths per batch element [B][path_length]
            log_prob: Total log probability per batch [B]
            entropy: Total entropy per batch [B]
            state_value: State value per batch [B]
        """
        batch_size = node_features.shape[0]
        num_nodes = node_features.shape[1]

        # Extract src, dst, demand for this commodity
        # x_commodities shape: [B, C, 3] where 3 = (src, dst, demand)
        src_nodes = state['x_commodities'][:, commodity_idx, 0].long()  # [B]
        dst_nodes = state['x_commodities'][:, commodity_idx, 1].long()  # [B]
        demands = state['x_commodities'][:, commodity_idx, 2]  # [B]

        # Initialize paths and log probabilities
        # IMPORTANT: Initialize paths with source nodes
        paths = [[src_nodes[b].item()] for b in range(batch_size)]
        log_probs = torch.zeros(batch_size, device=self.device)
        entropies_sum = torch.zeros(batch_size, device=self.device)
        num_steps = torch.zeros(batch_size, device=self.device)

        # Track current position for each batch element
        current_nodes = src_nodes.clone()  # [B]

        # Track visited nodes per batch (to prevent loops)
        visited_nodes = [set([src_nodes[b].item()]) for b in range(batch_size)]

        # Track which batch elements have reached destination
        reached_dst = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        # Compute state value once at the beginning (Critic evaluation)
        state_value = self.model.critic(node_features)  # [B]

        # Sequential path generation
        for step in range(self.max_path_length):
            # Check which batch elements have reached destination
            reached_dst = reached_dst | (current_nodes == dst_nodes)

            # If all reached destination, stop
            if reached_dst.all():
                break

            # Create valid action mask (including capacity constraints)
            valid_mask = self._create_valid_mask(
                current_nodes, dst_nodes, state['x_edges_capacity'],
                visited_nodes, reachability, reached_dst, state['x_edges_usage'], demands
            )

            # Get action probabilities from Policy Head
            action_probs, action_log_probs, entropy = self.model.actor(
                node_features,
                edge_features,
                current_nodes,
                dst_nodes,
                commodity_idx,
                valid_edges_mask=valid_mask,
                reached_destination=False
            )

            # Sample next node
            if deterministic:
                # Greedy: select highest probability action
                next_nodes = torch.argmax(action_probs, dim=-1)  # [B]
            else:
                # Stochastic: sample from distribution
                next_nodes = self.model.actor.sample_action(
                    action_probs,
                    temperature=self.temperature,
                    top_p=self.top_p
                )  # [B]

            # Update paths and log probabilities (only for batch elements not yet at destination)
            for b in range(batch_size):
                if not reached_dst[b]:
                    next_node = next_nodes[b].item()
                    paths[b].append(next_node)
                    visited_nodes[b].add(next_node)

                    # Accumulate log probability of this action
                    log_probs[b] += action_log_probs[b, next_node]

                    # Accumulate entropy
                    entropies_sum[b] += entropy[b]
                    num_steps[b] += 1

            # Move to next nodes
            current_nodes = next_nodes

            # Per-step GNN update (experimental, not default)
            if self.gnn_update_frequency == 'per_step':
                # Update edge usage for this step
                self._update_edge_usage_step(state, current_nodes, next_nodes, demands, commodity_idx)

                # Re-encode graph
                node_features, edge_features, _ = self.model.encoder(
                    state['x_nodes'],
                    state['x_commodities'],
                    state['x_edges_capacity'],
                    state['x_edges_usage']
                )

        # Compute mean entropy per path
        mean_entropy = entropies_sum / (num_steps + 1e-8)

        # Verify all paths reached destination
        # If not, penalize by setting large negative log probability
        final_nodes = torch.tensor([paths[b][-1] if len(paths[b]) > 0 else src_nodes[b].item()
                                    for b in range(batch_size)], device=self.device)
        reached_destination_final = (final_nodes == dst_nodes)

        # Apply heavy penalty to log_probs for paths that didn't reach destination
        # This discourages incomplete paths during learning
        penalty_for_incomplete = -10.0
        log_probs = torch.where(
            reached_destination_final,
            log_probs,
            log_probs + penalty_for_incomplete
        )

        return paths, log_probs, mean_entropy, state_value

    def _create_valid_mask(self, current_nodes, dst_nodes, edges_capacity,
                           visited_nodes, reachability, reached_dst, edges_usage, demands):
        """
        Create valid action mask for current step.

        Args:
            current_nodes: Current positions [B]
            dst_nodes: Destination nodes [B]
            edges_capacity: Edge capacity [B, V, V]
            visited_nodes: List of sets of visited nodes per batch
            reachability: Reachability matrix [B, V, V]
            reached_dst: Boolean mask [B] indicating which reached destination
            edges_usage: Current edge usage [B, V, V]
            demands: Demand for current commodity [B]

        Returns:
            valid_mask: Valid action mask [B, V]
        """
        batch_size = len(current_nodes)
        num_nodes = edges_capacity.shape[1]
        device = edges_capacity.device

        # Generate full valid mask (with capacity constraints, no reachability check)
        valid_mask = MaskGenerator.create_full_valid_mask(
            current_nodes,
            dst_nodes,
            edges_capacity,
            visited_nodes=visited_nodes,
            reachability=None,
            check_reachability=False,
            edges_usage=edges_usage,
            demands=demands
        )

        # For batch elements that already reached destination, mask all actions
        # (they shouldn't take any more actions)
        if reached_dst.any():
            valid_mask[reached_dst] = False  # Mask all actions for reached elements

        return valid_mask

    def _update_edge_usage(self, state, paths, commodity_idx):
        """
        Update edge usage after a commodity's path is determined.

        This implements per-commodity state update (Decision #3-B).

        Args:
            state: State dictionary
            paths: List of paths per batch [B][path_length]
            commodity_idx: Index of the commodity
        """
        batch_size = len(paths)
        demands = state['x_commodities'][:, commodity_idx, 2]  # [B]
        dst_nodes = state['x_commodities'][:, commodity_idx, 1].long()  # [B]

        for b in range(batch_size):
            path = paths[b]
            demand = demands[b].item()

            # Only update edge usage if path reached destination
            if len(path) > 0 and path[-1] == dst_nodes[b].item():
                # Update edge usage along the path
                for i in range(len(path) - 1):
                    u = path[i]
                    v = path[i + 1]
                    state['x_edges_usage'][b, u, v] += demand
            # If path didn't reach destination, don't count its edge usage
            # This prevents incomplete paths from affecting capacity constraints

    def _update_edge_usage_step(self, state, prev_nodes, next_nodes, demands, commodity_idx):
        """
        Update edge usage for a single step (per-step GNN update).

        This is for experimental per-step updates (Decision #3-C, not default).

        Args:
            state: State dictionary
            prev_nodes: Previous node positions [B]
            next_nodes: Next node positions [B]
            demands: Demands [B]
            commodity_idx: Commodity index
        """
        batch_size = len(prev_nodes)

        for b in range(batch_size):
            u = prev_nodes[b].item()
            v = next_nodes[b].item()
            demand = demands[b].item()
            state['x_edges_usage'][b, u, v] += demand
