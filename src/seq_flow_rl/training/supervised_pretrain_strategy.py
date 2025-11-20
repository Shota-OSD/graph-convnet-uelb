"""
Supervised Pre-training Strategy for SeqFlowRL

Implements behavior cloning from optimal solution paths.
The policy network learns to mimic expert actions (next node selection)
at each step of the ground truth paths.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SupervisedPretrainStrategy:
    """
    Supervised pre-training strategy using behavior cloning.

    This strategy trains the policy (actor) to imitate optimal paths:
    1. For each step in ground truth path, predict next node
    2. Compute cross-entropy loss against ground truth next node
    3. Update policy network to maximize accuracy

    The value network (critic) is NOT trained during pre-training.
    """

    def __init__(self, model, rollout_engine, config, device):
        """
        Args:
            model: SeqFlowRL model (with encoder, actor, critic)
            rollout_engine: Sequential rollout engine
            config: Configuration dictionary
            device: Device for computation
        """
        self.model = model
        self.rollout_engine = rollout_engine
        self.config = config
        self.device = device

        # Loss function for classification
        self.criterion = nn.CrossEntropyLoss(reduction='mean')

    def train_step(self, batch_data):
        """
        Perform one training step with behavior cloning.

        Args:
            batch_data: Dictionary containing:
                       - x_nodes: [B, V, C]
                       - x_edges_capacity: [B, V, V]
                       - x_commodities: [B, C, 3]
                       - path_sequences: [B, C, max_path_len]
                       - path_lengths: [B, C]

        Returns:
            metrics: Dictionary with loss and accuracy
        """
        self.model.train()

        # Move data to device
        x_nodes = batch_data['x_nodes'].to(self.device)
        x_edges_capacity = batch_data['x_edges_capacity'].to(self.device)
        x_commodities = batch_data['x_commodities'].to(self.device)
        path_sequences = batch_data['path_sequences'].to(self.device)
        path_lengths = batch_data['path_lengths'].to(self.device)

        batch_size = x_nodes.shape[0]
        num_nodes = x_nodes.shape[1]
        num_commodities = x_commodities.shape[1]

        # Initialize edge usage (start with zero usage)
        x_edges_usage = torch.zeros_like(x_edges_capacity)

        # Track metrics
        accumulated_loss = 0.0  # For backward (tensor)
        total_loss_value = 0.0  # For logging (float)
        total_correct = 0
        total_predictions = 0

        # Process each commodity sequentially
        for commodity_idx in range(num_commodities):
            # GNN encoding (per-commodity update frequency)
            node_features, edge_features, _ = self.model.encoder(
                x_nodes,
                x_commodities,
                x_edges_capacity,
                x_edges_usage
            )

            # Extract src, dst for this commodity
            src_nodes = x_commodities[:, commodity_idx, 0].long()  # [B]
            dst_nodes = x_commodities[:, commodity_idx, 1].long()  # [B]
            demands = x_commodities[:, commodity_idx, 2]  # [B]

            # Get ground truth paths
            gt_paths = path_sequences[:, commodity_idx, :]  # [B, max_path_len]
            gt_path_lens = path_lengths[:, commodity_idx]  # [B]

            # Process each step in the path
            for step_idx in range(gt_paths.shape[1] - 1):  # -1 because we need next node
                # Check which batch elements have a valid step
                valid_batch = (step_idx < gt_path_lens - 1)  # -1 for next node
                if not valid_batch.any():
                    break  # No more valid steps

                # Current nodes at this step
                current_nodes = gt_paths[:, step_idx]  # [B]
                next_nodes_gt = gt_paths[:, step_idx + 1]  # [B] - ground truth next nodes

                # Create a simple valid mask (we're following ground truth, so mask is permissive)
                # Mask out invalid edges (no capacity) and self-loops
                invalid_edges = (x_edges_capacity <= 0) | torch.eye(num_nodes, device=self.device).bool().unsqueeze(0)
                valid_mask = ~invalid_edges[torch.arange(batch_size), current_nodes, :]  # [B, V]

                # Get action probabilities from policy
                # Note: We don't update edge usage during supervised pre-training
                # to keep it simple and focus on learning path selection
                action_logits, _, _ = self.model.actor(
                    node_features,
                    edge_features,
                    current_nodes,
                    dst_nodes,
                    commodity_idx,
                    valid_edges_mask=valid_mask,
                    reached_destination=False
                )

                # Filter to only valid batch elements
                valid_indices = valid_batch.nonzero(as_tuple=True)[0]
                if len(valid_indices) == 0:
                    continue

                # Compute loss only for valid batch elements
                logits_valid = action_logits[valid_indices]  # [num_valid, V]
                targets_valid = next_nodes_gt[valid_indices]  # [num_valid]

                loss = self.criterion(logits_valid, targets_valid)

                # Accumulate loss for backward (keep as tensor)
                if isinstance(accumulated_loss, float):
                    accumulated_loss = loss
                else:
                    accumulated_loss = accumulated_loss + loss

                total_loss_value += loss.item() * len(valid_indices)

                # Compute accuracy
                predictions = torch.argmax(logits_valid, dim=-1)
                correct = (predictions == targets_valid).sum().item()
                total_correct += correct
                total_predictions += len(valid_indices)

            # Update edge usage for next commodity (simplified - just mark edges as used)
            # This is optional for pre-training, but helps with multi-commodity scenarios
            for b in range(batch_size):
                path_len = gt_path_lens[b].item()
                if path_len > 1:
                    for step in range(path_len - 1):
                        src = gt_paths[b, step].item()
                        tgt = gt_paths[b, step + 1].item()
                        x_edges_usage[b, src, tgt] += demands[b]

        # Backward pass (only once, after accumulating all losses)
        if not isinstance(accumulated_loss, float) and total_predictions > 0:
            accumulated_loss.backward()

        # Average metrics
        avg_loss = total_loss_value / max(total_predictions, 1)
        accuracy = (total_correct / max(total_predictions, 1)) * 100.0

        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'total_predictions': total_predictions,
        }

        return metrics

    def eval_step(self, batch_data):
        """
        Evaluate the model on validation data.

        Args:
            batch_data: Same format as train_step

        Returns:
            metrics: Dictionary with loss and accuracy
        """
        self.model.eval()

        with torch.no_grad():
            # Move data to device
            x_nodes = batch_data['x_nodes'].to(self.device)
            x_edges_capacity = batch_data['x_edges_capacity'].to(self.device)
            x_commodities = batch_data['x_commodities'].to(self.device)
            path_sequences = batch_data['path_sequences'].to(self.device)
            path_lengths = batch_data['path_lengths'].to(self.device)

            batch_size = x_nodes.shape[0]
            num_nodes = x_nodes.shape[1]
            num_commodities = x_commodities.shape[1]

            # Initialize edge usage
            x_edges_usage = torch.zeros_like(x_edges_capacity)

            # Track metrics
            total_loss = 0.0
            total_correct = 0
            total_predictions = 0

            # Process each commodity
            for commodity_idx in range(num_commodities):
                # GNN encoding
                node_features, edge_features, _ = self.model.encoder(
                    x_nodes,
                    x_commodities,
                    x_edges_capacity,
                    x_edges_usage
                )

                src_nodes = x_commodities[:, commodity_idx, 0].long()
                dst_nodes = x_commodities[:, commodity_idx, 1].long()
                demands = x_commodities[:, commodity_idx, 2]

                gt_paths = path_sequences[:, commodity_idx, :]
                gt_path_lens = path_lengths[:, commodity_idx]

                # Process each step
                for step_idx in range(gt_paths.shape[1] - 1):
                    valid_batch = (step_idx < gt_path_lens - 1)
                    if not valid_batch.any():
                        break

                    current_nodes = gt_paths[:, step_idx]
                    next_nodes_gt = gt_paths[:, step_idx + 1]

                    invalid_edges = (x_edges_capacity <= 0) | torch.eye(num_nodes, device=self.device).bool().unsqueeze(0)
                    valid_mask = ~invalid_edges[torch.arange(batch_size), current_nodes, :]

                    action_logits, _, _ = self.model.actor(
                        node_features,
                        edge_features,
                        current_nodes,
                        dst_nodes,
                        commodity_idx,
                        valid_edges_mask=valid_mask,
                        reached_destination=False
                    )

                    valid_indices = valid_batch.nonzero(as_tuple=True)[0]
                    if len(valid_indices) == 0:
                        continue

                    logits_valid = action_logits[valid_indices]
                    targets_valid = next_nodes_gt[valid_indices]

                    loss = self.criterion(logits_valid, targets_valid)
                    total_loss += loss.item() * len(valid_indices)

                    predictions = torch.argmax(logits_valid, dim=-1)
                    correct = (predictions == targets_valid).sum().item()
                    total_correct += correct
                    total_predictions += len(valid_indices)

                # Update edge usage
                for b in range(batch_size):
                    path_len = gt_path_lens[b].item()
                    if path_len > 1:
                        for step in range(path_len - 1):
                            src = gt_paths[b, step].item()
                            tgt = gt_paths[b, step + 1].item()
                            x_edges_usage[b, src, tgt] += demands[b]

            # Average metrics
            avg_loss = total_loss / max(total_predictions, 1)
            accuracy = (total_correct / max(total_predictions, 1)) * 100.0

            metrics = {
                'loss': avg_loss,
                'accuracy': accuracy,
                'total_predictions': total_predictions,
            }

        return metrics
