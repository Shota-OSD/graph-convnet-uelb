"""
Path Sampler for Reinforcement Learning

Samples paths probabilistically from GCN edge predictions
instead of using deterministic beam search.
This ensures on-policy learning for REINFORCE algorithm.
"""

import torch
import torch.nn.functional as F
import numpy as np


class PathSampler:
    """
    Samples paths from edge probability distributions for RL training.

    Unlike beam search which finds top-K paths deterministically,
    this sampler draws paths stochastically from the learned policy,
    which is required for proper REINFORCE gradient estimation.
    """

    def __init__(self, y_pred_edges, edges_capacity, commodities,
                 num_samples=1, temperature=1.0, top_p=0.9,
                 dtypeFloat=torch.float, dtypeLong=torch.long):
        """
        Args:
            y_pred_edges: Edge predictions [batch, nodes, nodes, classes]
            edges_capacity: Edge capacities [batch, nodes, nodes]
            commodities: List of (src, dst, demand) tuples
            num_samples: Number of path samples to draw
            temperature: Temperature for softmax (lower = more deterministic)
            top_p: Nucleus sampling threshold (0.9 = sample from top 90% probability mass)
            dtypeFloat: Float tensor type
            dtypeLong: Long tensor type
        """
        self.y_pred_edges = y_pred_edges
        self.edges_capacity = edges_capacity
        self.commodities = commodities
        self.num_samples = num_samples
        self.temperature = temperature
        self.top_p = top_p
        self.dtypeFloat = dtypeFloat
        self.dtypeLong = dtypeLong

        self.batch_size = y_pred_edges.shape[0]
        self.num_nodes = y_pred_edges.shape[1]

    def sample(self):
        """
        Sample paths for all commodities using probabilistic selection.

        Returns:
            paths: List of paths for each batch and commodity
            log_probs: Log probabilities of sampled paths (for REINFORCE)
            is_feasible: Whether the solution respects capacity constraints
        """
        device = self.y_pred_edges.device
        batch_paths = []
        batch_log_probs = []

        for b in range(self.batch_size):
            # Get edge predictions for this batch item
            # y_pred_edges shape can be:
            # - [batch, nodes, nodes, classes] or
            # - [batch, nodes, nodes, commodities, classes]
            edge_logits = self.y_pred_edges[b]

            commodity_paths = []
            commodity_log_probs = []
            remaining_capacity = self.edges_capacity[b].clone()

            # Get commodities for this batch item
            # commodities can be:
            # 1. List of tuples/lists (same for all batches)
            # 2. Tensor [batch_size, num_commodities, 3]
            if isinstance(self.commodities, torch.Tensor) and len(self.commodities.shape) == 3:
                batch_commodities = self.commodities[b]  # [num_commodities, 3]
            else:
                batch_commodities = self.commodities  # Same for all batches

            for c_idx, commodity in enumerate(batch_commodities):
                # Handle both list/tuple and tensor formats
                if isinstance(commodity, (list, tuple)):
                    src, dst, demand = commodity
                else:
                    # Tensor format: [src, dst, demand, ...]
                    src = int(commodity[0].item() if hasattr(commodity[0], 'item') else commodity[0])
                    dst = int(commodity[1].item() if hasattr(commodity[1], 'item') else commodity[1])
                    demand = float(commodity[2].item() if hasattr(commodity[2], 'item') else commodity[2])

                # Get edge probabilities for this commodity
                if len(edge_logits.shape) == 4:
                    # Shape: [num_nodes, num_nodes, commodities, classes]
                    commodity_edge_logits = edge_logits[:, :, c_idx, :]
                else:
                    # Shape: [num_nodes, num_nodes, classes]
                    commodity_edge_logits = edge_logits

                # Convert to probabilities (class 1 = "use this edge")
                # Shape: [num_nodes, num_nodes]
                edge_probs = F.softmax(commodity_edge_logits / self.temperature, dim=-1)[:, :, 1]

                # Sample a path from src to dst
                path, log_prob = self._sample_single_path(
                    edge_probs, src, dst, demand, remaining_capacity
                )

                commodity_paths.append(path)
                commodity_log_probs.append(log_prob)

                # Update remaining capacity
                if len(path) > 1:
                    for i in range(len(path) - 1):
                        u, v = path[i], path[i + 1]
                        remaining_capacity[u, v] -= demand

            batch_paths.append(commodity_paths)

            # Sum log probabilities for this batch (total trajectory log prob)
            total_log_prob = sum(commodity_log_probs)
            batch_log_probs.append(total_log_prob)

        # Check feasibility
        is_feasible = self._check_feasibility(batch_paths)

        # Convert log probs to tensor
        log_probs_tensor = torch.tensor(batch_log_probs, dtype=torch.float32, device=device)

        return batch_paths, log_probs_tensor, is_feasible

    def _sample_single_path(self, edge_probs, src, dst, demand, remaining_capacity):
        """
        Sample a single path from src to dst using top-p sampling.

        Args:
            edge_probs: Edge probabilities [num_nodes, num_nodes]
            src: Source node
            dst: Destination node
            demand: Demand amount
            remaining_capacity: Remaining edge capacities [num_nodes, num_nodes]

        Returns:
            path: List of nodes forming the path
            log_prob: Log probability of the sampled path
        """
        # Ensure src and dst are integers
        src = int(src)
        dst = int(dst)

        path = [src]
        log_prob = 0.0
        current = src
        max_steps = self.num_nodes * 2  # Prevent infinite loops

        for step in range(max_steps):
            if current == dst:
                break

            # Get outgoing edge probabilities from current node
            # Shape: [num_nodes]
            outgoing_probs = edge_probs[current].clone()

            # Mask out infeasible edges (insufficient capacity)
            for next_node in range(self.num_nodes):
                if remaining_capacity[current, next_node] < demand:
                    outgoing_probs[next_node] = 0.0

            # Mask out already visited nodes (prevent cycles)
            for visited in path:
                outgoing_probs[visited] = 0.0

            # Normalize probabilities
            prob_sum = outgoing_probs.sum()
            if prob_sum == 0:
                # No feasible next node - fallback to shortest path or random
                # For now, just go to destination directly (greedy fallback)
                path.append(dst)
                # Add small penalty to log_prob
                log_prob += np.log(1e-8)
                break

            outgoing_probs = outgoing_probs / prob_sum

            # Top-p (nucleus) sampling
            next_node = self._top_p_sample(outgoing_probs)

            # Record log probability
            log_prob += torch.log(outgoing_probs[next_node] + 1e-8).item()

            # Move to next node (ensure integer)
            path.append(int(next_node))
            current = int(next_node)

        # If we didn't reach destination, add it (fallback)
        if path[-1] != dst:
            path.append(dst)
            log_prob += np.log(1e-8)  # Penalty for fallback

        return path, log_prob

    def _top_p_sample(self, probs):
        """
        Nucleus (top-p) sampling: sample from smallest set of tokens
        whose cumulative probability exceeds p.

        Args:
            probs: Probability distribution [num_nodes]

        Returns:
            sampled_idx: Sampled node index
        """
        # Sort probabilities in descending order
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)

        # Compute cumulative probabilities
        cumsum_probs = torch.cumsum(sorted_probs, dim=0)

        # Find cutoff index where cumsum exceeds top_p
        cutoff_idx = torch.where(cumsum_probs > self.top_p)[0]
        if len(cutoff_idx) > 0:
            cutoff_idx = cutoff_idx[0].item() + 1
        else:
            cutoff_idx = len(sorted_probs)

        # Keep only top-p probability mass
        top_p_probs = sorted_probs[:cutoff_idx]
        top_p_indices = sorted_indices[:cutoff_idx]

        # Renormalize
        top_p_probs = top_p_probs / top_p_probs.sum()

        # Sample from top-p distribution
        sampled_idx_in_top_p = torch.multinomial(top_p_probs, num_samples=1).item()
        sampled_idx = top_p_indices[sampled_idx_in_top_p].item()

        return sampled_idx

    def _check_feasibility(self, batch_paths):
        """
        Check if sampled paths respect capacity constraints.

        Args:
            batch_paths: List of paths for each batch

        Returns:
            is_feasible: True if all paths are feasible
        """
        for b, commodity_paths in enumerate(batch_paths):
            edge_usage = torch.zeros_like(self.edges_capacity[b])

            # Get commodities for this batch item
            if isinstance(self.commodities, torch.Tensor) and len(self.commodities.shape) == 3:
                batch_commodities = self.commodities[b]  # [num_commodities, 3]
            else:
                batch_commodities = self.commodities  # Same for all batches

            for i, path in enumerate(commodity_paths):
                # Handle both list/tuple and tensor formats
                commodity = batch_commodities[i]
                if isinstance(commodity, (list, tuple)):
                    src, dst, demand = commodity
                else:
                    # Tensor format: [src, dst, demand, ...]
                    src = int(commodity[0].item() if hasattr(commodity[0], 'item') else commodity[0])
                    dst = int(commodity[1].item() if hasattr(commodity[1], 'item') else commodity[1])
                    demand = float(commodity[2].item() if hasattr(commodity[2], 'item') else commodity[2])

                # Track edge usage
                if len(path) > 1:
                    for j in range(len(path) - 1):
                        u, v = path[j], path[j + 1]
                        edge_usage[u, v] += demand

            # Check capacity violations
            if (edge_usage > self.edges_capacity[b]).any():
                return False

        return True
