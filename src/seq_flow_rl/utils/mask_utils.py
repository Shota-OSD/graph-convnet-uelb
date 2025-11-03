"""
Mask generation utilities for valid edge/node masking during rollout.

Handles multiple types of masks:
1. Invalid edges (zero capacity)
2. Self-loops
3. Visited nodes (to prevent loops)
4. Destination reached (no further actions needed)
5. Reachability (ensure path to destination exists)
"""

import torch
from collections import deque


class MaskGenerator:
    """
    Generate various masks for valid action selection during path sampling.
    """

    @staticmethod
    def create_invalid_edges_mask(edges_capacity, num_nodes, device, edges_usage=None, demands=None):
        """
        Create mask for invalid edges (zero/insufficient capacity or self-loops).

        Args:
            edges_capacity: Edge capacity matrix [B, V, V]
            num_nodes: Number of nodes
            device: Device to create mask on
            edges_usage: Current edge usage [B, V, V] (optional, for capacity constraints)
            demands: Demand for current commodity [B] (optional, for demand-aware capacity check)

        Returns:
            invalid_mask: Boolean mask [B, V, V], True = invalid edge
        """
        batch_size = edges_capacity.shape[0]

        # Edges with zero capacity are invalid
        zero_capacity_mask = edges_capacity <= 0

        # If edge usage is provided, check remaining capacity
        if edges_usage is not None:
            remaining_capacity = edges_capacity - edges_usage

            # If demands are provided, check if remaining capacity >= demand
            if demands is not None:
                # demands: [B] -> [B, 1, 1] for broadcasting
                demands_expanded = demands.view(batch_size, 1, 1)
                # Edges with insufficient remaining capacity for this demand are invalid
                insufficient_capacity_mask = remaining_capacity < demands_expanded
            else:
                # Edges with no remaining capacity are invalid (with small epsilon for numerical stability)
                insufficient_capacity_mask = remaining_capacity <= 1e-6

            zero_capacity_mask = zero_capacity_mask | insufficient_capacity_mask

        # Self-loops are invalid
        self_loop_mask = torch.eye(num_nodes, device=device, dtype=torch.bool).unsqueeze(0).expand(batch_size, -1, -1)

        # Combine masks
        invalid_mask = zero_capacity_mask | self_loop_mask

        return invalid_mask

    @staticmethod
    def create_valid_neighbors_mask(current_node, edges_capacity, invalid_edges_mask):
        """
        Create mask for valid neighbor nodes from current node.

        Args:
            current_node: Current node index [B] or int
            edges_capacity: Edge capacity matrix [B, V, V]
            invalid_edges_mask: Invalid edges mask [B, V, V]

        Returns:
            valid_neighbors_mask: Boolean mask [B, V], True = valid neighbor
        """
        batch_size = edges_capacity.shape[0]
        num_nodes = edges_capacity.shape[1]
        device = edges_capacity.device

        # Handle scalar current_node
        if isinstance(current_node, int):
            current_node = torch.tensor([current_node] * batch_size, device=device)

        # Extract outgoing edges from current node
        batch_indices = torch.arange(batch_size, device=device)
        outgoing_edges_valid = ~invalid_edges_mask[batch_indices, current_node]  # [B, V]

        return outgoing_edges_valid

    @staticmethod
    def apply_visited_nodes_mask(valid_mask, visited_nodes):
        """
        Mask out already visited nodes to prevent loops.

        Args:
            valid_mask: Current valid mask [B, V]
            visited_nodes: Set or list of visited node indices per batch

        Returns:
            updated_mask: Valid mask with visited nodes masked out [B, V]
        """
        if isinstance(visited_nodes, list):
            # visited_nodes is a list (one per batch)
            for b, visited_set in enumerate(visited_nodes):
                for node in visited_set:
                    valid_mask[b, node] = False
        else:
            # visited_nodes is a tensor [B, V] boolean mask
            valid_mask = valid_mask & ~visited_nodes

        return valid_mask

    @staticmethod
    def compute_reachability(edges_capacity, edges_usage=None):
        """
        Compute reachability matrix using BFS for each batch.

        Args:
            edges_capacity: Edge capacity matrix [B, V, V]
            edges_usage: Current edge usage [B, V, V] (optional, for capacity constraints)

        Returns:
            reachability: Reachability matrix [B, V, V]
                         reachability[b, i, j] = True if node j is reachable from node i
        """
        batch_size, num_nodes, _ = edges_capacity.shape
        device = edges_capacity.device

        # Create adjacency matrix based on available capacity
        if edges_usage is not None:
            remaining_capacity = edges_capacity - edges_usage
            adjacency = (remaining_capacity > 1e-6).float()
        else:
            adjacency = (edges_capacity > 0).float()

        # Initialize reachability matrix (start with direct connections)
        reachability = adjacency.clone()

        # Add self-loops (node is reachable from itself)
        eye = torch.eye(num_nodes, device=device).unsqueeze(0).expand(batch_size, -1, -1)
        reachability = reachability + eye
        reachability = (reachability > 0).float()

        # Floyd-Warshall algorithm for transitive closure
        # This computes all-pairs reachability
        for k in range(num_nodes):
            # reachability[i,j] = reachability[i,j] OR (reachability[i,k] AND reachability[k,j])
            reachability_ik = reachability[:, :, k:k+1]  # [B, V, 1]
            reachability_kj = reachability[:, k:k+1, :]  # [B, 1, V]
            reachability = torch.maximum(
                reachability,
                (reachability_ik * reachability_kj)
            )

        return reachability.bool()

    @staticmethod
    def apply_reachability_mask(valid_mask, current_node, dst_node, reachability):
        """
        Mask out nodes from which the destination is not reachable.

        This prevents selecting nodes that lead to dead ends.

        Args:
            valid_mask: Current valid mask [B, V]
            current_node: Current node [B] or int
            dst_node: Destination node [B] or int
            reachability: Reachability matrix [B, V, V]

        Returns:
            updated_mask: Valid mask with unreachable nodes masked out [B, V]
        """
        batch_size = valid_mask.shape[0]
        device = valid_mask.device

        # Handle scalar inputs
        if isinstance(current_node, int):
            current_node = torch.tensor([current_node] * batch_size, device=device)
        if isinstance(dst_node, int):
            dst_node = torch.tensor([dst_node] * batch_size, device=device)

        # For each potential next node, check if destination is reachable
        # reachability[b, next_node, dst_node] must be True
        batch_indices = torch.arange(batch_size, device=device)

        # Create mask: for each next node, check if dst is reachable from it
        # reachability: [B, V, V]
        # We want reachability[:, :, dst_node] for each batch
        reachable_to_dst = reachability[batch_indices, :, dst_node]  # [B, V]

        # Apply reachability constraint
        valid_mask = valid_mask & reachable_to_dst

        return valid_mask

    @staticmethod
    def create_full_valid_mask(current_node, dst_node, edges_capacity,
                               visited_nodes=None, reachability=None,
                               check_reachability=True, edges_usage=None, demands=None):
        """
        Create complete valid action mask combining all constraints.

        Args:
            current_node: Current node [B] or int
            dst_node: Destination node [B] or int
            edges_capacity: Edge capacity [B, V, V]
            visited_nodes: Visited nodes (list of sets or tensor)
            reachability: Precomputed reachability matrix [B, V, V]
            check_reachability: Whether to apply reachability constraint
            edges_usage: Current edge usage [B, V, V] (for capacity constraints)
            demands: Demand for current commodity [B] (for demand-aware capacity check)

        Returns:
            valid_mask: Complete valid action mask [B, V], True = valid action
        """
        batch_size = edges_capacity.shape[0]
        num_nodes = edges_capacity.shape[1]
        device = edges_capacity.device

        # Handle scalar inputs
        if isinstance(current_node, int):
            current_node_tensor = torch.tensor([current_node] * batch_size, device=device)
        else:
            current_node_tensor = current_node

        if isinstance(dst_node, int):
            dst_node_tensor = torch.tensor([dst_node] * batch_size, device=device)
        else:
            dst_node_tensor = dst_node

        # 1. Create invalid edges mask (including capacity constraints with demand check)
        invalid_edges_mask = MaskGenerator.create_invalid_edges_mask(
            edges_capacity, num_nodes, device, edges_usage=edges_usage, demands=demands
        )

        # 2. Get valid neighbors from current node
        valid_mask = MaskGenerator.create_valid_neighbors_mask(
            current_node_tensor, edges_capacity, invalid_edges_mask
        )

        # 3. Apply visited nodes mask (prevent loops)
        if visited_nodes is not None:
            valid_mask = MaskGenerator.apply_visited_nodes_mask(valid_mask, visited_nodes)

        # 4. Apply reachability constraint
        if check_reachability and reachability is not None:
            valid_mask = MaskGenerator.apply_reachability_mask(
                valid_mask, current_node_tensor, dst_node_tensor, reachability
            )

        # 5. Ensure at least one valid action exists (safety check)
        # If no valid actions, allow moving to any unvisited neighbor
        has_valid_action = valid_mask.any(dim=1)  # [B]
        if not has_valid_action.all():
            # Fallback: allow any neighbor with capacity
            fallback_mask = MaskGenerator.create_valid_neighbors_mask(
                current_node_tensor, edges_capacity, invalid_edges_mask
            )
            valid_mask = torch.where(
                has_valid_action.unsqueeze(1),
                valid_mask,
                fallback_mask
            )

        return valid_mask
