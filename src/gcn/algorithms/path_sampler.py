"""
Improved Path Sampler with full masking and feasibility constraints
for Reinforcement Learning (REINFORCE algorithm).
"""

import torch
import torch.nn.functional as F
import numpy as np
from collections import deque


class PathSampler:
    """
    Samples paths probabilistically from GCN edge predictions
    while respecting feasibility constraints (capacity, connectivity).

    This ensures proper on-policy sampling for REINFORCE learning
    and prevents invalid (infeasible) paths.
    """

    def __init__(self, y_pred_edges, edges_capacity, commodities,
                 num_samples=1, temperature=1.0, top_p=0.8, entropy_epsilon: float = 0.0,
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
        self.entropy_epsilon = float(entropy_epsilon) if entropy_epsilon is not None else 0.0
        self.dtypeFloat = dtypeFloat
        self.dtypeLong = dtypeLong

        self.batch_size = y_pred_edges.shape[0]
        self.num_nodes = y_pred_edges.shape[1]

        # Precompute reachability matrix for each batch
        self.reachability = self._precompute_reachability()

    def sample(self):
        """
        Sample paths for all commodities using top-p sampling.

        Note: Capacity constraints are NOT enforced during sampling.
        The model can explore infeasible solutions and learn from
        the penalty in the reward signal.

        Returns:
            batch_paths: list of sampled paths
            log_probs_tensor: tensor of total log-probabilities per batch
            is_feasible: whether all flows respect capacity constraints (checked post-hoc)
        """
        device = self.y_pred_edges.device
        batch_paths, batch_log_probs = [], []
        batch_stepwise_entropies = []  # List[List[List[float]]] per batch -> per commodity -> per step

        # DEBUG: Print shapes for first batch only
        debug_enabled = False  # Set to True to enable debug output
        debug_first_batch = debug_enabled and (not hasattr(self, '_debug_printed'))
        if debug_first_batch:
            self._debug_printed = True
            print(f"\n=== PathSampler Debug ===")
            print(f"  y_pred_edges shape: {self.y_pred_edges.shape}")
            print(f"  edges_capacity shape: {self.edges_capacity.shape}")

        for b in range(self.batch_size):
            # Store batch index for later use in _sample_single_path
            self._current_batch_idx = b
            edge_logits = self.y_pred_edges[b]

            # --- Global mask for physically non-existent edges ---
            # (zero capacity = edge doesn't exist, self-loops are invalid)
            # Use a very large negative value to effectively zero out probabilities after softmax
            invalid_mask = (self.edges_capacity[b] <= 0) | torch.eye(self.num_nodes, device=device).bool()

            if debug_first_batch and b == 0:
                print(f"  edge_logits shape: {edge_logits.shape}")
                print(f"  invalid_mask shape: {invalid_mask.shape}")
                print(f"  Number of valid edges: {(~invalid_mask).sum().item()}/{invalid_mask.numel()}")
                print(f"  Capacity stats - min: {self.edges_capacity[b].min():.2f}, max: {self.edges_capacity[b].max():.2f}, nonzero: {(self.edges_capacity[b] > 0).sum().item()}")

                # Check graph connectivity
                num_isolated_nodes = 0
                for node in range(self.num_nodes):
                    outgoing = (~invalid_mask[node]).sum().item()
                    incoming = (~invalid_mask[:, node]).sum().item()
                    if outgoing == 0 or incoming == 0:
                        num_isolated_nodes += 1
                print(f"  Isolated/dead-end nodes: {num_isolated_nodes}/{self.num_nodes}")

            # --- Apply invalid edge mask to logits ---
            # edge_logits shape: [nodes_from, nodes_to, commodities]
            # Mask out invalid edges (zero capacity or self-loops)
            edge_logits = edge_logits.masked_fill(invalid_mask.unsqueeze(-1), -1e20)

            # --- Convert logits to edge probabilities ---
            # UPDATED 2025-10-19: Compute next-node selection probabilities
            # dim=1 is the "to_node" dimension (14 choices for next node from current node)
            edge_probs_all = F.softmax(edge_logits / self.temperature, dim=1)

            # Ensure zero-capacity edges have exactly zero probability
            edge_probs_all = edge_probs_all * (~invalid_mask).unsqueeze(-1).float()

            if debug_first_batch and b == 0:
                print(f"  edge_probs_all shape: {edge_probs_all.shape}")
                print(f"  edge_probs_all stats - min: {edge_probs_all.min():.6f}, max: {edge_probs_all.max():.6f}, mean: {edge_probs_all.mean():.6f}")
                print(f"  Nonzero probabilities: {(edge_probs_all > 1e-10).sum().item()}/{edge_probs_all.numel()}")
                print("=" * 50)

            # --- Commodity processing ---
            commodity_paths, commodity_log_probs = [], []
            commodity_step_entropies = []

            batch_commodities = (
                self.commodities[b] if isinstance(self.commodities, torch.Tensor) and len(self.commodities.shape) == 3
                else self.commodities
            )

            for c_idx, commodity in enumerate(batch_commodities):
                src, dst, demand = self._parse_commodity(commodity)

                if len(edge_probs_all.shape) == 3:
                    edge_probs = edge_probs_all[:, :, c_idx]
                else:
                    edge_probs = edge_probs_all

                # DEBUG: Print first path
                if debug_first_batch and b == 0 and c_idx == 0:
                    print(f"\n  Commodity {c_idx}: src={src}, dst={dst}, demand={demand}")
                    print(f"  edge_probs for this commodity - nonzero: {(edge_probs > 1e-10).sum().item()}/{edge_probs.numel()}")
                    print(f"  edge_probs from src {src} - nonzero: {(edge_probs[src] > 1e-10).sum().item()}/{len(edge_probs[src])}")
                    # Check if edge to dst is masked
                    print(f"  edge_probs[{src}, {dst}] (direct to dst): {edge_probs[src, dst]:.6f}")
                    print(f"  invalid_mask[{src}, {dst}]: {invalid_mask[src, dst]}")

                # remaining_capacity is no longer tracked or enforced
                path, log_prob, step_entropies = self._sample_single_path(
                    edge_probs, src, dst, demand, remaining_capacity=None
                )

                if debug_first_batch and b == 0 and c_idx == 0:
                    path_length = len(path) - 1 if len(path) > 1 else 0  # Number of edges
                    print(f"  Generated path: {path} (nodes={len(path)}, edges={path_length})")
                    # Check if path reaches destination
                    if len(path) == 0 or path[-1] != dst:
                        print(f"    WARNING: Path incomplete (doesn't reach dst={dst})")
                    elif len(path) == 1:
                        print(f"    WARNING: Zero-length path (src == dst = {src})")
                    # Check if path uses invalid edges
                    uses_invalid = False
                    for i in range(len(path) - 1):
                        u, v = path[i], path[i+1]
                        if invalid_mask[u, v]:
                            print(f"    WARNING: Path uses invalid edge ({u}, {v})")
                            print(f"      edge_probs[{u}, {v}]: {edge_probs[u, v]:.6f}")
                            print(f"      edges_capacity[{b}, {u}, {v}]: {self.edges_capacity[b, u, v]:.2f}")
                            uses_invalid = True
                    print(f"  Uses invalid edges: {uses_invalid}")

                commodity_paths.append(path)
                commodity_log_probs.append(log_prob)
                commodity_step_entropies.append(step_entropies)

            # --- Aggregate batch results ---
            batch_paths.append(commodity_paths)
            batch_log_probs.append(sum(commodity_log_probs))
            batch_stepwise_entropies.append(commodity_step_entropies)

            # DEBUG: Check how many commodities use invalid edges or are incomplete
            if debug_first_batch and b == 0:
                invalid_count = 0
                incomplete_count = 0
                for c_idx, path in enumerate(commodity_paths):
                    dst = int(batch_commodities[c_idx][1].item())
                    # Check if incomplete
                    if len(path) == 0 or path[-1] != dst:
                        incomplete_count += 1
                    # Check if uses invalid edges
                    for i in range(len(path) - 1):
                        u, v = path[i], path[i+1]
                        if invalid_mask[u, v]:
                            invalid_count += 1
                            break  # One invalid edge per commodity is enough
                print(f"  Commodities using invalid edges: {invalid_count}/{len(commodity_paths)}")
                print(f"  Incomplete paths (dst not reached): {incomplete_count}/{len(commodity_paths)}")

        is_feasible = self._check_feasibility(batch_paths)
        log_probs_tensor = torch.tensor(batch_log_probs, dtype=torch.float32, device=device)

        return batch_paths, log_probs_tensor, is_feasible, batch_stepwise_entropies

    # ==============================================================
    # Internal Methods
    # ==============================================================

    def _precompute_reachability(self):
        """
        Precompute reachability matrix for all node pairs in each batch.

        Returns:
            reachability: [batch_size, num_nodes, num_nodes] boolean tensor
                         reachability[b][i][j] = True if node j is reachable from node i in batch b
        """
        device = self.edges_capacity.device
        reachability = torch.zeros(
            (self.batch_size, self.num_nodes, self.num_nodes),
            dtype=torch.bool,
            device=device
        )

        for b in range(self.batch_size):
            # Build adjacency list from capacity matrix
            # edge exists if capacity > 0
            adj_list = [[] for _ in range(self.num_nodes)]
            for i in range(self.num_nodes):
                for j in range(self.num_nodes):
                    # PyTorch scalar comparison works directly
                    if self.edges_capacity[b, i, j] > 0 and i != j:
                        adj_list[i].append(j)

            # BFS from each source node
            for src in range(self.num_nodes):
                # Use deque for efficient BFS (O(1) popleft vs O(n) pop(0))
                queue = deque([src])
                visited = set([src])
                reachability[b, src, src] = True

                while queue:
                    current = queue.popleft()
                    for neighbor in adj_list[current]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            reachability[b, src, neighbor] = True
                            queue.append(neighbor)

        return reachability

    def _parse_commodity(self, commodity):
        """Extract (src, dst, demand) as native Python types."""
        if isinstance(commodity, (list, tuple)):
            src, dst, demand = commodity
        else:
            src = int(commodity[0].item() if hasattr(commodity[0], 'item') else commodity[0])
            dst = int(commodity[1].item() if hasattr(commodity[1], 'item') else commodity[1])
            demand = float(commodity[2].item() if hasattr(commodity[2], 'item') else commodity[2])
        return src, dst, demand

    def _sample_single_path(self, edge_probs, src, dst, demand, remaining_capacity=None):
        """Sample a path using top-p nucleus sampling.

        Note: Capacity constraints are NOT enforced during sampling.
        This allows the model to explore infeasible solutions and learn
        from the penalty in the reward signal.

        Args:
            edge_probs: Edge probabilities for current commodity
            src: Source node
            dst: Destination node
            demand: Commodity demand (not used for capacity checking)
            remaining_capacity: Ignored (kept for API compatibility)
        """
        src, dst = int(src), int(dst)

        path, log_prob = [src], 0.0
        step_entropies: list = []
        current = src
        max_steps = self.num_nodes * 2  # avoid infinite loops

        for step_idx in range(max_steps):
            if current == dst:
                break

            outgoing_probs = edge_probs[current].clone()

            # --- Apply Reachability Mask + Visited Mask ---
            # Get batch index
            batch_idx = self._current_batch_idx if hasattr(self, '_current_batch_idx') else 0

            # Create combined mask: not visited AND reachable to destination
            visited_mask = torch.ones(self.num_nodes, dtype=torch.bool, device=outgoing_probs.device)
            for visited_node in path:
                visited_mask[int(visited_node)] = False  # exclude visited nodes to prevent loops

            # Reachability mask: only nodes that can reach the destination
            # Ensure dst is Python int for indexing
            dst_idx = int(dst)
            reachable_mask = self.reachability[batch_idx, :, dst_idx]  # [num_nodes]

            # Combined mask: not visited AND reachable
            combined_mask = visited_mask & reachable_mask

            # Apply mask
            outgoing_probs = outgoing_probs * combined_mask.float()

            # Remove numerical noise
            outgoing_probs = torch.clamp(outgoing_probs, min=0.0)

            # --- Normalize ---
            total_prob = outgoing_probs.sum()

            if total_prob <= 1e-20:
                # Fallback: No valid edges available with reachability constraint
                # Try to find edges that are: has capacity AND not visited AND reachable
                device = edge_probs.device
                batch_idx = self._current_batch_idx if hasattr(self, '_current_batch_idx') else 0

                if len(self.edges_capacity.shape) == 2:
                    capacity_available = self.edges_capacity[current] > 0
                else:
                    capacity_available = self.edges_capacity[batch_idx, current] > 0

                # Combine: has capacity AND not visited AND reachable
                fallback_mask = capacity_available & combined_mask

                if fallback_mask.any():
                    # Pick valid edge with SOME heuristic guidance
                    # Use unmasked probabilities (from model) as guidance, but only among valid edges
                    valid_indices = torch.where(fallback_mask)[0]

                    # Get model's probabilities for valid edges only
                    unmasked_probs = edge_probs[current].clone()
                    valid_probs = unmasked_probs[valid_indices]

                    if valid_probs.sum() > 1e-10:
                        # If model assigns some probability to valid edges, use it
                        valid_probs = valid_probs / valid_probs.sum()
                        # Sample from model's distribution over valid edges
                        sampled_idx = torch.multinomial(valid_probs, 1).item()
                        next_node = valid_indices[sampled_idx].item()
                        log_prob += torch.log(valid_probs[sampled_idx] + 1e-8).item()
                    else:
                        # Model assigns 0 probability to all valid edges
                        # Fall back to uniform distribution
                        next_node = valid_indices[torch.randint(len(valid_indices), (1,))].item()
                        log_prob += np.log(1.0 / len(valid_indices))

                    path.append(int(next_node))
                    current = int(next_node)
                    continue
                else:
                    # Truly stuck: no valid edges at all from current node
                    # Cannot reach dst without using invalid edges
                    # Return incomplete path (don't force-add dst through invalid edge)
                    log_prob += np.log(1e-8)
                    break

            outgoing_probs /= total_prob

            # --- Epsilon mixture with uniform over valid, non-visited next nodes (ε-greedy over valid set) ---
            eps = self.entropy_epsilon
            if eps > 0.0:
                # Capacity-available mask for current row (respect batch index)
                batch_idx = self._current_batch_idx if hasattr(self, '_current_batch_idx') else 0
                if len(self.edges_capacity.shape) == 2:
                    capacity_available = self.edges_capacity[current] > 0
                else:
                    capacity_available = self.edges_capacity[batch_idx, current] > 0
                # Valid and not-visited AND reachable to destination
                valid_support = capacity_available & combined_mask
                if valid_support.any():
                    uniform = valid_support.float()
                    uniform = uniform / (uniform.sum() + 1e-8)
                    outgoing_probs = (1.0 - eps) * outgoing_probs + eps * uniform
                    outgoing_probs = outgoing_probs / (outgoing_probs.sum() + 1e-8)

            # --- Record step entropy before sampling ---
            step_entropy = -(outgoing_probs * torch.log(outgoing_probs + 1e-8)).sum().item()
            step_entropies.append(step_entropy)

            # --- Top-p nucleus sampling ---
            next_node = self._top_p_sample(outgoing_probs)
            log_prob += torch.log(outgoing_probs[next_node] + 1e-8).item()
            path.append(int(next_node))
            current = int(next_node)

        # Return path as-is (may be incomplete if destination unreachable with valid edges)
        return path, log_prob, step_entropies

    def _top_p_sample(self, probs):
        """Perform top-p (nucleus) sampling robustly."""
        probs = probs * (probs > 1e-8)  # Remove zero-prob nodes
        if probs.sum() == 0:
            # fallback to random selection if all invalid
            return torch.randint(0, len(probs), (1,)).item()

        probs = probs / probs.sum()
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumsum_probs = torch.cumsum(sorted_probs, dim=0)

        cutoff_idx = torch.where(cumsum_probs > self.top_p)[0]
        cutoff_idx = cutoff_idx[0].item() + 1 if len(cutoff_idx) > 0 else len(sorted_probs)

        top_p_probs = sorted_probs[:cutoff_idx]
        top_p_indices = sorted_indices[:cutoff_idx]
        top_p_probs = top_p_probs / top_p_probs.sum()

        sampled_idx = top_p_indices[torch.multinomial(top_p_probs, 1).item()].item()
        return sampled_idx

    def _check_feasibility(self, batch_paths):
        """Verify that total edge usage does not exceed capacities."""
        for b, commodity_paths in enumerate(batch_paths):
            edge_usage = torch.zeros_like(self.edges_capacity[b])
            batch_commodities = (
                self.commodities[b]
                if isinstance(self.commodities, torch.Tensor) and len(self.commodities.shape) == 3
                else self.commodities
            )

            for i, path in enumerate(commodity_paths):
                src, dst, demand = self._parse_commodity(batch_commodities[i])
                for j in range(len(path) - 1):
                    u, v = path[j], path[j + 1]
                    edge_usage[u, v] += demand

            if (edge_usage > self.edges_capacity[b] + 1e-6).any():
                return False
        return True