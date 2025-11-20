"""
A2C (Advantage Actor-Critic) Training Strategy for SeqFlowRL.

Implements the confirmed training algorithm (Decision #4: Option 4-A).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ..algorithms.sequential_rollout import SequentialRolloutEngine


class A2CStrategy:
    """
    A2C Training Strategy for SeqFlowRL.

    Design decision #4: A2C algorithm (confirmed)
    - Actor-Critic architecture
    - Advantage = Reward - Value (baseline)
    - Single-step updates (on-policy)
    - Entropy regularization for exploration
    """

    def __init__(self, model, config):
        """
        Args:
            model: SeqFlowRLModel instance
            config: Configuration dictionary containing:
                - learning_rate: Learning rate (default: 0.0005)
                - entropy_weight: Entropy regularization weight (default: 0.01)
                - value_loss_weight: Value loss weight (default: 0.5)
                - gamma: Discount factor (default: 0.99)
                - normalize_advantages: Normalize advantages (default: True)
                - grad_clip_norm: Gradient clipping norm (default: 1.0)
        """
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device

        # Rollout engine
        self.rollout_engine = SequentialRolloutEngine(model, config)

        # A2C hyperparameters
        self.entropy_weight = config.get('entropy_weight', 0.01)
        self.value_loss_weight = config.get('value_loss_weight', 0.5)
        self.gamma = config.get('gamma', 0.99)
        self.normalize_advantages = config.get('normalize_advantages', True)
        self.grad_clip_norm = config.get('grad_clip_norm', 1.0)

        # Optimizer
        learning_rate = config.get('learning_rate', 0.0005)
        weight_decay = config.get('weight_decay', 0.0001)
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(config.get('adam_beta1', 0.9), config.get('adam_beta2', 0.999))
        )

        # Learning rate scheduler (optional)
        if config.get('lr_scheduler', None) == 'step':
            decay_rate = config.get('decay_rate', 1.2)
            step_size = 10  # Decay every 10 epochs
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=step_size,
                gamma=1.0 / decay_rate
            )
        else:
            self.scheduler = None

        # Metrics tracking
        self.training_stats = {
            'actor_loss': [],
            'critic_loss': [],
            'entropy': [],
            'total_loss': [],
            'rewards': [],
            'advantages': [],
            'load_factors': [],
        }

    def train_step(self, batch_data):
        """
        Single training step using A2C algorithm.

        Args:
            batch_data: Dictionary containing:
                - x_nodes: Node features [B, V, C]
                - x_commodities: Commodity data [B, C, 3]
                - x_edges_capacity: Edge capacity [B, V, V]

        Returns:
            metrics: Dictionary of training metrics
        """
        self.model.train()

        # 0. Compute shortest path distances if needed
        shortest_path_dist = self._compute_shortest_paths(batch_data)

        # 1. Rollout: Generate paths and collect log probs, values
        rollout_results = self.rollout_engine.rollout(
            batch_data,
            mode='train',
            deterministic=False
        )

        # 2. Compute rewards from paths
        rewards, reward_breakdown = self._compute_rewards(rollout_results, batch_data, shortest_path_dist)

        # 3. Compute A2C loss
        loss, loss_components = self._compute_a2c_loss(
            rollout_results,
            rewards
        )

        # 4. Backward pass and optimization
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if self.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.grad_clip_norm
            )

        self.optimizer.step()

        # 5. Collect metrics (pass batch_data for ground truth)
        metrics = self._collect_metrics(loss_components, rewards, rollout_results, batch_data)

        # Add reward breakdown to metrics
        metrics.update(reward_breakdown)

        return metrics

    def _compute_shortest_paths(self, batch_data):
        """
        Compute shortest path distances using Floyd-Warshall algorithm.
        Always computed to enable distance-based reward shaping.

        Args:
            batch_data: Dictionary containing 'x_edges_capacity' [B, V, V]

        Returns:
            shortest_path_dist: Tensor [V, V] of shortest path distances
                                (computed from the first batch element)
        """
        x_edges_capacity = batch_data['x_edges_capacity']  # [B, V, V]
        device = x_edges_capacity.device

        # Use first batch element to compute graph structure
        # (assuming all instances share the same graph topology)
        adjacency = (x_edges_capacity[0] > 0).float()  # [V, V]
        num_nodes = adjacency.shape[0]

        # Initialize distance matrix
        dist = torch.full((num_nodes, num_nodes), float('inf'), device=device)
        dist.fill_diagonal_(0.0)

        # Set distances for existing edges to 1 (unweighted shortest path)
        dist = torch.where(adjacency > 0, torch.ones_like(dist), dist)

        # Floyd-Warshall algorithm
        for k in range(num_nodes):
            dist = torch.minimum(dist, dist[:, k:k+1] + dist[k:k+1, :])

        return dist

    def _compute_rewards(self, rollout_results, batch_data, shortest_path_dist):
        """
        Compute rewards from rollout results.
        Combines load factor reward + capacity violation penalty + completion penalty + distance shaping.

        Reward logic:
        1. Base reward: 2.0 - 2.0 * load_factor
        2. Capacity violation penalty: -penalty_lambda * max(0, load_factor - 1.0)
        3. Completion penalty: -penalty_weight * (incomplete_commodities / total_commodities)
           - Penalizes NOT reaching destinations for all commodities
        4. Distance-based completion shaping:
           - For each commodity, compute progress: 1.0 - (distance_to_goal / max_distance)
           - Average across commodities to get completion_ratio
           - Apply shaping: +weight * (completion_ratio - 0.5)
           - This rewards getting closer to destinations and penalizes being far

        Args:
            rollout_results: Dict containing 'paths', 'edge_usage'
            batch_data: Dict containing 'x_edges_capacity', 'x_commodities'
            shortest_path_dist: Tensor [V, V] with shortest path distances between all node pairs
                                (always provided, computed via Floyd-Warshall)
        Returns:
            rewards: Tensor [B]
            reward_breakdown: Dict with individual reward components (for logging)
        """
        paths = rollout_results['paths']
        edge_usage = rollout_results['edge_usage']  # [B, V, V]
        x_edges_capacity = batch_data['x_edges_capacity']  # [B, V, V]
        x_commodities = batch_data.get('x_commodities', None)

        batch_size = len(paths)
        device = edge_usage.device

        # Track reward components for logging
        reward_breakdown = {}

        # === 1. Load factor base reward ===
        load_factors = self._compute_load_factors(edge_usage, x_edges_capacity)
        base_reward = 2.0 - 2.0 * load_factors
        rewards = base_reward.clone()
        reward_breakdown['base_reward'] = base_reward.mean().item()

        # === 2. Smooth penalty for capacity violation ===
        capacity_penalty = torch.zeros_like(rewards)
        if self.config.get('rl_use_smooth_penalty', True):
            penalty_lambda = self.config.get('rl_penalty_lambda', 5.0)
            violations = torch.clamp(load_factors - 1.0, min=0.0)
            capacity_penalty = -penalty_lambda * violations
            rewards = rewards + capacity_penalty
        reward_breakdown['capacity_penalty'] = capacity_penalty.mean().item()

        # === 3. Completion penalty ===
        # Penalize for NOT reaching destinations (incomplete paths)
        completion_penalty = torch.zeros_like(rewards)
        if x_commodities is not None:
            completion_penalty_weight = self.config.get('rl_completion_penalty_weight', 0.0)
            if completion_penalty_weight != 0.0:
                incomplete_rates = torch.zeros(batch_size, device=device)

                for b in range(batch_size):
                    num_commodities = len(paths[b])
                    incomplete_count = 0
                    for c in range(num_commodities):
                        path = paths[b][c]
                        dst_node = int(x_commodities[b, c, 1].item())

                        # Check if commodity did NOT reach destination
                        if len(path) == 0 or path[-1] != dst_node:
                            incomplete_count += 1

                    # Incomplete rate: 0.0 (all reached) to 1.0 (none reached)
                    incomplete_rates[b] = incomplete_count / max(num_commodities, 1)

                # Apply penalty for incomplete paths (negative weight for penalty)
                completion_penalty = -completion_penalty_weight * incomplete_rates
                rewards = rewards + completion_penalty
        reward_breakdown['completion_penalty'] = completion_penalty.mean().item()

        # === 4. Distance-based completion shaping ===
        # Always apply - uses distance to destination for fine-grained reward
        distance_shaping = torch.zeros_like(rewards)
        if x_commodities is not None and shortest_path_dist is not None:
            shaping_weight = self.config.get('rl_distance_shaping_weight', 0.0)
            if shaping_weight != 0.0:
                completion_ratios = torch.zeros(batch_size, device=device)

                for b in range(batch_size):
                    num_commodities = len(paths[b])
                    total_ratio = 0.0
                    for c in range(num_commodities):
                        path = paths[b][c]
                        src_node = int(x_commodities[b, c, 0].item())
                        dst_node = int(x_commodities[b, c, 1].item())

                        if len(path) == 0:
                            # Empty path: ratio = 0 (no progress)
                            ratio = 0.0
                        else:
                            final_node = path[-1]

                            # Distance from final node to destination
                            d_final = shortest_path_dist[final_node, dst_node]
                            # Maximum possible distance (from source to destination)
                            d_max = shortest_path_dist[src_node, dst_node] + 1e-8

                            # Completion ratio: 1.0 if reached, proportional to progress otherwise
                            ratio = max(0.0, 1.0 - (d_final / d_max))

                        total_ratio += ratio

                    # Average completion ratio across all commodities
                    completion_ratios[b] = total_ratio / max(num_commodities, 1)

                # Apply shaping: 0.5 is neutral, > 0.5 is bonus, < 0.5 is penalty
                distance_shaping = shaping_weight * (completion_ratios - 0.5)
                rewards = rewards + distance_shaping
        reward_breakdown['distance_shaping'] = distance_shaping.mean().item()

        return rewards, reward_breakdown

    def _compute_load_factors(self, edge_usage, edge_capacity):
        """
        Compute maximum load factor for each batch element.

        load_factor = max_edge(usage / capacity)

        Args:
            edge_usage: Edge usage [B, V, V]
            edge_capacity: Edge capacity [B, V, V]

        Returns:
            load_factors: Maximum load factor per batch [B]
        """
        # Avoid division by zero
        epsilon = 1e-8
        load_ratios = edge_usage / (edge_capacity + epsilon)

        # Mask out edges with zero capacity
        valid_edges = edge_capacity > 0
        load_ratios = torch.where(valid_edges, load_ratios, torch.zeros_like(load_ratios))

        # Maximum load factor per batch
        load_factors = load_ratios.view(load_ratios.shape[0], -1).max(dim=1)[0]

        return load_factors

    def _compute_a2c_loss(self, rollout_results, rewards):
        """
        Compute A2C loss components.

        Loss = actor_loss + value_loss_weight * critic_loss - entropy_weight * entropy

        Actor loss (Policy Gradient):
            L_actor = -sum(log_prob * advantage)

        Critic loss (Value function MSE):
            L_critic = MSE(value, reward)

        Entropy bonus (for exploration):
            L_entropy = -sum(entropy)

        Args:
            rollout_results: Dictionary from rollout
            rewards: Rewards [B]

        Returns:
            total_loss: Combined loss
            loss_components: Dictionary of individual loss components
        """
        log_probs = rollout_results['log_probs']  # [B, C]
        entropies = rollout_results['entropies']  # [B, C]
        state_values = rollout_results['state_values']  # [B, C]

        batch_size = log_probs.shape[0]
        num_commodities = log_probs.shape[1]

        # Expand rewards to match log_probs shape
        # rewards: [B] -> [B, C]
        rewards_expanded = rewards.unsqueeze(1).expand(-1, num_commodities)

        # Compute advantages: A = R - V(s)
        # Use detached values as baseline (don't backprop through baseline)
        advantages = rewards_expanded - state_values.detach()

        # Normalize advantages (reduces variance)
        if self.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Actor loss: -E[log π(a|s) * A]
        # Negative because we want to maximize expected reward
        actor_loss = -(log_probs * advantages).mean()

        # Critic loss: MSE between predicted value and actual reward
        critic_loss = F.mse_loss(state_values, rewards_expanded)

        # Entropy bonus (negative because we want to maximize entropy)
        entropy_loss = -entropies.mean()

        # Combined loss
        total_loss = (
            actor_loss +
            self.value_loss_weight * critic_loss +
            self.entropy_weight * entropy_loss
        )

        loss_components = {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'total_loss': total_loss.item(),
            'mean_advantage': advantages.mean().item(),
            'mean_value': state_values.mean().item(),
        }

        return total_loss, loss_components

    def _collect_metrics(self, loss_components, rewards, rollout_results, batch_data=None):
        """
        Collect training metrics.

        Args:
            loss_components: Dictionary of loss components
            rewards: Rewards [B]
            rollout_results: Dictionary from rollout
            batch_data: Original batch data (optional, for ground truth metrics)

        Returns:
            metrics: Dictionary of metrics
        """
        paths = rollout_results['paths']
        edge_usage = rollout_results['edge_usage']

        # Compute load factors for logging
        x_edges_capacity = self.config.get('_batch_capacity', None)
        if x_edges_capacity is not None:
            load_factors = self._compute_load_factors(edge_usage, x_edges_capacity)
        else:
            load_factors = torch.zeros_like(rewards)

        # Path quality metrics
        path_lengths = []
        total_commodities = 0
        completed_commodities = 0

        # Track completion status per instance (for approximation ratio filtering)
        instance_fully_completed = []

        # Extract destination nodes from batch_data
        if batch_data is not None and 'x_commodities' in batch_data:
            x_commodities = batch_data['x_commodities']  # [B, C, 3]
            batch_size = len(paths)
            num_commodities = len(paths[0]) if paths else 0

            for b in range(batch_size):
                instance_completed_count = 0
                for c in range(num_commodities):
                    path = paths[b][c]
                    dst_node = int(x_commodities[b, c, 1].item())
                    path_lengths.append(len(path))
                    total_commodities += 1

                    # Check if path reached destination
                    if len(path) > 0 and path[-1] == dst_node:
                        completed_commodities += 1
                        instance_completed_count += 1

                # Mark instance as fully completed if all commodities reached destination
                instance_fully_completed.append(instance_completed_count == num_commodities)

        # Compute approximation ratio (only for fully completed instances)
        approximation_ratio = None
        if batch_data is not None and 'load_factor' in batch_data and len(instance_fully_completed) > 0:
            gt_load_factors = batch_data['load_factor']  # Ground truth from exact solution [B]

            # Convert to tensor if needed and move to same device
            if not isinstance(gt_load_factors, torch.Tensor):
                gt_load_factors = torch.tensor(gt_load_factors, device=load_factors.device, dtype=load_factors.dtype)
            else:
                gt_load_factors = gt_load_factors.to(load_factors.device)

            # Create mask for fully completed instances
            completion_mask = torch.tensor(instance_fully_completed, device=load_factors.device, dtype=torch.bool)

            # Only compute approximation ratio for fully completed instances
            if completion_mask.any():
                epsilon = 1e-8
                # approximation_ratio = (gt / model) * 100 for each completed instance
                per_instance_ratios = (gt_load_factors[completion_mask] / (load_factors[completion_mask] + epsilon)) * 100.0

                # Additional filter: exclude instances where model_lf is near zero
                valid_mask = load_factors[completion_mask] > epsilon
                if valid_mask.any():
                    approximation_ratio = per_instance_ratios[valid_mask].mean().item()
                else:
                    approximation_ratio = None
            else:
                # No fully completed instances
                approximation_ratio = None
        else:
            # Fallback: just count path lengths
            for batch_paths in paths:
                for path in batch_paths:
                    path_lengths.append(len(path))

        # Compute completion rate
        completion_rate = (completed_commodities / total_commodities * 100.0) if total_commodities > 0 else 0.0

        # Count capacity violations (load_factor > 1.0)
        num_violations = (load_factors > 1.0).sum().item()
        batch_size = load_factors.shape[0]

        metrics = {
            # Losses
            'actor_loss': loss_components['actor_loss'],
            'critic_loss': loss_components['critic_loss'],
            'entropy_loss': loss_components['entropy_loss'],
            'total_loss': loss_components['total_loss'],

            # Values and advantages
            'mean_advantage': loss_components['mean_advantage'],
            'mean_value': loss_components['mean_value'],

            # Rewards and load factors
            'mean_reward': rewards.mean().item(),
            'min_reward': rewards.min().item(),
            'max_reward': rewards.max().item(),
            'mean_load_factor': load_factors.mean().item(),
            'min_load_factor': load_factors.min().item(),
            'max_load_factor': load_factors.max().item(),
            'num_violations': num_violations,
            'batch_size': batch_size,

            # Approximation ratio (if available)
            'approximation_ratio': approximation_ratio,

            # Path statistics
            'mean_path_length': np.mean(path_lengths) if path_lengths else 0.0,
            'max_path_length': np.max(path_lengths) if path_lengths else 0.0,
            'completion_rate': completion_rate,

            # Exploration
            'mean_entropy': rollout_results['mean_entropy'].mean().item(),
        }

        # Add termination statistics if available
        if 'path_termination_stats' in rollout_results:
            term_stats = rollout_results['path_termination_stats']
            total = term_stats.get('total_paths', 0)
            if total > 0:
                metrics['termination_max_steps'] = term_stats.get('max_steps', 0)
                metrics['termination_no_valid_actions'] = term_stats.get('no_valid_actions', 0)
                metrics['termination_reached_destination'] = term_stats.get('reached_destination', 0)
                metrics['termination_max_steps_pct'] = (term_stats.get('max_steps', 0) / total * 100.0)
                metrics['termination_no_valid_actions_pct'] = (term_stats.get('no_valid_actions', 0) / total * 100.0)
                metrics['termination_reached_destination_pct'] = (term_stats.get('reached_destination', 0) / total * 100.0)

        return metrics

    def eval_step(self, batch_data):
        """
        Evaluation step (no gradient updates).

        Args:
            batch_data: Dictionary containing batch data

        Returns:
            metrics: Dictionary of evaluation metrics
        """
        self.model.eval()

        with torch.no_grad():
            # Compute shortest path distances if needed
            shortest_path_dist = self._compute_shortest_paths(batch_data)

            # Rollout with deterministic policy
            rollout_results = self.rollout_engine.rollout(
                batch_data,
                mode='eval',
                deterministic=True  # Greedy action selection
            )

            # Compute rewards
            rewards, reward_breakdown = self._compute_rewards(rollout_results, batch_data, shortest_path_dist)

            # Compute load factors
            load_factors = self._compute_load_factors(
                rollout_results['edge_usage'],
                batch_data['x_edges_capacity']
            )

            # Collect metrics (no loss computation)
            paths = rollout_results['paths']
            path_lengths = []
            total_commodities = 0
            completed_commodities = 0

            # Track completion status per instance (for approximation ratio filtering)
            instance_fully_completed = []

            # Extract destination nodes and compute completion rate
            x_commodities = batch_data['x_commodities']  # [B, C, 3]
            batch_size = len(paths)
            num_commodities = len(paths[0]) if paths else 0

            for b in range(batch_size):
                instance_completed_count = 0
                for c in range(num_commodities):
                    path = paths[b][c]
                    dst_node = int(x_commodities[b, c, 1].item())
                    path_lengths.append(len(path))
                    total_commodities += 1

                    # Check if path reached destination
                    if len(path) > 0 and path[-1] == dst_node:
                        completed_commodities += 1
                        instance_completed_count += 1

                # Mark instance as fully completed if all commodities reached destination
                instance_fully_completed.append(instance_completed_count == num_commodities)

            # Compute completion rate
            completion_rate = (completed_commodities / total_commodities * 100.0) if total_commodities > 0 else 0.0

            # Compute approximation ratio (only for fully completed instances)
            approximation_ratio = None
            if 'load_factor' in batch_data and len(instance_fully_completed) > 0:
                gt_load_factors = batch_data['load_factor']  # Ground truth from exact solution [B]

                # Convert to tensor if needed and move to same device
                if not isinstance(gt_load_factors, torch.Tensor):
                    gt_load_factors = torch.tensor(gt_load_factors, device=load_factors.device, dtype=load_factors.dtype)
                else:
                    gt_load_factors = gt_load_factors.to(load_factors.device)

                # Create mask for fully completed instances
                completion_mask = torch.tensor(instance_fully_completed, device=load_factors.device, dtype=torch.bool)

                # Only compute approximation ratio for fully completed instances
                if completion_mask.any():
                    epsilon = 1e-8
                    # approximation_ratio = (gt / model) * 100 for each completed instance
                    per_instance_ratios = (gt_load_factors[completion_mask] / (load_factors[completion_mask] + epsilon)) * 100.0

                    # Additional filter: exclude instances where model_lf is near zero
                    valid_mask = load_factors[completion_mask] > epsilon
                    if valid_mask.any():
                        approximation_ratio = per_instance_ratios[valid_mask].mean().item()
                    else:
                        approximation_ratio = None
                else:
                    # No fully completed instances
                    approximation_ratio = None

            # Count capacity violations (load_factor > 1.0)
            num_violations = (load_factors > 1.0).sum().item()
            batch_size = load_factors.shape[0]

            metrics = {
                'mean_reward': rewards.mean().item(),
                'mean_load_factor': load_factors.mean().item(),
                'min_load_factor': load_factors.min().item(),
                'max_load_factor': load_factors.max().item(),
                'mean_path_length': np.mean(path_lengths) if path_lengths else 0.0,
                'completion_rate': completion_rate,
                'num_violations': num_violations,
                'batch_size': batch_size,
                'approximation_ratio': approximation_ratio,
            }

            # Add termination statistics if available
            if 'path_termination_stats' in rollout_results:
                term_stats = rollout_results['path_termination_stats']
                total = term_stats.get('total_paths', 0)
                if total > 0:
                    metrics['termination_max_steps'] = term_stats.get('max_steps', 0)
                    metrics['termination_no_valid_actions'] = term_stats.get('no_valid_actions', 0)
                    metrics['termination_reached_destination'] = term_stats.get('reached_destination', 0)
                    metrics['termination_max_steps_pct'] = (term_stats.get('max_steps', 0) / total * 100.0)
                    metrics['termination_no_valid_actions_pct'] = (term_stats.get('no_valid_actions', 0) / total * 100.0)
                    metrics['termination_reached_destination_pct'] = (term_stats.get('reached_destination', 0) / total * 100.0)

        return metrics

    def step_scheduler(self):
        """Step the learning rate scheduler."""
        if self.scheduler is not None:
            self.scheduler.step()

    def get_current_lr(self):
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']
