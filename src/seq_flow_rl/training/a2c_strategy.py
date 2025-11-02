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

        # 1. Rollout: Generate paths and collect log probs, values
        rollout_results = self.rollout_engine.rollout(
            batch_data,
            mode='train',
            deterministic=False
        )

        # 2. Compute rewards from paths
        rewards = self._compute_rewards(rollout_results, batch_data)

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

        return metrics

    def _compute_rewards(self, rollout_results, batch_data):
        """
        Compute rewards from rollout results.

        Uses load factor as the reward signal:
        reward = 2.0 - 2.0 * load_factor

        This gives:
        - reward = 2.0 when load_factor = 0 (best)
        - reward = 0.0 when load_factor = 1 (at capacity)
        - reward < 0.0 when load_factor > 1 (over capacity, penalized)

        Args:
            rollout_results: Dictionary from rollout
            batch_data: Original batch data

        Returns:
            rewards: Tensor [B]
        """
        paths = rollout_results['paths']
        edge_usage = rollout_results['edge_usage']  # [B, V, V]
        x_edges_capacity = batch_data['x_edges_capacity']  # [B, V, V]

        batch_size = len(paths)
        device = edge_usage.device

        # Compute load factor for each batch element
        load_factors = self._compute_load_factors(edge_usage, x_edges_capacity)

        # Convert to rewards (continuous reward function)
        # reward = 2.0 - 2.0 * load_factor
        rewards = 2.0 - 2.0 * load_factors

        # Smooth penalty for infeasible solutions
        if self.config.get('rl_use_smooth_penalty', True):
            penalty_lambda = self.config.get('rl_penalty_lambda', 5.0)
            violations = torch.clamp(load_factors - 1.0, min=0.0)
            penalty = -penalty_lambda * violations
            rewards = rewards + penalty

        return rewards

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

        # Compute approximation ratio if ground truth is available
        approximation_ratio = None
        if batch_data is not None and 'load_factor' in batch_data:
            gt_load_factors = batch_data['load_factor']  # Ground truth from exact solution

            # Handle device mismatch (gt might be on CPU, load_factors on GPU)
            if isinstance(gt_load_factors, torch.Tensor):
                mean_gt_lf = gt_load_factors.mean().item()
            else:
                mean_gt_lf = float(gt_load_factors.mean())

            mean_model_lf = load_factors.mean().item()

            if mean_model_lf > 0:
                # approximation_ratio = (gt / model) * 100
                # Higher is better (closer to optimal)
                approximation_ratio = (mean_gt_lf / mean_model_lf) * 100
            else:
                approximation_ratio = 0.0

        # Path quality metrics
        path_lengths = []
        completion_rates = []
        for batch_paths in paths:
            for path in batch_paths:
                path_lengths.append(len(path))
            # Completion rate would require dst info (skip for now)

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

            # Approximation ratio (if available)
            'approximation_ratio': approximation_ratio,

            # Path statistics
            'mean_path_length': np.mean(path_lengths) if path_lengths else 0.0,
            'max_path_length': np.max(path_lengths) if path_lengths else 0.0,

            # Exploration
            'mean_entropy': rollout_results['mean_entropy'].mean().item(),
        }

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
            # Rollout with deterministic policy
            rollout_results = self.rollout_engine.rollout(
                batch_data,
                mode='eval',
                deterministic=True  # Greedy action selection
            )

            # Compute rewards
            rewards = self._compute_rewards(rollout_results, batch_data)

            # Compute load factors
            load_factors = self._compute_load_factors(
                rollout_results['edge_usage'],
                batch_data['x_edges_capacity']
            )

            # Collect metrics (no loss computation)
            paths = rollout_results['paths']
            path_lengths = []
            for batch_paths in paths:
                for path in batch_paths:
                    path_lengths.append(len(path))

            # Compute approximation ratio if ground truth is available
            approximation_ratio = None
            if 'load_factor' in batch_data:
                gt_load_factors = batch_data['load_factor']  # Ground truth from exact solution

                # Handle device mismatch (gt might be on CPU, load_factors on GPU)
                if isinstance(gt_load_factors, torch.Tensor):
                    mean_gt_lf = gt_load_factors.mean().item()
                else:
                    mean_gt_lf = float(gt_load_factors.mean())

                mean_model_lf = load_factors.mean().item()

                if mean_model_lf > 0:
                    # approximation_ratio = (gt / model) * 100
                    # Higher is better (closer to optimal)
                    approximation_ratio = (mean_gt_lf / mean_model_lf) * 100
                else:
                    approximation_ratio = 0.0

            metrics = {
                'mean_reward': rewards.mean().item(),
                'mean_load_factor': load_factors.mean().item(),
                'min_load_factor': load_factors.min().item(),
                'max_load_factor': load_factors.max().item(),
                'mean_path_length': np.mean(path_lengths) if path_lengths else 0.0,
                'approximation_ratio': approximation_ratio,
            }

        return metrics

    def step_scheduler(self):
        """Step the learning rate scheduler."""
        if self.scheduler is not None:
            self.scheduler.step()

    def get_current_lr(self):
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']
