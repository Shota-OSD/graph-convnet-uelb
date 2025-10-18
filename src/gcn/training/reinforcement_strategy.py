"""
Reinforcement Learning Strategy

This strategy implements a policy gradient approach where the model
learns directly from the maximum load factor (environment reward)
rather than supervised labels.
"""

import torch
import torch.nn.functional as F
import numpy as np

from .base_strategy import BaseTrainingStrategy
from ..algorithms.beamsearch_uelb import BeamsearchUELB, BeamSearchFactory
from ..models.model_utils import mean_feasible_load_factor


class ReinforcementLearningStrategy(BaseTrainingStrategy):
    """
    Reinforcement learning strategy using maximum load factor as reward.

    This strategy uses REINFORCE (Policy Gradient) algorithm:
    1. Model outputs edge probabilities (policy)
    2. Beam search generates paths based on these probabilities
    3. Maximum load factor is computed as reward signal
    4. Policy is updated to maximize expected reward

    Key differences from supervised learning:
    - No ground truth labels required
    - Learns directly from load factor (optimization objective)
    - Uses policy gradient instead of cross-entropy loss
    """

    def __init__(self, config):
        super().__init__(config)
        self.beam_size = config.get('beam_size', 1280)
        self.batch_size = config.get('batch_size', 1)
        self.num_nodes = config.get('num_nodes', 14)
        self.num_commodities = config.get('num_commodities', 5)

        # RL-specific hyperparameters
        self.reward_type = config.get('rl_reward_type', 'load_factor')  # 'load_factor' or 'inverse_load_factor'
        self.use_baseline = config.get('rl_use_baseline', True)
        self.baseline_momentum = config.get('rl_baseline_momentum', 0.9)
        self.entropy_weight = config.get('rl_entropy_weight', 0.01)  # For exploration

        # Beam search algorithm selection
        self.beam_search_type = config.get('rl_beam_search_type', 'standard')  # 'standard', 'unconstrained', etc.

        # Baseline for variance reduction
        self.reward_baseline = None

    def compute_loss(self, model, batch_data, device=None):
        """
        Compute policy gradient loss using load factor as reward.

        Args:
            model: GCN model (outputs edge probabilities)
            batch_data: Dictionary with inputs and capacities
            device: Device for computation

        Returns:
            loss: Policy gradient loss
            metrics: Dictionary with 'mean_load_factor', 'reward', 'entropy'
        """
        x_edges = batch_data['x_edges']
        x_commodities = batch_data['x_commodities']
        x_edges_capacity = batch_data['x_edges_capacity']
        x_nodes = batch_data['x_nodes']
        batch_commodities = batch_data['batch_commodities']

        # Forward pass (get predictions without computing supervised loss)
        y_preds, _ = model.forward(
            x_edges, x_commodities, x_edges_capacity, x_nodes,
            y_edges=None, edge_cw=None, compute_loss=False
        )

        # Use beam search to generate paths based on model predictions
        # Select beam search algorithm based on configuration
        beam_search = BeamSearchFactory.create_algorithm(
            self.beam_search_type,
            y_pred_edges=y_preds,
            beam_size=self.beam_size,
            batch_size=self.batch_size,
            edges_capacity=x_edges_capacity,
            commodities=batch_commodities,
            dtypeFloat=torch.float,
            dtypeLong=torch.long,
            mode_strict=True
        )
        pred_paths, is_feasible = beam_search.search()

        # Compute maximum load factor (our optimization objective)
        mean_maximum_load_factor, individual_load_factors = mean_feasible_load_factor(
            self.batch_size,
            self.num_commodities,
            self.num_nodes,
            pred_paths,
            x_edges_capacity,
            batch_commodities
        )

        # Design reward signal
        if self.reward_type == 'load_factor':
            # Reward = negative load factor (we want to minimize it)
            # Clamp to avoid extreme values
            if mean_maximum_load_factor > 1 or mean_maximum_load_factor == 0:
                # Infeasible solution - large penalty
                reward = -10.0
            else:
                reward = -mean_maximum_load_factor
        elif self.reward_type == 'inverse_load_factor':
            # Reward = inverse of load factor (higher is better)
            if mean_maximum_load_factor > 1 or mean_maximum_load_factor == 0:
                reward = -10.0
            else:
                reward = 1.0 / mean_maximum_load_factor
        else:
            raise ValueError(f"Unknown reward type: {self.reward_type}")

        # Update baseline using moving average
        if self.use_baseline:
            if self.reward_baseline is None:
                self.reward_baseline = reward
            else:
                self.reward_baseline = (
                    self.baseline_momentum * self.reward_baseline +
                    (1 - self.baseline_momentum) * reward
                )
            advantage = reward - self.reward_baseline
        else:
            advantage = reward

        # Compute policy gradient loss
        # We need to compute log probabilities of selected actions
        log_probs = F.log_softmax(y_preds, dim=-1)

        # Extract log probabilities for the paths chosen by beam search
        # This is a simplified version - full implementation would track exact action sequence
        # For now, we compute average log prob across all predictions
        policy_loss = -log_probs.mean() * advantage

        # Add entropy bonus for exploration
        probs = F.softmax(y_preds, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        entropy_bonus = -self.entropy_weight * entropy

        total_loss = policy_loss + entropy_bonus

        # Store metrics
        metrics = {
            'mean_load_factor': mean_maximum_load_factor,
            'reward': reward,
            'advantage': advantage,
            'entropy': entropy.item(),
            'baseline': self.reward_baseline if self.reward_baseline else 0.0,
            'is_feasible': 1 if mean_maximum_load_factor <= 1 and mean_maximum_load_factor > 0 else 0
        }

        return total_loss, metrics

    def backward_step(self, loss, optimizer, accumulation_steps=1, batch_num=0):
        """
        Perform backward pass for policy gradient.

        Args:
            loss: Policy gradient loss
            optimizer: PyTorch optimizer
            accumulation_steps: Number of batches to accumulate gradients
            batch_num: Current batch number

        Returns:
            bool: True if optimizer step was performed
        """
        # Policy gradient backward
        loss = loss / accumulation_steps
        loss.backward()

        # Update weights every accumulation_steps batches
        if (batch_num + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            return True

        return False

    def reset_metrics(self):
        """Reset metrics for new epoch (but keep baseline)."""
        super().reset_metrics()
        # Keep baseline across epochs for stability
