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
from ..algorithms.path_sampler import PathSampler
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

        # Sampling configuration
        self.use_sampling = config.get('rl_use_sampling', True)  # Use top-p sampling for both training and evaluation (True) or beam search (False)
        self.sampling_temperature = config.get('rl_sampling_temperature', 1.0)  # Lower = more deterministic
        self.sampling_top_p = config.get('rl_sampling_top_p', 0.9)  # Nucleus sampling threshold

        # Advantage normalization for stability
        self.normalize_advantages = config.get('rl_normalize_advantages', True)

        # Smooth penalty for infeasible solutions
        self.use_smooth_penalty = config.get('rl_use_smooth_penalty', True)
        self.penalty_lambda = config.get('rl_penalty_lambda', 5.0)  # Penalty weight for constraint violation

        # Invalid edge masking
        self.mask_invalid_edges = config.get('rl_mask_invalid_edges', True)

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
        # Mask invalid edges (zero capacity) to prevent sampling impossible paths
        y_preds, _ = model.forward(
            x_edges, x_commodities, x_edges_capacity, x_nodes,
            y_edges=None, edge_cw=None, compute_loss=False,
            mask_invalid_edges=self.mask_invalid_edges
        )

        # Generate paths: use top-p sampling for training, beam search for evaluation
        if self.use_sampling and model.training:
            # Probabilistic sampling for training (REINFORCE requires on-policy samples)
            # Note: top-p sampling does not guarantee feasible solutions
            sampler = PathSampler(
                y_pred_edges=y_preds,
                edges_capacity=x_edges_capacity,
                commodities=batch_commodities,
                num_samples=1,
                temperature=self.sampling_temperature,
                top_p=self.sampling_top_p,
                dtypeFloat=torch.float,
                dtypeLong=torch.long
            )
            pred_paths, path_log_probs, is_feasible = sampler.sample()
        else:
            # Beam search for evaluation (guarantees feasible solutions)
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
            # For beam search, we don't have exact log probs, so use surrogate
            path_log_probs = None

        # Compute maximum load factor (our optimization objective)
        mean_maximum_load_factor, individual_load_factors = mean_feasible_load_factor(
            self.batch_size,
            self.num_commodities,
            self.num_nodes,
            pred_paths,
            x_edges_capacity,
            batch_commodities
        )

        # DEBUG: Print load factors for first batch of each epoch
        debug_enabled = False  # Set to True to enable debug output
        if debug_enabled:
            if not hasattr(self, '_debug_last_epoch'):
                self._debug_last_epoch = -1
                self._debug_batch_in_epoch = 0

            # Reset batch counter at start of new epoch (detect by checking if batch counter wrapped)
            if model.training:
                self._debug_batch_in_epoch += 1
                # Print for first batch of epoch or every 5 batches
                if self._debug_batch_in_epoch == 1 or self._debug_batch_in_epoch % 5 == 0:
                    print(f"\n=== Load Factor Debug (Batch {self._debug_batch_in_epoch}) ===")
                    print(f"  Shape: {individual_load_factors.shape}")
                    # Format mean properly
                    mean_val = mean_maximum_load_factor.item() if isinstance(mean_maximum_load_factor, torch.Tensor) else mean_maximum_load_factor
                    mean_str = f"{mean_val:.4f}" if not np.isinf(mean_val) else "inf"
                    print(f"  Mean: {mean_str}")
                    if individual_load_factors.numel() <= 20:  # Only print all values if batch size <= 20
                        print(f"  Values: {individual_load_factors}")
                    else:
                        print(f"  First 5: {individual_load_factors[:5]}")
                    if not torch.isinf(individual_load_factors).all():
                        finite_vals = individual_load_factors[~torch.isinf(individual_load_factors)]
                        if len(finite_vals) > 0:
                            print(f"  Finite - Min: {finite_vals.min():.4f}, Max: {finite_vals.max():.4f}")
                    print(f"  Contains inf: {torch.isinf(individual_load_factors).any()} ({torch.isinf(individual_load_factors).sum().item()}/{len(individual_load_factors)} samples)")
                    print("=" * 50)

        # Calculate path quality metrics BEFORE designing rewards
        total_commodities = 0
        complete_commodities = 0
        total_path_length = 0
        path_count = 0
        finite_solutions = 0
        finite_load_factors = []
        capacity_violations = 0

        for i in range(self.batch_size):
            sample_paths = pred_paths[i]
            sample_commodities = batch_commodities[i]

            for path_idx, path in enumerate(sample_paths):
                total_commodities += 1
                dst = int(sample_commodities[path_idx][1].item())

                # Check if path is complete
                if len(path) > 0 and path[-1] == dst:
                    complete_commodities += 1

                # Track path length
                total_path_length += len(path)
                path_count += 1

            # Check if sample has finite load factor
            load_factor_i = individual_load_factors[i] if i < len(individual_load_factors) else mean_maximum_load_factor
            if isinstance(load_factor_i, torch.Tensor):
                load_factor_i = load_factor_i.item()

            if not np.isinf(load_factor_i) and load_factor_i > 0:
                finite_solutions += 1
                finite_load_factors.append(load_factor_i)
                if load_factor_i > 1.0:
                    capacity_violations += 1

        # Calculate rates
        complete_paths_rate = (complete_commodities / total_commodities * 100) if total_commodities > 0 else 0.0
        finite_solution_rate = (finite_solutions / self.batch_size * 100) if self.batch_size > 0 else 0.0
        avg_finite_load_factor = np.mean(finite_load_factors) if len(finite_load_factors) > 0 else 0.0
        avg_path_length = (total_path_length / path_count) if path_count > 0 else 0.0
        commodity_success_rate = (complete_commodities / total_commodities * 100) if total_commodities > 0 else 0.0
        capacity_violation_rate = (capacity_violations / finite_solutions * 100) if finite_solutions > 0 else 0.0

        # Design reward signal PER SAMPLE
        rewards = []
        for i in range(self.batch_size):
            load_factor_i = individual_load_factors[i] if i < len(individual_load_factors) else mean_maximum_load_factor

            # Convert tensor to float
            if isinstance(load_factor_i, torch.Tensor):
                load_factor_i = load_factor_i.item()

            # Check if all paths reach their destinations
            sample_paths = pred_paths[i]
            sample_commodities = batch_commodities[i]
            all_paths_complete = True
            for path_idx, path in enumerate(sample_paths):
                dst = int(sample_commodities[path_idx][1].item())
                if len(path) == 0 or path[-1] != dst:
                    all_paths_complete = False
                    break

            # Handle incomplete paths (destination not reached)
            if not all_paths_complete:
                # Severe penalty for incomplete paths (couldn't reach destination)
                reward_i = -100.0
            # Handle inf (paths using non-existent edges)
            elif np.isinf(load_factor_i):
                # Severe penalty for using non-existent edges
                reward_i = -100.0
            elif self.reward_type == 'load_factor':
                # Reward = negative load factor (we want to minimize it)
                if self.use_smooth_penalty:
                    # Smooth, graduated penalty system
                    if load_factor_i == 0:
                        # Empty or invalid path
                        reward_i = -50.0
                    elif load_factor_i > 10.0:
                        # Extremely over capacity (likely using wrong edges)
                        reward_i = -50.0
                    elif load_factor_i > 2.0:
                        # Severely over capacity
                        violation = load_factor_i - 1.0
                        penalty = F.softplus(torch.tensor(violation, dtype=torch.float32)).item() * self.penalty_lambda
                        reward_i = -load_factor_i - penalty
                    elif load_factor_i > 1.0:
                        # Slightly over capacity - smaller penalty
                        violation = load_factor_i - 1.0
                        penalty = violation * self.penalty_lambda  # Linear penalty instead of softplus
                        reward_i = -load_factor_i - penalty
                    else:
                        # Feasible solution - just minimize load factor
                        reward_i = -load_factor_i
                else:
                    # Original discrete penalty
                    if load_factor_i > 1 or load_factor_i == 0:
                        reward_i = -10.0  # Infeasible solution - large penalty
                    else:
                        reward_i = -load_factor_i
            elif self.reward_type == 'inverse_load_factor':
                # Reward = inverse of load factor (higher is better)
                # Note: inf is already handled above
                if self.use_smooth_penalty:
                    if load_factor_i == 0:
                        reward_i = -10.0  # Still penalize zero (invalid solution)
                    else:
                        violation = load_factor_i - 1.0
                        penalty = F.softplus(torch.tensor(violation, dtype=torch.float32)).item() * self.penalty_lambda
                        base_reward = 1.0 / load_factor_i if load_factor_i > 0 else -10.0
                        reward_i = base_reward - penalty
                else:
                    # Original discrete penalty
                    if load_factor_i > 1 or load_factor_i == 0:
                        reward_i = -10.0
                    else:
                        reward_i = 1.0 / load_factor_i
            else:
                raise ValueError(f"Unknown reward type: {self.reward_type}")

            rewards.append(reward_i)

        # Convert to tensor
        rewards = torch.tensor(rewards, dtype=torch.float32, device=y_preds.device)

        # Update baseline using moving average PER SAMPLE
        # Initialize baseline if needed
        if self.reward_baseline is None:
            self.reward_baseline = rewards.mean().item()

        if self.use_baseline:
            # Compute advantages per sample
            advantages = rewards - self.reward_baseline

            # Update baseline with batch mean
            batch_mean_reward = rewards.mean().item()
            self.reward_baseline = (
                self.baseline_momentum * self.reward_baseline +
                (1 - self.baseline_momentum) * batch_mean_reward
            )
        else:
            advantages = rewards

        # Normalize advantages for better gradient stability
        # This reduces variance without changing the expected gradient
        if self.normalize_advantages and self.batch_size > 1:
            advantage_mean = advantages.mean()
            advantage_std = advantages.std()

            # DEBUG: Check if advantages have variance
            if advantage_std < 1e-8:
                # No variance in advantages - skip normalization to preserve signal
                pass
            else:
                advantages = (advantages - advantage_mean) / (advantage_std + 1e-8)

        # Compute policy gradient loss PER SAMPLE
        if path_log_probs is not None and model.training:
            # CORRECT REINFORCE: Use log probability of sampled trajectory
            # L = -Σ_i [log π(τ_i) * (R_i - b)]
            # where τ_i is the sampled trajectory for sample i
            # Shape: path_log_probs [batch_size], advantages [batch_size]
            per_sample_loss = -path_log_probs * advantages
            policy_loss = per_sample_loss.mean()  # Average over batch
        else:
            # Fallback for beam search (not theoretically correct, but allows mixed training)
            log_probs = F.log_softmax(y_preds, dim=-1)
            # Use mean advantage for fallback
            mean_advantage = advantages.mean() if isinstance(advantages, torch.Tensor) else advantages
            policy_loss = -log_probs.mean() * mean_advantage

        # Add entropy bonus for exploration
        probs = F.softmax(y_preds, dim=-1)
        log_probs_for_entropy = F.log_softmax(y_preds, dim=-1)
        entropy = -(probs * log_probs_for_entropy).sum(dim=-1).mean()
        entropy_bonus = -self.entropy_weight * entropy

        total_loss = policy_loss + entropy_bonus

        # Store metrics (use batch averages for logging)
        mean_reward = rewards.mean().item()
        reward_std = rewards.std().item() if isinstance(rewards, torch.Tensor) else 0.0
        mean_advantage = advantages.mean().item() if isinstance(advantages, torch.Tensor) else advantages
        advantage_std = advantages.std().item() if isinstance(advantages, torch.Tensor) else 0.0

        # Convert tensors to lists for individual value tracking
        rewards_list = rewards.detach().cpu().tolist() if isinstance(rewards, torch.Tensor) else []
        advantages_list = advantages.detach().cpu().tolist() if isinstance(advantages, torch.Tensor) else []
        load_factors_list = individual_load_factors.detach().cpu().tolist() if isinstance(individual_load_factors, torch.Tensor) else []

        metrics = {
            'mean_load_factor': mean_maximum_load_factor,
            'reward': mean_reward,
            'reward_std': reward_std,
            'advantage': mean_advantage,
            'advantage_std': advantage_std,
            'entropy': entropy.item(),
            'baseline': self.reward_baseline if self.reward_baseline else 0.0,
            'is_feasible': 1 if mean_maximum_load_factor <= 1 and mean_maximum_load_factor > 0 else 0,
            'policy_loss': policy_loss.item(),
            'entropy_bonus': entropy_bonus.item(),
            # Add individual values for proper epoch-level std calculation
            'reward_individual': rewards_list,
            'advantage_individual': advantages_list,
            'load_factor_individual': load_factors_list,
            # Add new path quality metrics
            'complete_paths_rate': complete_paths_rate,
            'finite_solution_rate': finite_solution_rate,
            'avg_finite_load_factor': avg_finite_load_factor,
            'avg_path_length': avg_path_length,
            'commodity_success_rate': commodity_success_rate,
            'capacity_violation_rate': capacity_violation_rate
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
