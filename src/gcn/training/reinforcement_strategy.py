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

        # Temperature scheduling parameters
        self.initial_temperature = config.get('rl_sampling_temperature', 1.0)
        self.min_temperature = config.get('rl_min_temperature', 0.5)
        self.temperature_decay = config.get('rl_temperature_decay', 0.05)
        self.current_epoch = 0
        self.max_epochs = config.get('max_epochs', 10)

    def set_epoch(self, epoch):
        """Set current epoch for temperature scheduling."""
        self.current_epoch = epoch

    def get_current_temperature(self):
        """Calculate temperature based on current epoch (linear decay)."""
        # Linear decay: T = max(T_min, T_initial - epoch * decay)
        temperature = max(
            self.min_temperature,
            self.initial_temperature - self.current_epoch * self.temperature_decay
        )
        return temperature

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

            # Use temperature scheduling: start high (exploration), gradually decrease (exploitation)
            current_temp = self.get_current_temperature()

            sampler = PathSampler(
                y_pred_edges=y_preds,
                edges_capacity=x_edges_capacity,
                commodities=batch_commodities,
                num_samples=1,
                temperature=current_temp,  # Use scheduled temperature
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
        debug_enabled = False  # Set to True to enable debug output for load factors
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

        # DEBUG: Check first sample's paths
        debug_metrics = False  # Set to True to enable debug output for metrics calculation
        if debug_metrics and model.training:
            if not hasattr(self, '_debug_metrics_printed'):
                self._debug_metrics_printed = True
                print(f"\n=== Metrics Calculation Debug ===")
                print(f"  pred_paths type: {type(pred_paths)}")
                print(f"  pred_paths length: {len(pred_paths)}")
                if len(pred_paths) > 0:
                    print(f"  pred_paths[0] (first sample): {pred_paths[0]}")
                    print(f"  Number of paths in first sample: {len(pred_paths[0])}")
                    for idx, p in enumerate(pred_paths[0]):
                        print(f"    Path {idx}: {p} (nodes={len(p)}, edges={max(0, len(p)-1)})")
                print("=" * 50)

        for i in range(self.batch_size):
            sample_paths = pred_paths[i]
            sample_commodities = batch_commodities[i]

            if debug_metrics and model.training and i == 0:
                if not hasattr(self, '_debug_path_count_printed'):
                    self._debug_path_count_printed = True
                    print(f"\n=== Path Counting Debug (Sample 0) ===")
                    print(f"  Number of paths: {len(sample_paths)}")
                    print(f"  Number of commodities: {len(sample_commodities)}")

            for path_idx, path in enumerate(sample_paths):
                total_commodities += 1
                dst = int(sample_commodities[path_idx][1].item())

                # Check if path is complete
                if len(path) > 0 and path[-1] == dst:
                    complete_commodities += 1

                # Track path length (number of edges, not nodes)
                # Path length = number of edges = len(path) - 1
                path_edge_count = max(0, len(path) - 1)
                total_path_length += path_edge_count
                path_count += 1

                if debug_metrics and model.training and i == 0 and path_idx < 3:
                    if not hasattr(self, f'_debug_path_{path_idx}_printed'):
                        setattr(self, f'_debug_path_{path_idx}_printed', True)
                        print(f"  Path {path_idx}: {path}")
                        print(f"    dst={dst}, complete={len(path) > 0 and path[-1] == dst}, edges={path_edge_count}")

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

        if debug_metrics and model.training:
            if not hasattr(self, '_debug_metrics_calc_printed'):
                self._debug_metrics_calc_printed = True
                print(f"\n=== Metrics Calculation Results ===")
                print(f"  total_commodities: {total_commodities}")
                print(f"  complete_commodities: {complete_commodities}")
                print(f"  total_path_length: {total_path_length}")
                print(f"  path_count: {path_count}")
                print(f"  avg_path_length: {avg_path_length:.2f}")
                print(f"  complete_paths_rate: {complete_paths_rate:.2f}%")
                print("=" * 50)

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

            # IMPROVED REWARD DESIGN (2025-10-19)
            # Based on analysis: moderate reward span (~20), positive rewards for valid solutions

            # Handle incomplete paths (destination not reached)
            if not all_paths_complete:
                # Count how many paths are incomplete
                num_incomplete = sum(1 for path_idx, path in enumerate(sample_paths)
                                    if len(path) == 0 or path[-1] != int(sample_commodities[path_idx][1].item()))
                # Graduated penalty based on severity (UPDATED 2025-10-19: softer penalties)
                if num_incomplete <= 1:
                    reward_i = -2.0  # Only one path failed - mild penalty
                elif num_incomplete <= 2:
                    reward_i = -5.0  # Two paths failed - moderate penalty
                else:
                    reward_i = -10.0  # Multiple paths failed - severe penalty
            # Handle inf (paths using non-existent edges but all complete)
            elif np.isinf(load_factor_i):
                # Paths are complete but use invalid edges
                reward_i = -5.0  # Less severe than incomplete paths
            elif self.reward_type == 'load_factor':
                # Valid solution with finite load factor
                # Reward range: 7~10 for feasible, 1~5 for infeasible
                if load_factor_i <= 1.0:
                    # Feasible solution - positive reward
                    reward_i = 10.0 - load_factor_i * 3.0  # 7.0 ~ 10.0
                else:
                    # Capacity exceeded - still some reward to encourage valid paths
                    reward_i = 5.0 - load_factor_i * 2.0  # Decreases as violation increases
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
        # UPDATED 2025-10-19: Calculate entropy over next-node selection distribution
        # y_preds shape: [batch, nodes_from, nodes_to, commodities]
        # Entropy measures diversity of next-node choices (14 options per source node)

        if model.training and self.use_sampling:
            current_temp = self.get_current_temperature()
            # Compute next-node selection probabilities (dim=2: which next node to choose)
            probs = F.softmax(y_preds / current_temp, dim=2)
            log_probs_for_entropy = F.log_softmax(y_preds / current_temp, dim=2)
        else:
            # No temperature scaling for evaluation
            probs = F.softmax(y_preds, dim=2)
            log_probs_for_entropy = F.log_softmax(y_preds, dim=2)

        # Entropy: sum over next-node choices (dim=2), average over batch/source/commodities
        entropy = -(probs * log_probs_for_entropy).sum(dim=2).mean()

        # DEBUG: Check entropy calculation (disabled after investigation)
        # Root cause found: Model converging to deterministic policy (entropy_weight too low)
        # if not hasattr(self, '_entropy_debug_printed'):
        #     self._entropy_debug_printed = True
        #     print(f"\n=== Entropy Calculation Debug ===")
        #     print(f"  final entropy: {entropy.item():.6f}")

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
            'capacity_violation_rate': capacity_violation_rate,
            # Add temperature for monitoring
            'temperature': self.get_current_temperature()
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
