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

        # Entropy epsilon (epsilon-greedy style uniform mixing over valid next nodes)
        self.entropy_epsilon = config.get('rl_entropy_epsilon', 0.05)

        # Use trajectory entropy instead of grid entropy
        self.use_trajectory_entropy = config.get('rl_use_trajectory_entropy', True)

        # Debug logging controls
        self.debug_logging = config.get('rl_debug_logging', True)
        self.debug_log_frequency = config.get('rl_debug_log_frequency', 20)
        self.grad_log_frequency = config.get('rl_grad_log_frequency', 50)
        self.grad_zero_threshold = config.get('rl_grad_zero_threshold', 1e-8)
        self.debug_log_max_samples = config.get('rl_debug_log_max_samples', 5)

        # Gradient explosion detection
        self.detect_grad_explosion = config.get('rl_detect_grad_explosion', True)
        self.grad_explosion_threshold = config.get('rl_grad_explosion_threshold', 10.0)
        self.grad_norm_history = []  # Track gradient norms over time
        self.grad_norm_history_size = config.get('rl_grad_norm_history_size', 100)
        self.grad_clip_max_norm = config.get('rl_grad_clip_max_norm', 1.0)  # Gradient clipping threshold

        # Internal counters for logging cadence
        self._debug_batch_index = 0
        self._grad_step_index = 0

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
        # Store model reference for gradient debugging
        if hasattr(model, 'module'):
            self._last_model_ref = model.module  # Unwrap DataParallel
        else:
            self._last_model_ref = model

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
                entropy_epsilon=self.entropy_epsilon,
                dtypeFloat=torch.float,
                dtypeLong=torch.long
            )
            pred_paths, path_log_probs, is_feasible, stepwise_entropies = sampler.sample()

            # DEBUG: Check why fallback is used
            if self.debug_logging and self._debug_batch_index % max(1, self.debug_log_frequency) == 0:
                print(f"[RL-SAMPLER-DEBUG] After sampling:")
                print(f"  path_log_probs is None: {path_log_probs is None}")
                print(f"  model.training: {model.training}")
                if path_log_probs is not None:
                    print(f"  path_log_probs.requires_grad: {path_log_probs.requires_grad}")
                    print(f"  path_log_probs.shape: {path_log_probs.shape}")
                    print(f"  path_log_probs dtype: {path_log_probs.dtype}")
                    print(f"  path_log_probs device: {path_log_probs.device}")
                else:
                    print(f"  WARNING: path_log_probs is None - this will trigger fallback mode!")
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
            stepwise_entropies = None

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

            # IMPROVED REWARD DESIGN (2025-10-22)
            # Based on analysis: moderate penalties (1.5x) to encourage complete paths

            # IMPROVED REWARD DESIGN (2025-10-22) - Continuous reward function
            # Target range: -2.0 ~ +2.0 with continuous gradient

            # Handle incomplete paths (destination not reached)
            if not all_paths_complete:
                # Count how many paths are incomplete
                num_incomplete = sum(1 for path_idx, path in enumerate(sample_paths)
                                    if len(path) == 0 or path[-1] != int(sample_commodities[path_idx][1].item()))
                # Graduated penalty based on severity
                if num_incomplete <= 1:
                    reward_i = -0.5  # Only one path failed - mild penalty
                elif num_incomplete <= 2:
                    reward_i = -1.0  # Two paths failed - moderate penalty
                else:
                    reward_i = -2.0  # Multiple paths failed - strong penalty
            # Handle inf (paths using non-existent edges but all complete)
            elif np.isinf(load_factor_i):
                # Paths are complete but use invalid edges
                reward_i = -1.0  # Less severe than incomplete paths
            elif self.reward_type == 'load_factor':
                # Valid solution with finite load factor
                # Continuous reward function: reward = 2.0 - 2.0 * load_factor
                # load_factor = 0.0  -> reward = +2.0 (best)
                # load_factor = 1.0  -> reward =  0.0 (feasible boundary)
                # load_factor = 2.0  -> reward = -2.0 (capacity violation)
                # This ensures C1 continuity at load_factor = 1.0
                reward_i = 2.0 - 2.0 * load_factor_i

                # Clamp to [-2.0, +2.0] for extreme cases
                reward_i = max(-2.0, min(2.0, reward_i))
            else:
                raise ValueError(f"Unknown reward type: {self.reward_type}")

            rewards.append(reward_i)

        # Convert to tensor
        rewards = torch.tensor(rewards, dtype=torch.float32, device=y_preds.device)
        rewards_detached = rewards.detach()

        # Update baseline using moving average PER SAMPLE
        # Initialize baseline if needed
        if self.reward_baseline is None:
            self.reward_baseline = rewards.mean().item()

        baseline_before_update = self.reward_baseline

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
        adv_before_normalization = advantages.detach().clone()

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
        adv_after_normalization = advantages.detach().clone()

        # Compute policy gradient loss PER SAMPLE
        # Record if using fallback mode BEFORE computing loss
        using_fallback_mode = not (path_log_probs is not None and model.training and path_log_probs.requires_grad)

        if not using_fallback_mode:
            # CORRECT REINFORCE: Use log probability of sampled trajectory
            # L = -Σ_i [log π(τ_i) * (R_i - b)]
            # where τ_i is the sampled trajectory for sample i
            # Shape: path_log_probs [batch_size], advantages [batch_size]
            per_sample_loss = -path_log_probs * advantages
            policy_loss = per_sample_loss.mean()  # Average over batch

            # Debug: Check gradient flow
            if self.debug_logging and self._debug_batch_index % max(1, self.debug_log_frequency) == 0:
                print(f"[RL-LOSS-DEBUG] USING REINFORCE LOSS (correct)")
                print(f"[RL-LOSS-DEBUG] path_log_probs: requires_grad={path_log_probs.requires_grad}, "
                      f"mean={path_log_probs.mean().item():.4f}, std={path_log_probs.std().item():.4f}")
                print(f"[RL-LOSS-DEBUG] advantages: mean={advantages.mean().item():.4f}, "
                      f"std={advantages.std().item():.4f}")
                print(f"[RL-LOSS-DEBUG] per_sample_loss: mean={per_sample_loss.mean().item():.4f}, "
                      f"std={per_sample_loss.std().item():.4f}")
                print(f"[RL-LOSS-DEBUG] policy_loss: {policy_loss.item():.6f}, requires_grad={policy_loss.requires_grad}")
        else:
            # Fallback: Use surrogate loss based on model outputs
            # This is an approximation but maintains gradient flow
            log_probs = F.log_softmax(y_preds, dim=-1)
            # Use mean advantage for fallback
            mean_advantage = advantages.mean() if isinstance(advantages, torch.Tensor) else advantages
            policy_loss = -log_probs.mean() * mean_advantage

            if self.debug_logging and self._debug_batch_index % max(1, self.debug_log_frequency) == 0:
                print(f"[RL-LOSS-DEBUG] ⚠️  USING FALLBACK MODE (approximation)")
                print(f"[RL-LOSS-DEBUG] FALLBACK REASON: path_log_probs={path_log_probs is not None}, "
                      f"model.training={model.training}, "
                      f"requires_grad={path_log_probs.requires_grad if path_log_probs is not None else 'N/A'}")
                print(f"[RL-LOSS-DEBUG] policy_loss: {policy_loss.item():.6f}, requires_grad={policy_loss.requires_grad}")

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

        # Entropy over next-node choices (dim=2). Use only rows with >=2 viable options.
        row_entropy = -(probs * log_probs_for_entropy).sum(dim=2)  # [B, V, C]
        multi_choice_mask = (probs > 1e-8).sum(dim=2) >= 2  # [B, V, C]
        if multi_choice_mask.any():
            entropy_grid = row_entropy[multi_choice_mask].mean()
        else:
            entropy_grid = row_entropy.mean()

        # Trajectory entropy from sampler (average across all steps/commodities/batch)
        if stepwise_entropies is not None:
            flat_steps = []
            for per_batch in stepwise_entropies:
                for per_commodity in per_batch:
                    flat_steps.extend(per_commodity)
            if len(flat_steps) > 0:
                traj_entropy = torch.tensor(flat_steps, device=y_preds.device, dtype=torch.float32).mean()
            else:
                traj_entropy = torch.tensor(0.0, device=y_preds.device)
        else:
            traj_entropy = torch.tensor(0.0, device=y_preds.device)

        # Always use grid entropy for gradient computation (connected to y_preds)
        # Trajectory entropy is tracked as a metric but cannot be used for gradients
        # since it's computed from sampled discrete actions
        entropy_bonus = -self.entropy_weight * entropy_grid

        # For monitoring purposes
        if self.use_trajectory_entropy and stepwise_entropies is not None and len(flat_steps) > 0:
            entropy_for_bonus = traj_entropy  # Log trajectory entropy
        else:
            entropy_for_bonus = entropy_grid  # Log grid entropy

        total_loss = policy_loss + entropy_bonus

        # Debug: Check total loss
        if self.debug_logging and self._debug_batch_index % max(1, self.debug_log_frequency) == 0:
            print(f"[RL-LOSS-DEBUG] entropy_grid: {entropy_grid.item():.6f}, requires_grad={entropy_grid.requires_grad}")
            print(f"[RL-LOSS-DEBUG] entropy_bonus: {entropy_bonus.item():.6f}, requires_grad={entropy_bonus.requires_grad}")
            print(f"[RL-LOSS-DEBUG] total_loss: {total_loss.item():.6f}, requires_grad={total_loss.requires_grad}")
            print(f"[RL-LOSS-DEBUG] Loss components: policy_loss={policy_loss.item():.6f}, "
                  f"entropy_bonus={entropy_bonus.item():.6f}, "
                  f"entropy_weight={self.entropy_weight}")

        # Store metrics (use batch averages for logging)
        def _calc_stats(tensor: torch.Tensor):
            if tensor.numel() == 0:
                return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
            detached = tensor.detach()
            mean_val = detached.mean().item()
            if detached.numel() > 1:
                std_val = detached.std(unbiased=False).item()
            else:
                std_val = 0.0
            min_val = detached.min().item()
            max_val = detached.max().item()
            return {'mean': mean_val, 'std': std_val, 'min': min_val, 'max': max_val}

        reward_stats = _calc_stats(rewards_detached)
        adv_pre_stats = _calc_stats(adv_before_normalization)
        adv_post_stats = _calc_stats(adv_after_normalization)
        baseline_after_update = self.reward_baseline
        normalization_applied = not torch.allclose(
            adv_before_normalization, adv_after_normalization, atol=1e-8, rtol=1e-5
        )

        # Use the previously recorded fallback status
        fallback_used = using_fallback_mode

        if self.debug_logging:
            self._debug_batch_index += 1

            def _tensor_preview(values_tensor):
                if values_tensor.numel() == 0:
                    return []
                preview = values_tensor[:self.debug_log_max_samples]
                return [float(x) for x in preview.cpu().tolist()]

            should_log = False
            adv_abs_max = float(adv_before_normalization.abs().max().item()) if adv_before_normalization.numel() > 0 else 0.0
            adv_std_small = adv_pre_stats['std'] < 1e-6
            if adv_abs_max < 1e-6 or adv_std_small:
                should_log = True
            if fallback_used:
                should_log = True
            if self._debug_batch_index % max(1, self.debug_log_frequency) == 0:
                should_log = True

            if should_log:
                print(
                    "[RL-DEBUG] "
                    f"epoch={self.current_epoch} batch={self._debug_batch_index} "
                    f"reward(mean={reward_stats['mean']:.4f}, std={reward_stats['std']:.4f}, "
                    f"min={reward_stats['min']:.4f}, max={reward_stats['max']:.4f}) "
                    f"baseline(before={baseline_before_update:.4f}, after={baseline_after_update:.4f}) "
                    f"adv(pre_mean={adv_pre_stats['mean']:.4f}, pre_std={adv_pre_stats['std']:.4f}, "
                    f"pre_min={adv_pre_stats['min']:.4f}, pre_max={adv_pre_stats['max']:.4f}; "
                    f"post_mean={adv_post_stats['mean']:.4f}, post_std={adv_post_stats['std']:.4f}) "
                    f"norm_applied={normalization_applied} fallback={fallback_used}"
                )
                if adv_abs_max < 1e-6 or adv_std_small:
                    print(
                        "[RL-DEBUG] Advantage signal near zero. "
                        f"adv_abs_max={adv_abs_max:.2e}, adv_pre_std={adv_pre_stats['std']:.2e}, "
                        f"rewards_preview={_tensor_preview(rewards_detached)}, "
                        f"advantages_preview={_tensor_preview(adv_before_normalization)}"
                    )

        mean_reward = reward_stats['mean']
        reward_std = reward_stats['std']
        mean_advantage = adv_post_stats['mean']
        advantage_std = adv_post_stats['std']

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
            'entropy': entropy_grid.item(),
            'traj_entropy': traj_entropy.item(),
            'entropy_used': entropy_for_bonus.item(),
            'baseline': self.reward_baseline if self.reward_baseline else 0.0,
            'is_feasible': 1 if mean_maximum_load_factor <= 1 and mean_maximum_load_factor > 0 else 0,
            'policy_loss': policy_loss.item(),
            'entropy_bonus': entropy_bonus.item(),
            'baseline_before_update': baseline_before_update,
            'baseline_after_update': baseline_after_update,
            'advantage_pre_norm_std': adv_pre_stats['std'],
            'advantage_post_norm_std': adv_post_stats['std'],
            'advantage_pre_norm_min': adv_pre_stats['min'],
            'advantage_pre_norm_max': adv_pre_stats['max'],
            'reward_min': reward_stats['min'],
            'reward_max': reward_stats['max'],
            'normalization_applied': normalization_applied,
            'reinforce_fallback_used': fallback_used,
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
        loss_value = loss.detach().item()

        if self.debug_logging:
            self._grad_step_index += 1
            log_gradients = (self._grad_step_index % max(1, self.grad_log_frequency) == 0)
        else:
            log_gradients = False

        loss.backward()

        # ALWAYS compute gradient norm for explosion detection (even if not logging)
        total_params = 0
        zero_grad_params = 0
        near_zero_grad_params = 0
        nan_grad_params = 0
        inf_grad_params = 0
        grad_norm_sq = 0.0
        max_abs_grad = 0.0

        with torch.no_grad():
            for group in optimizer.param_groups:
                for param in group['params']:
                    total_params += 1
                    if param.grad is None:
                        zero_grad_params += 1
                        continue
                    grad = param.grad

                    # Check for NaN/Inf
                    has_nan = torch.isnan(grad).any().item()
                    has_inf = torch.isinf(grad).any().item()
                    if has_nan:
                        nan_grad_params += 1
                    if has_inf:
                        inf_grad_params += 1

                    grad_norm = grad.norm().item()
                    grad_norm_sq += grad_norm * grad_norm
                    param_max = grad.abs().max().item()
                    max_abs_grad = max(max_abs_grad, param_max)
                    if param_max < self.grad_zero_threshold:
                        near_zero_grad_params += 1

        total_grad_norm = grad_norm_sq ** 0.5

        # Track gradient norm history
        self.grad_norm_history.append(total_grad_norm)
        if len(self.grad_norm_history) > self.grad_norm_history_size:
            self.grad_norm_history.pop(0)

        # Detect gradient explosion
        grad_explosion_detected = False
        if self.detect_grad_explosion and len(self.grad_norm_history) >= 10:
            # Check if current gradient norm is much larger than recent average
            recent_avg = sum(self.grad_norm_history[-10:]) / 10
            if total_grad_norm > self.grad_explosion_threshold * recent_avg and total_grad_norm > 1.0:
                grad_explosion_detected = True
                print(f"\n{'='*80}")
                print(f"⚠️  GRADIENT EXPLOSION DETECTED!")
                print(f"{'='*80}")
                print(f"  Current grad norm: {total_grad_norm:.4e}")
                print(f"  Recent avg (last 10): {recent_avg:.4e}")
                print(f"  Ratio: {total_grad_norm/recent_avg:.2f}x")
                print(f"  Max abs gradient: {max_abs_grad:.4e}")
                print(f"  NaN gradients: {nan_grad_params}/{total_params}")
                print(f"  Inf gradients: {inf_grad_params}/{total_params}")
                print(f"  Loss value: {loss_value:.6f}")
                print(f"{'='*80}\n")

        # Always warn on NaN/Inf regardless of logging settings
        if nan_grad_params > 0 or inf_grad_params > 0:
            print(f"\n⚠️  WARNING: NaN/Inf in gradients detected!")
            print(f"  NaN gradients: {nan_grad_params}/{total_params}")
            print(f"  Inf gradients: {inf_grad_params}/{total_params}")
            print(f"  Grad norm: {total_grad_norm:.4e}\n")

        if log_gradients:
            print(
                "[RL-GRAD] "
                f"step={self._grad_step_index} loss={loss_value:.6f} "
                f"grad_norm={total_grad_norm:.4e} max_abs_grad={max_abs_grad:.4e} "
                f"zero_grad={zero_grad_params}/{total_params} "
                f"near_zero_grad={near_zero_grad_params}/{total_params} "
                f"nan_grad={nan_grad_params} inf_grad={inf_grad_params}"
            )

            # Show gradient norm statistics
            if len(self.grad_norm_history) >= 10:
                recent_avg = sum(self.grad_norm_history[-10:]) / 10
                recent_max = max(self.grad_norm_history[-10:])
                recent_min = min(self.grad_norm_history[-10:])
                print(f"[RL-GRAD-STATS] Recent 10 steps: avg={recent_avg:.4e}, "
                      f"min={recent_min:.4e}, max={recent_max:.4e}, current={total_grad_norm:.4e}")

            # Log layer-by-layer gradients if model is accessible
            # This requires passing the model to backward_step
            if hasattr(self, '_last_model_ref') and self._last_model_ref is not None:
                model = self._last_model_ref
                print("[RL-GRAD-LAYERS] Gradient flow by layer:")
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_mean = param.grad.abs().mean().item()
                        grad_max = param.grad.abs().max().item()
                        grad_norm = param.grad.norm().item()
                        has_nan = torch.isnan(param.grad).any().item()
                        has_inf = torch.isinf(param.grad).any().item()

                        # Show all layers if explosion detected, otherwise only significant ones
                        if grad_explosion_detected or grad_max > self.grad_zero_threshold:
                            status = ""
                            if has_nan:
                                status += " [NaN]"
                            if has_inf:
                                status += " [Inf]"
                            if grad_norm > 10.0:
                                status += " [LARGE]"
                            print(f"  {name}: grad_norm={grad_norm:.4e}, mean={grad_mean:.4e}, max={grad_max:.4e}{status}")
                    else:
                        print(f"  {name}: NO GRADIENT")

        # Apply gradient clipping before optimizer step
        if (batch_num + 1) % accumulation_steps == 0:
            # Gradient clipping
            if self.grad_clip_max_norm > 0:
                params_to_clip = [p for group in optimizer.param_groups for p in group['params'] if p.grad is not None]
                grad_norm_before_clip = torch.nn.utils.clip_grad_norm_(params_to_clip, self.grad_clip_max_norm)

                if log_gradients or (grad_explosion_detected and self.debug_logging):
                    print(f"[RL-GRAD-CLIP] Gradient clipped: {grad_norm_before_clip:.4e} -> {min(grad_norm_before_clip, self.grad_clip_max_norm):.4e} (max_norm={self.grad_clip_max_norm})")

            optimizer.step()
            optimizer.zero_grad()
            return True

        return False

    def reset_metrics(self):
        """Reset metrics for new epoch (but keep baseline)."""
        super().reset_metrics()
        # Keep baseline across epochs for stability
        self._debug_batch_index = 0
        self._grad_step_index = 0
