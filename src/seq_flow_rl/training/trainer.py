"""
Trainer for SeqFlowRL.

Manages the training loop, evaluation, and checkpointing.
"""

import os
import time
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

from ..models.seqflowrl_model import SeqFlowRLModel
from .a2c_strategy import A2CStrategy
from src.common.types import BatchData, validate_batch_types


class SeqFlowRLTrainer:
    """
    Main trainer class for SeqFlowRL.

    Responsibilities:
    - Model instantiation
    - Training loop management
    - Evaluation and validation
    - Checkpointing and logging
    - Learning rate scheduling
    """

    def __init__(self, config, dtypeFloat=torch.float32, dtypeLong=torch.long):
        """
        Args:
            config: Configuration dictionary
            dtypeFloat: Float data type
            dtypeLong: Long data type
        """
        self.config = config
        self.dtypeFloat = dtypeFloat
        self.dtypeLong = dtypeLong

        # Device setup
        self.device = self._setup_device()

        # Model instantiation
        self.model = self._instantiate_model()

        # Training strategy (A2C)
        self.strategy = A2CStrategy(self.model, config)
        print(f"✓ Using A2C Training Strategy")
        print(f"  - Entropy weight: {config.get('entropy_weight', 0.01)}")
        print(f"  - Value loss weight: {config.get('value_loss_weight', 0.5)}")
        print(f"  - Learning rate: {config.get('learning_rate', 0.0005)}")

        # Checkpointing
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'saved_models/seqflowrl/'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_every = config.get('save_every', 5)
        self.save_best_only = config.get('save_best_only', True)

        # Logging
        self.log_dir = Path('logs')
        self.log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = self.log_dir / f'seqflowrl_training_{timestamp}.txt'

        # Training state
        self.current_epoch = 0
        self.best_val_load_factor = float('inf')
        self.training_history = {
            'train_loss': [],
            'train_reward': [],
            'train_load_factor': [],
            'train_approx_ratio': [],
            'val_load_factor': [],
            'val_reward': [],
            'val_approx_ratio': [],
            'learning_rate': [],
            'epoch_times': [],
        }
        self.training_start_time = None

    def _setup_device(self):
        """Setup computation device."""
        use_gpu = self.config.get('use_gpu', True)
        gpu_id = self.config.get('gpu_id', '0')

        if use_gpu and torch.cuda.is_available():
            device = torch.device(f'cuda:{gpu_id}')
            print(f"✓ Using GPU: {gpu_id}")
        else:
            device = torch.device('cpu')
            print(f"✓ Using CPU")

        return device

    def _instantiate_model(self):
        """Instantiate SeqFlowRL model."""
        model = SeqFlowRLModel(self.config, self.dtypeFloat, self.dtypeLong)
        model = model.to(self.device)

        # Load pretrained model (if 2-phase training)
        if self.config.get('load_pretrained_model', False):
            pretrained_path = self.config.get('pretrained_model_path')
            if pretrained_path and os.path.exists(pretrained_path):
                print(f"\n{'='*70}")
                print(f"LOADING PRE-TRAINED MODEL (2-PHASE TRAINING)")
                print(f"{'='*70}")
                success = model.load_pretrained_actor(pretrained_path, device=self.device)
                if success:
                    print(f"✓ Phase 1 (Supervised) → Phase 2 (RL Fine-tuning)")
                print(f"{'='*70}\n")
            else:
                print(f"Warning: Pretrained model path not found: {pretrained_path}")

        # Print model summary
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n{'='*70}")
        print(f"MODEL SUMMARY")
        print(f"{'='*70}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Architecture:")
        print(f"    - Encoder layers: {self.config.get('num_layers', 8)}")
        print(f"    - Hidden dim: {self.config.get('hidden_dim', 128)}")
        print(f"    - Action type: {self.config.get('action_type', 'node')}")
        print(f"    - GNN update: {self.config.get('gnn_update_frequency', 'per_commodity')}")
        print(f"{'='*70}\n")

        return model

    def train(self, train_loader, val_loader=None, num_epochs=None):
        """
        Main training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            num_epochs: Number of epochs (overrides config if provided)

        Returns:
            training_history: Dictionary of training metrics
        """
        if num_epochs is None:
            num_epochs = self.config.get('max_epochs', 50)

        print(f"\n{'='*70}")
        print(f"STARTING TRAINING")
        print(f"{'='*70}")
        print(f"  Epochs: {num_epochs}")
        print(f"  Batch size: {self.config.get('batch_size', 32)}")
        print(f"  Training batches: {train_loader.max_iter}")
        if val_loader:
            print(f"  Validation batches: {val_loader.max_iter}")
        print(f"  Log file: {self.log_file}")
        print(f"{'='*70}\n")

        # Start training timer
        self.training_start_time = time.time()

        val_every = self.config.get('val_every', 5)

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()

            # Training
            train_metrics = self._train_epoch(train_loader, epoch)

            # Validation
            val_metrics = None
            if val_loader and (epoch + 1) % val_every == 0:
                val_metrics = self._validate_epoch(val_loader, epoch)

            # Learning rate scheduling
            self.strategy.step_scheduler()
            current_lr = self.strategy.get_current_lr()

            # Logging
            epoch_time = time.time() - epoch_start_time
            self._log_epoch(epoch, train_metrics, val_metrics, current_lr, epoch_time)

            # Update training history
            self._update_history(train_metrics, val_metrics, current_lr, epoch_time)

            # Checkpointing
            if (epoch + 1) % self.save_every == 0:
                self._save_checkpoint(epoch, train_metrics, val_metrics)

            # Save best model (only if validation was performed)
            if val_metrics is not None and self.save_best_only:
                if val_metrics['mean_load_factor'] < self.best_val_load_factor:
                    self.best_val_load_factor = val_metrics['mean_load_factor']
                    self._save_checkpoint(epoch, train_metrics, val_metrics, is_best=True)
                    print(f"  ★ New best validation load factor: {self.best_val_load_factor:.4f}")

        # Calculate total training time
        total_training_time = time.time() - self.training_start_time

        print(f"\n{'='*70}")
        print(f"TRAINING COMPLETED")
        print(f"{'='*70}")
        if self.save_best_only:
            print(f"  Best validation load factor: {self.best_val_load_factor:.4f}")
        print(f"  Total training time: {total_training_time:.2f}s")
        print(f"{'='*70}\n")

        # Save training log to file
        self._save_training_log(total_training_time)

        return self.training_history

    def _prepare_batch(self, batch: Any) -> BatchData:
        """
        Convert DotDict batch from DatasetReader to dictionary format expected by strategy.

        DatasetReader returns:
            - batch.edges: Adjacency matrix [B, V, V] (numpy int)
            - batch.edges_capacity: Capacity matrix [B, V, V] (numpy float)
            - batch.nodes: Node features [B, V, C] (numpy int: 0=none, 1=source, 2=target)
            - batch.commodities: Commodity data [B, C, 3] (numpy int: source, target, demand)
            - batch.nodes_target: Node flow targets [B, C, V] (numpy int)
            - batch.edges_target: Edge flow targets [B, V, V, C] (numpy int)
            - batch.load_factor: Optimal load factor [B] (numpy float)

        Strategy expects (with correct dtypes):
            - x_nodes: torch.long (for embedding layer)
            - x_commodities: torch.float
            - x_edges_capacity: torch.float
            - x_edges: torch.long
            - edges_target: torch.long
            - nodes_target: torch.long
            - load_factor: torch.float

        Args:
            batch: DotDict from DatasetReader

        Returns:
            batch_data: Dictionary with PyTorch tensors and correct dtypes

        Raises:
            TypeError: If tensor dtypes don't match expected types
        """
        batch_data: BatchData = {
            'x_nodes': torch.from_numpy(batch.nodes).long(),  # CRITICAL: must be long for nn.Embedding
            'x_commodities': torch.from_numpy(batch.commodities).float(),
            'x_edges_capacity': torch.from_numpy(batch.edges_capacity).float(),
            'x_edges': torch.from_numpy(batch.edges).long(),
            'edges_target': torch.from_numpy(batch.edges_target).long(),
            'nodes_target': torch.from_numpy(batch.nodes_target).long(),
            'load_factor': torch.from_numpy(batch.load_factor).float(),
        }

        # Runtime validation (can be disabled in production for performance)
        if self.config.get('validate_batch_types', True):
            validate_batch_types(batch_data, strict=True)

        return batch_data

    def _train_epoch(self, train_loader, epoch):
        """Train for one epoch.

        Args:
            train_loader: DatasetReader instance (iterator)
            epoch: Current epoch number
        """
        self.model.train()

        epoch_metrics = {
            'actor_loss': [],
            'critic_loss': [],
            'total_loss': [],
            'mean_reward': [],
            'mean_load_factor': [],
            'mean_entropy': [],
            'approximation_ratio': [],
        }

        # DatasetReader has max_iter attribute
        num_batches = train_loader.max_iter
        log_every = self.config.get('log_every', 10)

        # Create iterator from DatasetReader
        dataset_iter = iter(train_loader)

        for batch_idx in range(num_batches):
            try:
                batch = next(dataset_iter)
            except StopIteration:
                break

            # Convert DotDict batch to dictionary format expected by strategy
            batch_data = self._prepare_batch(batch)

            # Move data to device
            batch_data = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch_data.items()}

            # Store capacity for reward computation (workaround)
            self.config['_batch_capacity'] = batch_data['x_edges_capacity']

            # Training step
            metrics = self.strategy.train_step(batch_data)

            # Accumulate metrics
            for key in epoch_metrics:
                if key in metrics:
                    epoch_metrics[key].append(metrics[key])

            # Logging
            if (batch_idx + 1) % log_every == 0:
                self._log_batch(epoch, batch_idx, num_batches, metrics)

        # Average metrics over epoch
        avg_metrics = {}
        for k, v in epoch_metrics.items():
            if not v:
                avg_metrics[k] = 0.0
            elif k == 'approximation_ratio':
                # Filter out None values for approximation ratio
                valid_values = [x for x in v if x is not None]
                avg_metrics[k] = np.mean(valid_values) if valid_values else None
            else:
                avg_metrics[k] = np.mean(v)

        return avg_metrics

    def _validate_epoch(self, val_loader, epoch):
        """Validate for one epoch.

        Args:
            val_loader: DatasetReader instance (iterator)
            epoch: Current epoch number
        """
        self.model.eval()

        # DatasetReader has max_iter attribute
        num_batches = val_loader.max_iter

        # If no validation batches, return None (skip validation)
        if num_batches == 0:
            return None

        epoch_metrics = {
            'mean_reward': [],
            'mean_load_factor': [],
            'min_load_factor': [],
            'max_load_factor': [],
            'approximation_ratio': [],
        }

        # Create iterator from DatasetReader
        dataset_iter = iter(val_loader)

        with torch.no_grad():
            for batch_idx in range(num_batches):
                try:
                    batch = next(dataset_iter)
                except StopIteration:
                    break

                # Convert DotDict batch to dictionary format expected by strategy
                batch_data = self._prepare_batch(batch)

                # Move data to device
                batch_data = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                             for k, v in batch_data.items()}

                # Evaluation step
                metrics = self.strategy.eval_step(batch_data)

                # Accumulate metrics
                for key in epoch_metrics:
                    if key in metrics:
                        epoch_metrics[key].append(metrics[key])

        # Average metrics (only if we have data)
        if not epoch_metrics['mean_load_factor']:
            return None

        avg_metrics = {}
        for k, v in epoch_metrics.items():
            if not v:
                avg_metrics[k] = 0.0
            elif k == 'approximation_ratio':
                # Filter out None values for approximation ratio
                valid_values = [x for x in v if x is not None]
                avg_metrics[k] = np.mean(valid_values) if valid_values else None
            else:
                avg_metrics[k] = np.mean(v)

        return avg_metrics

    def _log_epoch(self, epoch, train_metrics, val_metrics, lr, epoch_time):
        """Log epoch summary."""
        print(f"\nEpoch {epoch + 1}/{self.config.get('max_epochs', 50)} | Time: {epoch_time:.2f}s | LR: {lr:.6f}")

        # Training metrics with approximation ratio
        approx_ratio_str = ""
        if train_metrics.get('approximation_ratio') is not None:
            approx_ratio_str = f" | Approx Ratio: {train_metrics.get('approximation_ratio'):.2f}%"

        print(f"  Train - Loss: {train_metrics.get('total_loss', 0):.4f} | "
              f"Reward: {train_metrics.get('mean_reward', 0):.4f} | "
              f"Load Factor: {train_metrics.get('mean_load_factor', 0):.4f}"
              f"{approx_ratio_str}")

        # Validation metrics with approximation ratio
        if val_metrics is not None:
            val_approx_str = ""
            if val_metrics.get('approximation_ratio') is not None:
                val_approx_str = f" | Approx Ratio: {val_metrics.get('approximation_ratio'):.2f}%"

            print(f"  Val   - Load Factor: {val_metrics.get('mean_load_factor', 0):.4f} "
                  f"(min: {val_metrics.get('min_load_factor', 0):.4f}, "
                  f"max: {val_metrics.get('max_load_factor', 0):.4f})"
                  f"{val_approx_str}")
        else:
            print(f"  Val   - Skipped (insufficient validation data)")

    def _log_batch(self, epoch, batch_idx, num_batches, metrics):
        """Log batch progress."""
        print(f"  [{epoch + 1}][{batch_idx + 1}/{num_batches}] "
              f"Loss: {metrics.get('total_loss', 0):.4f} | "
              f"Reward: {metrics.get('mean_reward', 0):.4f} | "
              f"LF: {metrics.get('mean_load_factor', 0):.4f}")

    def _update_history(self, train_metrics, val_metrics, lr, epoch_time):
        """Update training history."""
        self.training_history['train_loss'].append(train_metrics.get('total_loss', 0))
        self.training_history['train_reward'].append(train_metrics.get('mean_reward', 0))
        self.training_history['train_load_factor'].append(train_metrics.get('mean_load_factor', 0))
        self.training_history['train_approx_ratio'].append(train_metrics.get('approximation_ratio', None))
        self.training_history['learning_rate'].append(lr)
        self.training_history['epoch_times'].append(epoch_time)

        if val_metrics is not None:
            self.training_history['val_load_factor'].append(val_metrics.get('mean_load_factor', 0))
            self.training_history['val_reward'].append(val_metrics.get('mean_reward', 0))
            self.training_history['val_approx_ratio'].append(val_metrics.get('approximation_ratio', None))

    def _save_checkpoint(self, epoch, train_metrics, val_metrics, is_best=False):
        """Save model checkpoint."""
        # Convert config to dict if it's a Settings object
        config_dict = dict(self.config) if hasattr(self.config, '__iter__') else self.config

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.strategy.optimizer.state_dict(),
            'config': config_dict,
            'training_history': self.training_history,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
        }

        if is_best:
            path = self.checkpoint_dir / 'best_model.pt'
            print(f"  → Saving best model to {path}")
        else:
            path = self.checkpoint_dir / f'checkpoint_epoch_{epoch + 1}.pt'
            print(f"  → Saving checkpoint to {path}")

        torch.save(checkpoint, path)

        # Also save latest
        latest_path = self.checkpoint_dir / 'latest_model.pt'
        torch.save(checkpoint, latest_path)

    def load_checkpoint(self, checkpoint_path):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.strategy.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.training_history = checkpoint.get('training_history', self.training_history)

        print(f"✓ Loaded checkpoint from epoch {self.current_epoch + 1}")

        return checkpoint

    def _save_training_log(self, total_training_time):
        """Save training log to file (similar to GCN logs format)."""
        with open(self.log_file, 'w') as f:
            # Header
            f.write("=" * 50 + "\n")
            f.write("SEQFLOWRL TRAINING RESULTS\n")
            f.write("=" * 50 + "\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y%m%d_%H%M%S')}\n")
            f.write(f"Config: {self.config.get('expt_name', 'seqflowrl_base')}\n")
            f.write(f"Model: SeqFlowRL with {self.config.get('rl_algorithm', 'a2c').upper()}\n")
            f.write("\n")

            # Final metrics
            f.write("FINAL METRICS:\n")
            f.write(f"  Final Train Loss: {self.training_history['train_loss'][-1]:.4f}\n")
            f.write(f"  Final Train Load Factor: {self.training_history['train_load_factor'][-1]:.4f}\n")
            if self.training_history['train_approx_ratio'][-1] is not None:
                f.write(f"  Final Train Approx Ratio: {self.training_history['train_approx_ratio'][-1]:.2f}%\n")
            if self.training_history['val_load_factor']:
                f.write(f"  Best Val Load Factor: {self.best_val_load_factor:.4f}\n")
            f.write("\n")

            # Time metrics
            f.write("TIME METRICS:\n")
            f.write(f"  Total Training Time: {total_training_time:.2f}s\n")
            avg_epoch_time = np.mean(self.training_history['epoch_times'])
            f.write(f"  Average Time per Epoch: {avg_epoch_time:.2f}s\n")
            f.write(f"  Total Epochs: {len(self.training_history['train_loss'])}\n")
            f.write("\n")

            # Configuration
            f.write("CONFIGURATION:\n")
            f.write(f"  Batch Size: {self.config.get('batch_size', 32)}\n")
            f.write(f"  Learning Rate: {self.config.get('learning_rate', 0.0005)}\n")
            f.write(f"  Hidden Dim: {self.config.get('hidden_dim', 128)}\n")
            f.write(f"  Num Layers: {self.config.get('num_layers', 8)}\n")
            f.write(f"  Action Type: {self.config.get('action_type', 'node')}\n")
            f.write(f"  GNN Update: {self.config.get('gnn_update_frequency', 'per_commodity')}\n")
            f.write(f"  Entropy Weight: {self.config.get('entropy_weight', 0.01)}\n")
            f.write(f"  Value Loss Weight: {self.config.get('value_loss_weight', 0.5)}\n")
            f.write("\n")

            # Detailed results table
            f.write("=" * 50 + "\n")
            f.write("DETAILED EPOCH RESULTS\n")
            f.write("=" * 50 + "\n")
            f.write(f"{'Epoch':<8}{'Loss':<12}{'Reward':<12}{'Load Factor':<14}{'Approx %':<12}{'Time (s)':<10}\n")
            f.write("-" * 68 + "\n")

            for i in range(len(self.training_history['train_loss'])):
                epoch = i + 1
                loss = self.training_history['train_loss'][i]
                reward = self.training_history['train_reward'][i]
                lf = self.training_history['train_load_factor'][i]
                approx = self.training_history['train_approx_ratio'][i]
                epoch_time = self.training_history['epoch_times'][i]

                approx_str = f"{approx:.2f}" if approx is not None else "N/A"
                f.write(f"{epoch:<8}{loss:<12.4f}{reward:<12.4f}{lf:<14.4f}{approx_str:<12}{epoch_time:<10.2f}\n")

            f.write("-" * 68 + "\n")
            f.write("\n")

            # Summary statistics
            f.write("SUMMARY STATISTICS:\n")
            f.write(f"  Mean Load Factor: {np.mean(self.training_history['train_load_factor']):.4f}\n")
            f.write(f"  Min Load Factor: {np.min(self.training_history['train_load_factor']):.4f}\n")
            f.write(f"  Max Load Factor: {np.max(self.training_history['train_load_factor']):.4f}\n")

            # Filter out None values for approximation ratio
            valid_approx = [x for x in self.training_history['train_approx_ratio'] if x is not None]
            if valid_approx:
                f.write(f"  Mean Approx Ratio: {np.mean(valid_approx):.2f}%\n")
                f.write(f"  Best Approx Ratio: {np.max(valid_approx):.2f}%\n")

        print(f"✓ Training log saved to: {self.log_file}")
