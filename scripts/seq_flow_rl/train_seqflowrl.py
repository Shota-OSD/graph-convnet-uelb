#!/usr/bin/env python3
"""
SeqFlowRL Training Script

Trains SeqFlowRL model using A2C algorithm with sequential rollout and per-commodity GNN updates.
Supports 2-phase training (supervised pre-training + RL fine-tuning).
"""

import sys
import os
import argparse
import torch

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.seq_flow_rl.training import SeqFlowRLTrainer
from src.common.config.config_manager import ConfigManager
from src.common.data_management.dataset_reader import DatasetReader


def _check_data_exists():
    """Check if data directories exist."""
    data_dirs = ['./data/train_data', './data/val_data', './data/test_data']
    required_subdirs = ['commodity_file', 'graph_file', 'node_flow_file']

    for data_dir in data_dirs:
        if not os.path.exists(data_dir):
            return False

        for subdir in required_subdirs:
            subdir_path = os.path.join(data_dir, subdir)
            if not os.path.exists(subdir_path):
                return False

        commodity_dir = os.path.join(data_dir, 'commodity_file')
        subdirs = [d for d in os.listdir(commodity_dir)
                  if os.path.isdir(os.path.join(commodity_dir, d)) and d.isdigit()]
        if len(subdirs) == 0:
            return False

    return True


def create_dataloaders(config, dtypeFloat, dtypeLong):
    """
    Create training and validation data iterators.

    Args:
        config: Configuration dictionary
        dtypeFloat: Float data type (not used, kept for compatibility)
        dtypeLong: Long data type (not used, kept for compatibility)

    Returns:
        train_dataset, val_dataset (DatasetReader instances)
    """
    # Get configuration parameters
    batch_size = config.get('batch_size', 32)
    num_train_data = config.get('num_train_data', 10000)
    num_val_data = config.get('num_val_data', 1000)

    # Create dataset readers (they are iterators)
    train_dataset = DatasetReader(num_train_data, batch_size, 'train')
    val_dataset = DatasetReader(num_val_data, batch_size, 'val')

    return train_dataset, val_dataset


def main():
    """Main training function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='SeqFlowRL Training')
    parser.add_argument('--config', type=str, default='configs/seqflowrl/seqflowrl_base.json',
                       help='Path to config file (default: configs/seqflowrl/seqflowrl_base.json)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs (overrides config)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate (overrides config)')
    parser.add_argument('--pretrained', type=str, default=None,
                       help='Path to pretrained supervised model (enables 2-phase training)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training')
    args = parser.parse_args()

    print("\n" + "="*70)
    print("SEQFLOWRL: SEQUENTIAL FLOW REINFORCEMENT LEARNING")
    print("="*70)
    print(f"Config: {args.config}")

    # Check if data exists
    if not _check_data_exists():
        print("\n⚠️  Data not found. Please generate data first:")
        print(f"    python scripts/common/generate_data.py --config {args.config}")
        sys.exit(1)

    # Load configuration
    config_manager = ConfigManager(args.config)
    config = config_manager.get_config()
    dtypeFloat, dtypeLong = config_manager.get_dtypes()

    # Override config with command line arguments
    if args.epochs is not None:
        config['max_epochs'] = args.epochs
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.lr is not None:
        config['learning_rate'] = args.lr
    if args.pretrained is not None:
        config['load_pretrained_model'] = True
        config['pretrained_model_path'] = args.pretrained

    # Print configuration
    print(f"\n📋 Configuration:")
    print(f"  - Epochs: {config.get('max_epochs', 50)}")
    print(f"  - Batch size: {config.get('batch_size', 32)}")
    print(f"  - Learning rate: {config.get('learning_rate', 0.0005)}")
    print(f"  - GNN update: {config.get('gnn_update_frequency', 'per_commodity')}")
    print(f"  - Action type: {config.get('action_type', 'node')}")
    print(f"  - RL algorithm: {config.get('rl_algorithm', 'a2c')}")

    if config.get('load_pretrained_model', False):
        print(f"  - 2-phase training: ENABLED")
        print(f"    Pretrained model: {config.get('pretrained_model_path', 'N/A')}")
    else:
        print(f"  - 2-phase training: DISABLED")

    # Create trainer
    print(f"\n🔧 Initializing trainer...")
    trainer = SeqFlowRLTrainer(config, dtypeFloat, dtypeLong)

    # Resume from checkpoint if specified
    if args.resume:
        print(f"\n📥 Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Create data iterators
    print(f"\n📊 Loading datasets...")
    train_dataset, val_dataset = create_dataloaders(config, dtypeFloat, dtypeLong)
    print(f"  - Training batches: {train_dataset.max_iter}")
    print(f"  - Validation batches: {val_dataset.max_iter}")

    # Train
    print(f"\n🚀 Starting training...")
    training_history = trainer.train(
        train_loader=train_dataset,
        val_loader=val_dataset,
        num_epochs=config.get('max_epochs', 50)
    )

    # Save final results
    print(f"\n💾 Saving training history...")
    history_path = trainer.checkpoint_dir / 'training_history.json'
    import json
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    print(f"  Saved to: {history_path}")

    print("\n" + "="*70)
    print("✅ TRAINING COMPLETED SUCCESSFULLY")
    print("="*70)
    print(f"  Final training load factor: {training_history['train_load_factor'][-1]:.4f}")
    if training_history['val_load_factor']:
        print(f"  Final validation load factor: {training_history['val_load_factor'][-1]:.4f}")
    print(f"  Best validation load factor: {trainer.best_val_load_factor:.4f}")
    print(f"  Checkpoints saved to: {trainer.checkpoint_dir}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
