#!/usr/bin/env python3
"""
GNN-ILS Training Script.

Usage:
    python scripts/gnn_ils/train_gnn_ils.py --config configs/gnn_ils/gnn_ils_base.json
"""

import sys
import os
import argparse
import json
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.common.config.config_manager import ConfigManager
from src.common.config.paths import dataset_exists
from src.common.data_management.dataset_reader import DatasetReader
from src.gnn_ils.training.trainer import GNNILSTrainer


def main():
    parser = argparse.ArgumentParser(description='Train GNN-ILS')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training')
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("GNN-ILS: GNN-GUIDED ITERATED LOCAL SEARCH")
    print("=" * 70)
    print(f"Config: {args.config}")

    config_manager = ConfigManager(args.config)
    config = config_manager.get_config()

    if not dataset_exists(config):
        print("\n⚠️  Data not found. Please generate data first:")
        print(f"    python scripts/common/generate_data.py --config {args.config}")
        sys.exit(1)

    dtypeFloat, dtypeLong = config_manager.get_dtypes()

    if args.epochs is not None:
        config['max_epochs'] = args.epochs
    if args.lr is not None:
        config['learning_rate'] = args.lr

    print(f"\nConfiguration:")
    print(f"  Epochs:           {config.get('max_epochs', 50)}")
    print(f"  Samples/epoch:    {config.get('samples_per_epoch', 100)}")
    print(f"  Learning rate:    {config.get('learning_rate', 0.0005)}")
    print(f"  Max ILS iters:    {config.get('max_iterations', 50)}")
    print(f"  Patience:         {config.get('no_improve_patience', 10)}")
    print(f"  K (KSP):          {config.get('K', 10)}")
    print(f"  Max paths/comm:   {config.get('max_candidate_paths', 15)}")

    num_train_data = config.get('num_train_data', 3200)
    num_val_data = config.get('num_val_data', 320)

    train_loader = DatasetReader(num_train_data, 1, 'train', config)
    val_loader = DatasetReader(num_val_data, 1, 'val', config)

    print(f"\nDataset:")
    print(f"  Train batches: {train_loader.max_iter}")
    print(f"  Val batches:   {val_loader.max_iter}")

    trainer = GNNILSTrainer(config, dtypeFloat, dtypeLong)

    if args.resume:
        print(f"\nResuming from: {args.resume}")
        trainer.load_checkpoint(args.resume)

    print(f"\nStarting training...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.get('max_epochs', 50),
    )

    history_path = trainer.checkpoint_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(
            {k: [x for x in v] for k, v in history.items()},
            f, indent=2, default=str
        )
    print(f"\nTraining history saved to: {history_path}")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)
    if history['train_load_factor']:
        print(f"  Final train LF:   {history['train_load_factor'][-1]:.4f}")
    if history['val_load_factor']:
        print(f"  Best val LF:      {trainer.best_val_load_factor:.4f}")
    print(f"  Checkpoints:      {trainer.checkpoint_dir}")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
