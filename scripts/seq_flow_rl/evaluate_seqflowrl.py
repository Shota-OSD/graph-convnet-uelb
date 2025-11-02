#!/usr/bin/env python3
"""
SeqFlowRL Evaluation Script

Evaluates trained SeqFlowRL model on test data.
"""

import sys
import os
import argparse
import torch
import json
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.seq_flow_rl.training import SeqFlowRLTrainer
from src.common.config.config_manager import ConfigManager
from src.common.data_management.dataset_reader import DatasetReader


def create_test_loader(config, dtypeFloat, dtypeLong):
    """Create test data loader."""
    dataset_reader = DatasetReader(config, dtypeFloat, dtypeLong)
    test_dataset = dataset_reader.read_test_data()

    batch_size = config.get('batch_size', 32)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=dataset_reader.collate_fn
    )

    return test_loader


def evaluate_model(trainer, test_loader):
    """
    Evaluate model on test set.

    Args:
        trainer: SeqFlowRLTrainer instance
        test_loader: Test data loader

    Returns:
        results: Dictionary of evaluation metrics
    """
    trainer.model.eval()

    all_metrics = {
        'mean_reward': [],
        'mean_load_factor': [],
        'min_load_factor': [],
        'max_load_factor': [],
        'mean_path_length': [],
    }

    print(f"\n🔍 Evaluating on {len(test_loader)} batches...")

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_loader):
            # Move data to device
            batch_data = {k: v.to(trainer.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch_data.items()}

            # Evaluation step (deterministic)
            metrics = trainer.strategy.eval_step(batch_data)

            # Accumulate metrics
            for key in all_metrics:
                if key in metrics:
                    all_metrics[key].append(metrics[key])

            # Progress
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{len(test_loader)} batches...")

    # Compute statistics
    results = {}
    for key, values in all_metrics.items():
        if values:
            results[f'{key}_mean'] = np.mean(values)
            results[f'{key}_std'] = np.std(values)
            results[f'{key}_min'] = np.min(values)
            results[f'{key}_max'] = np.max(values)

    return results


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='SeqFlowRL Evaluation')
    parser.add_argument('--config', type=str, default='configs/seqflowrl/seqflowrl_base.json',
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save evaluation results (default: checkpoint_dir/evaluation_results.json)')
    args = parser.parse_args()

    print("\n" + "="*70)
    print("SEQFLOWRL EVALUATION")
    print("="*70)
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")

    # Load configuration
    config_manager = ConfigManager(args.config)
    config = config_manager.get_config()
    dtypeFloat, dtypeLong = config_manager.get_dtypes()

    # Create trainer
    print(f"\n🔧 Initializing trainer...")
    trainer = SeqFlowRLTrainer(config, dtypeFloat, dtypeLong)

    # Load checkpoint
    print(f"\n📥 Loading checkpoint...")
    checkpoint = trainer.load_checkpoint(args.checkpoint)
    print(f"  - Loaded from epoch: {checkpoint['epoch'] + 1}")

    # Create test loader
    print(f"\n📊 Loading test dataset...")
    test_loader = create_test_loader(config, dtypeFloat, dtypeLong)
    print(f"  - Test samples: {len(test_loader.dataset)}")

    # Evaluate
    results = evaluate_model(trainer, test_loader)

    # Print results
    print("\n" + "="*70)
    print("📊 EVALUATION RESULTS")
    print("="*70)
    print(f"\n🎯 Load Factor:")
    print(f"  - Mean: {results.get('mean_load_factor_mean', 0):.4f} ± {results.get('mean_load_factor_std', 0):.4f}")
    print(f"  - Min:  {results.get('mean_load_factor_min', 0):.4f}")
    print(f"  - Max:  {results.get('mean_load_factor_max', 0):.4f}")

    print(f"\n🎁 Reward:")
    print(f"  - Mean: {results.get('mean_reward_mean', 0):.4f} ± {results.get('mean_reward_std', 0):.4f}")
    print(f"  - Min:  {results.get('mean_reward_min', 0):.4f}")
    print(f"  - Max:  {results.get('mean_reward_max', 0):.4f}")

    print(f"\n🛤️  Path Length:")
    print(f"  - Mean: {results.get('mean_path_length_mean', 0):.2f} ± {results.get('mean_path_length_std', 0):.2f}")

    # Save results
    if args.output:
        output_path = args.output
    else:
        output_path = trainer.checkpoint_dir / 'evaluation_results.json'

    print(f"\n💾 Saving results to: {output_path}")

    full_results = {
        'checkpoint': args.checkpoint,
        'config': args.config,
        'epoch': checkpoint['epoch'] + 1,
        'num_test_samples': len(test_loader.dataset),
        'metrics': results,
    }

    with open(output_path, 'w') as f:
        json.dump(full_results, f, indent=2)

    print("\n" + "="*70)
    print("✅ EVALUATION COMPLETED")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
