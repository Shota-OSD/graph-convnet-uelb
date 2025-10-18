#!/usr/bin/env python3
"""
GCN Test Script
Tests a trained GCN model on test dataset.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import argparse
import torch
from src.gcn.train.evaluator import Evaluator
from src.gcn.train.metrics import MetricsLogger
from src.common.config.config_manager import ConfigManager


def main():
    print("=" * 60)
    print("GCN (GRAPH CONVOLUTIONAL NETWORK) TESTING")
    print("=" * 60)

    # Parse arguments
    parser = argparse.ArgumentParser(description='Test GCN model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model file')
    parser.add_argument('--epoch', type=int, default=None,
                       help='Specific epoch to test (optional)')
    args = parser.parse_args()

    # Load configuration
    config_manager = ConfigManager(args.config)
    config = dict(config_manager.config)

    print(f"\nConfiguration: {args.config}")
    print(f"Model: {args.model}")
    if args.epoch:
        print(f"Testing epoch: {args.epoch}")

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Initialize evaluator and metrics logger
    evaluator = Evaluator(config)
    metrics_logger = MetricsLogger(config)

    # Load model
    if not os.path.exists(args.model):
        print(f"\nError: Model file not found: {args.model}")
        sys.exit(1)

    print(f"\nLoading model from {args.model}...")
    checkpoint = torch.load(args.model, map_location=device)

    # If testing specific epoch, verify it exists
    if args.epoch is not None:
        epoch_key = f'epoch_{args.epoch}'
        if epoch_key not in checkpoint:
            print(f"\nError: Epoch {args.epoch} not found in checkpoint")
            print(f"Available epochs: {[k for k in checkpoint.keys() if k.startswith('epoch_')]}")
            sys.exit(1)
        model_state = checkpoint[epoch_key]
        print(f"Testing model at epoch {args.epoch}")
    else:
        # Use latest epoch
        if 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
            print("Testing latest model")
        else:
            epoch_keys = [k for k in checkpoint.keys() if k.startswith('epoch_')]
            if not epoch_keys:
                print("\nError: No model state found in checkpoint")
                sys.exit(1)
            latest_epoch = max([int(k.split('_')[1]) for k in epoch_keys])
            model_state = checkpoint[f'epoch_{latest_epoch}']
            print(f"Testing latest model (epoch {latest_epoch})")

    # Import and create model
    from src.gcn.models.gcn_model import ResidualGatedGCNModel

    model = ResidualGatedGCNModel(config, device).to(device)
    model.load_state_dict(model_state)
    model.eval()

    print("\nModel loaded successfully!")

    # Run evaluation on test set
    print("\n" + "=" * 60)
    print("RUNNING TEST EVALUATION")
    print("=" * 60)

    with torch.no_grad():
        test_loss, test_pred_tour_len, test_gt_tour_len, test_max_load_factor = evaluator.evaluate_model(
            model,
            split='test'
        )

    # Log results
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"Test Loss: {test_loss:.6f}")
    print(f"Test Predicted Tour Length: {test_pred_tour_len:.6f}")
    print(f"Test Ground Truth Tour Length: {test_gt_tour_len:.6f}")
    print(f"Test Max Load Factor: {test_max_load_factor:.6f}")
    print("=" * 60)

    # Save test results to file
    results_dir = os.path.join(config.get('expt_dir', 'results'),
                               config.get('expt_name', 'experiment'))
    os.makedirs(results_dir, exist_ok=True)

    results_file = os.path.join(results_dir, 'test_results.txt')
    with open(results_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("GCN TEST RESULTS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Config: {args.config}\n")
        if args.epoch:
            f.write(f"Epoch: {args.epoch}\n")
        f.write("\n")
        f.write(f"Test Loss: {test_loss:.6f}\n")
        f.write(f"Test Predicted Tour Length: {test_pred_tour_len:.6f}\n")
        f.write(f"Test Ground Truth Tour Length: {test_gt_tour_len:.6f}\n")
        f.write(f"Test Max Load Factor: {test_max_load_factor:.6f}\n")
        f.write("=" * 60 + "\n")

    print(f"\nResults saved to: {results_file}")
    print("\nTesting complete!")


if __name__ == '__main__':
    main()
