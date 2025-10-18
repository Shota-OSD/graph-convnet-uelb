#!/usr/bin/env python3
"""
RL-KSP Test Script
Tests a trained RL-KSP model on test dataset.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import argparse
import torch
from src.rl_ksp.train.rl_trainer import RLTrainer
from src.common.config.config_manager import ConfigManager


def main():
    print("=" * 60)
    print("RL-KSP (REINFORCEMENT LEARNING) TESTING")
    print("=" * 60)

    # Parse arguments
    parser = argparse.ArgumentParser(description='Test RL-KSP model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model file')
    parser.add_argument('--episodes', type=int, default=None,
                       help='Number of test episodes (default: from config)')
    args = parser.parse_args()

    # Load configuration
    config_manager = ConfigManager(args.config)
    config = dict(config_manager.config)

    print(f"\nConfiguration: {args.config}")
    print(f"Model: {args.model}")

    # Determine number of test episodes
    test_episodes = args.episodes if args.episodes else config.get('test_episodes', 100)
    print(f"Test episodes: {test_episodes}")

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Verify model file exists
    if not os.path.exists(args.model):
        print(f"\nError: Model file not found: {args.model}")
        sys.exit(1)

    # Update config to load the specified model
    config['load_saved_model'] = True
    config['saved_model_path'] = args.model

    # Initialize RL trainer
    print("\nInitializing RL trainer...")
    rl_trainer = RLTrainer(config)

    # Load model
    print(f"\nLoading model from {args.model}...")
    checkpoint = torch.load(args.model, map_location=device)

    if 'model_state_dict' in checkpoint:
        rl_trainer.policy_net.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded successfully!")

        if 'episode' in checkpoint:
            print(f"Model was trained for {checkpoint['episode']} episodes")
    else:
        print("\nError: Invalid checkpoint format")
        sys.exit(1)

    # Run test
    print("\n" + "=" * 60)
    print("RUNNING TEST EVALUATION")
    print("=" * 60)

    rl_trainer.test(test_episodes)

    print("\n" + "=" * 60)
    print("Testing complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
