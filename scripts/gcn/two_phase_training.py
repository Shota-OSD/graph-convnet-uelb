#!/usr/bin/env python3
"""
Two-Phase Training Script: Supervised Pre-training + RL Fine-tuning

This script performs:
1. Phase 1: Supervised pre-training with voc_edges_out=2
2. Phase 2: RL fine-tuning with voc_edges_out=1 (converted from Phase 1)

Expected improvements:
- Complete Rate: 81-83% -> 90-95%
- Initial Reward: 0.7-1.7 -> 3.0-5.0
- Learning stability: Significantly improved
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)


def run_training_phase(config_path, phase_name, pretrained_model_path=None):
    """
    Run a single training phase.

    Args:
        config_path: Path to config file
        phase_name: Name of the phase (for display)
        pretrained_model_path: Optional path to pretrained model (for Phase 2)

    Returns:
        bool: True if successful, False otherwise
    """
    print("\n" + "="*80)
    print(f"PHASE: {phase_name}")
    print("="*80)
    print(f"Config: {config_path}\n")

    # If pretrained model path is provided, update the config temporarily
    import json
    import tempfile

    actual_config_path = config_path

    if pretrained_model_path:
        # Load config
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Update pretrained model path
        config['pretrained_model_path'] = pretrained_model_path

        # Save to temporary config file
        temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(config, temp_config, indent=4)
        temp_config.close()
        actual_config_path = temp_config.name

        print(f"Updated pretrained_model_path to: {pretrained_model_path}\n")

    # Run training script
    cmd = [
        sys.executable,
        os.path.join(project_root, 'scripts/gcn/train_gcn.py'),
        '--config', actual_config_path
    ]

    try:
        result = subprocess.run(cmd, check=True, cwd=project_root)
        print(f"\n✓ {phase_name} completed successfully!")

        # Clean up temp file if created
        if pretrained_model_path and os.path.exists(actual_config_path):
            os.unlink(actual_config_path)

        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {phase_name} failed with error code {e.returncode}")

        # Clean up temp file if created
        if pretrained_model_path and os.path.exists(actual_config_path):
            os.unlink(actual_config_path)

        return False


def check_pretrained_model_exists(model_path):
    """Check if pre-trained model exists."""
    if os.path.exists(model_path):
        print(f"✓ Pre-trained model found: {model_path}")
        return True
    else:
        print(f"✗ Pre-trained model not found: {model_path}")
        return False


def main():
    """Main function for two-phase training."""
    parser = argparse.ArgumentParser(description='Two-Phase Training: Supervised + RL')
    parser.add_argument('--supervised-config', type=str,
                       default='configs/gcn/supervised_pretraining.json',
                       help='Config file for supervised pre-training')
    parser.add_argument('--rl-config', type=str,
                       default='configs/gcn/rl_finetuning.json',
                       help='Config file for RL fine-tuning')
    parser.add_argument('--skip-supervised', action='store_true',
                       help='Skip supervised pre-training (use existing model)')
    parser.add_argument('--pretrained-model', type=str,
                       default='saved_models/supervised_pretrained.pt',
                       help='Path to save/load pre-trained model')
    args = parser.parse_args()

    print("\n" + "="*80)
    print("TWO-PHASE TRAINING: SUPERVISED PRE-TRAINING + RL FINE-TUNING")
    print("="*80)
    print("\nThis training approach:")
    print("  1. Pre-trains model with supervised learning (ground truth labels)")
    print("  2. Fine-tunes with reinforcement learning (load factor optimization)")
    print("\nExpected benefits:")
    print("  - Stable initialization (avoiding random exploration)")
    print("  - Higher initial performance (better starting point)")
    print("  - Faster convergence (less training time)")
    print("  - Improved final performance (better local optimum)")
    print("="*80 + "\n")

    # Phase 1: Supervised Pre-training
    if not args.skip_supervised:
        print("\n" + "🎓 " + "="*76)
        print("PHASE 1: SUPERVISED PRE-TRAINING")
        print("="*80)
        print("\nTraining model to imitate optimal solver solutions...")
        print(f"  - Model output: voc_edges_out=2 (binary classification)")
        print(f"  - Loss: Cross Entropy")
        print(f"  - Target: Learn 'good' path selection from optimal solutions\n")

        success = run_training_phase(args.supervised_config, "Supervised Pre-training")

        if not success:
            print("\n✗ Supervised pre-training failed. Aborting.")
            return 1

        # Automatically find the latest supervised model (just trained)
        saved_models_dir = os.path.join(project_root, 'saved_models')
        if os.path.exists(saved_models_dir):
            models = [f for f in os.listdir(saved_models_dir) if f.endswith('.pt')]
            if models:
                # Get the latest model by modification time (most recently saved)
                latest_model = max(models, key=lambda f: os.path.getmtime(
                    os.path.join(saved_models_dir, f)
                ))
                args.pretrained_model = os.path.join(saved_models_dir, latest_model)
                print(f"\n✓ Automatically using latest supervised model: {latest_model}")
            else:
                print("\n✗ No model files found in saved_models directory.")
                return 1
        else:
            print("\n✗ saved_models directory does not exist.")
            return 1

    else:
        print("\n⏭  Skipping supervised pre-training (using existing model)")
        # When skipping Phase 1, automatically find the latest model
        saved_models_dir = os.path.join(project_root, 'saved_models')
        if os.path.exists(saved_models_dir):
            models = [f for f in os.listdir(saved_models_dir) if f.endswith('.pt')]
            if models:
                latest_model = max(models, key=lambda f: os.path.getmtime(
                    os.path.join(saved_models_dir, f)
                ))
                args.pretrained_model = os.path.join(saved_models_dir, latest_model)
                print(f"✓ Automatically using latest model: {latest_model}")
            else:
                print("\n✗ No model files found in saved_models directory.")
                return 1
        else:
            print("\n✗ saved_models directory does not exist.")
            return 1

    # Phase 2: RL Fine-tuning
    print("\n" + "🚀 " + "="*76)
    print("PHASE 2: REINFORCEMENT LEARNING FINE-TUNING")
    print("="*80)
    print("\nFine-tuning pre-trained model with reinforcement learning...")
    print(f"  - Model output: voc_edges_out=1 (edge scores)")
    print(f"  - Initialization: Converted from supervised model (logit difference)")
    print(f"  - Reward: Load Factor optimization")
    print(f"  - Target: Optimize beyond supervised learning capabilities")
    print(f"  - Pre-trained model: {args.pretrained_model}\n")

    success = run_training_phase(args.rl_config, "RL Fine-tuning",
                                 pretrained_model_path=args.pretrained_model)

    if not success:
        print("\n✗ RL fine-tuning failed.")
        return 1

    # Success!
    print("\n" + "="*80)
    print("🎉 TWO-PHASE TRAINING COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nBoth phases completed:")
    print("  ✓ Phase 1: Supervised Pre-training")
    print("  ✓ Phase 2: RL Fine-tuning")
    print("\nCheck logs/ directory for detailed training results.")
    print("="*80 + "\n")

    return 0


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
