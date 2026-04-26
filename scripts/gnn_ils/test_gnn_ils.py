#!/usr/bin/env python3
"""
GNN-ILS Test Script.

Usage:
    python scripts/gnn_ils/test_gnn_ils.py --config configs/gnn_ils/gnn_ils_base.json
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.common.config.config_manager import ConfigManager
from src.common.data_management.dataset_reader import DatasetReader
from src.gnn_ils.training.trainer import GNNILSTrainer


def main():
    parser = argparse.ArgumentParser(description='Test GNN-ILS')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model checkpoint (default: saved_models/gnn_ils/best_model.pt)')
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("GNN-ILS: TEST")
    print("=" * 70)

    config_manager = ConfigManager(args.config)
    config = config_manager.get_config()
    dtypeFloat, dtypeLong = config_manager.get_dtypes()

    num_test_data = config.get('num_test_data', 320)
    test_loader = DatasetReader(num_test_data, 1, 'test', config)

    trainer = GNNILSTrainer(config, dtypeFloat, dtypeLong)

    model_path = args.model or config.get('load_model_path', None)
    if model_path is None:
        model_path = str(trainer.checkpoint_dir / 'best_model.pt')

    print(f"Loading model from: {model_path}")
    trainer.load_checkpoint(model_path)

    print(f"\nRunning evaluation on {num_test_data} test samples...")
    test_metrics = trainer._validate_epoch(test_loader, epoch=0)

    print(f"\n{'='*70}")
    print(f"TEST RESULTS")
    print(f"{'='*70}")
    if test_metrics is not None:
        print(f"  Load Factor:        {test_metrics.get('mean_load_factor', 0):.4f}")
        print(f"  Complete Rate:      {test_metrics.get('complete_rate', 100):.1f}%")
        print(f"  Mean Improvement:   {test_metrics.get('improvement', 0):.2f}%")
        print(f"  Mean Iterations:    {test_metrics.get('num_iterations', 0):.1f}")
        if test_metrics.get('approximation_ratio') is not None:
            print(f"  Approx Ratio:       {test_metrics['approximation_ratio']:.2f}%")
        else:
            print(f"  Approx Ratio:       N/A")
    else:
        print("  No test results.")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
