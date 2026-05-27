#!/usr/bin/env python3
"""
GNN-ILS Batch Runner.

複数の config に対して「データ生成 → 学習 → テスト」を一括実行し、
結果を summary.json / summary.csv に保存する。

Usage:
    python scripts/gnn_ils/batch_run.py configs/gnn_ils/case_a.json configs/gnn_ils/case_b.json
    python scripts/gnn_ils/batch_run.py configs/gnn_ils/experiment_*.json --results-dir batch_results
    python scripts/gnn_ils/batch_run.py configs/gnn_ils/*.json --skip-data-gen --epochs 30
"""

import sys
import os
import argparse
import json
import csv
import traceback
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.common.config.config_manager import ConfigManager
from src.common.config.paths import dataset_exists
from src.common.data_management.dataset_reader import DatasetReader
from src.gnn_ils.training.trainer import GNNILSTrainer
from scripts.common.generate_data import quick_generate


def run_single_case(config_path: str, results_dir: Path, args) -> dict:
    """単一 config のデータ生成→学習→テストを実行し、結果辞書を返す。"""
    result = {
        'config_path': config_path,
        'status': 'error',
    }

    config_manager = ConfigManager(config_path)
    config = config_manager.get_config()
    dtypeFloat, dtypeLong = config_manager.get_dtypes()

    expt_name = config.get('expt_name', Path(config_path).stem)
    result['expt_name'] = expt_name

    # --- データ生成 ---
    if not args.skip_data_gen:
        if not dataset_exists(config, check_count=True):
            print(f"\n  Generating data for {expt_name}...")
            quick_generate(config_path)
        else:
            print(f"  Data already exists for {expt_name}, skipping generation.")
    else:
        print(f"  --skip-data-gen: skipping data generation for {expt_name}")

    # checkpoint_dir を config 毎に分離
    config['checkpoint_dir'] = str(results_dir / expt_name / 'checkpoints')

    # エポック数上書き
    if args.epochs is not None:
        config['max_epochs'] = args.epochs

    # --- 学習 ---
    trainer = GNNILSTrainer(config, dtypeFloat, dtypeLong)
    train_metrics = {}

    if not args.skip_train:
        num_train_data = config.get('num_train_data', 3200)
        num_val_data = config.get('num_val_data', 320)
        train_loader = DatasetReader(num_train_data, 1, 'train', config)
        val_loader = DatasetReader(num_val_data, 1, 'val', config)

        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config.get('max_epochs', 50),
        )

        train_metrics = {
            'best_val_load_factor': trainer.best_val_load_factor,
            'final_train_loss': history['train_loss'][-1] if history['train_loss'] else None,
        }
    else:
        print(f"  --skip-train: loading existing checkpoint for {expt_name}")
        model_path = Path(config['checkpoint_dir']) / 'best_model.pt'
        if not model_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {model_path}. "
                f"Cannot skip training without an existing checkpoint."
            )
        trainer.load_checkpoint(str(model_path))

    # --- テスト ---
    num_test_data = config.get('num_test_data', 320)
    test_loader = DatasetReader(num_test_data, 1, 'test', config)

    print(f"\n  Running test evaluation for {expt_name} ({num_test_data} samples)...")
    test_metrics_raw = trainer._validate_epoch(test_loader, epoch=0)

    test_metrics = {}
    if test_metrics_raw is not None:
        test_metrics = {
            'mean_load_factor': test_metrics_raw.get('mean_load_factor'),
            'complete_rate': test_metrics_raw.get('complete_rate'),
            'complete_sample_rate': test_metrics_raw.get('complete_sample_rate'),
            'improvement': test_metrics_raw.get('improvement'),
            'num_iterations': test_metrics_raw.get('num_iterations'),
            'approximation_ratio': test_metrics_raw.get('approximation_ratio'),
            'mean_time_per_sample': test_metrics_raw.get('mean_time_per_sample'),
            'std_time_per_sample': test_metrics_raw.get('std_time_per_sample'),
            'total_time': test_metrics_raw.get('total_time'),
        }

    result['status'] = 'success'
    result['train_metrics'] = train_metrics
    result['test_metrics'] = test_metrics
    return result


def save_summary(summary: dict, results_dir: Path):
    """summary.json と summary.csv を保存する。"""
    results_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = results_dir / 'summary.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # CSV
    csv_path = results_dir / 'summary.csv'
    fieldnames = [
        'config_path', 'expt_name', 'status',
        'best_val_load_factor', 'final_train_loss',
        'mean_load_factor', 'complete_rate', 'complete_sample_rate',
        'improvement', 'num_iterations', 'approximation_ratio',
        'mean_time_per_sample', 'std_time_per_sample', 'total_time',
        'error',
    ]
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for case in summary['cases']:
            row = {
                'config_path': case.get('config_path', ''),
                'expt_name': case.get('expt_name', ''),
                'status': case.get('status', ''),
                'error': case.get('error', ''),
            }
            for k in ('best_val_load_factor', 'final_train_loss'):
                row[k] = case.get('train_metrics', {}).get(k, '')
            for k in ('mean_load_factor', 'complete_rate', 'complete_sample_rate',
                       'improvement', 'num_iterations', 'approximation_ratio',
                       'mean_time_per_sample', 'std_time_per_sample', 'total_time'):
                row[k] = case.get('test_metrics', {}).get(k, '')
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description='GNN-ILS Batch Runner: run data-gen, train, test for multiple configs')
    parser.add_argument('configs', nargs='+', help='Config file paths')
    parser.add_argument('--results-dir', type=str, default='batch_results',
                        help='Directory to save results (default: batch_results)')
    parser.add_argument('--skip-data-gen', action='store_true',
                        help='Skip data generation (assume data exists)')
    parser.add_argument('--skip-train', action='store_true',
                        help='Skip training (test with existing checkpoints)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override epoch count for all configs')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("GNN-ILS BATCH RUNNER")
    print("=" * 70)
    print(f"  Configs:      {len(args.configs)}")
    print(f"  Results dir:  {results_dir}")
    print(f"  Skip data:    {args.skip_data_gen}")
    print(f"  Skip train:   {args.skip_train}")
    if args.epochs is not None:
        print(f"  Epochs:       {args.epochs}")
    print("=" * 70)

    summary = {
        'timestamp': datetime.now().isoformat(),
        'cases': [],
    }

    interrupted = False
    for i, config_path in enumerate(args.configs):
        print(f"\n{'='*70}")
        print(f"[{i + 1}/{len(args.configs)}] {config_path}")
        print(f"{'='*70}")

        try:
            result = run_single_case(config_path, results_dir, args)
            summary['cases'].append(result)
        except KeyboardInterrupt:
            print(f"\n  Interrupted by user. Saving partial results...")
            summary['cases'].append({
                'config_path': config_path,
                'status': 'interrupted',
            })
            interrupted = True
        except Exception as e:
            print(f"\n  ERROR: {e}")
            traceback.print_exc()
            summary['cases'].append({
                'config_path': config_path,
                'status': 'error',
                'error': str(e),
            })

        # 中間保存
        save_summary(summary, results_dir)

        if interrupted:
            break

    # 最終サマリー表示
    print(f"\n{'='*70}")
    print("BATCH RUN COMPLETE")
    print(f"{'='*70}")

    success = sum(1 for c in summary['cases'] if c['status'] == 'success')
    failed = len(summary['cases']) - success
    print(f"  Success: {success}/{len(summary['cases'])}")
    if failed > 0:
        print(f"  Failed:  {failed}")

    for case in summary['cases']:
        status_mark = "OK" if case['status'] == 'success' else "NG"
        name = case.get('expt_name', case.get('config_path', '?'))
        lf = case.get('test_metrics', {}).get('mean_load_factor')
        lf_str = f"LF={lf:.4f}" if lf is not None else ""
        approx = case.get('test_metrics', {}).get('approximation_ratio')
        approx_str = f"Approx={approx:.2f}%" if approx is not None else ""
        print(f"  [{status_mark}] {name}  {lf_str}  {approx_str}")

    print(f"\n  Results: {results_dir / 'summary.json'}")
    print(f"           {results_dir / 'summary.csv'}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
