import os
import json
import csv
import itertools
import random
import copy
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np
import torch

from src.common.config.config_manager import ConfigManager
from src.common.data_management.dataset_reader import DatasetReader
from src.gnn_ils.training.trainer import GNNILSTrainer


class GNNILSHyperparameterTuner:
    """GNN-ILS 用ハイパーパラメータチューニングクラス"""

    def __init__(self, base_config_path: str, tuning_config_path: str, results_dir: str = "tuning_results_gnn_ils"):
        self.base_config_path = base_config_path
        self.tuning_config_path = tuning_config_path
        self.results_dir = results_dir

        os.makedirs(self.results_dir, exist_ok=True)

        with open(tuning_config_path, 'r') as f:
            self.tuning_config = json.load(f)

        self.results: List[Dict] = []
        self.best_result: Optional[Dict] = None
        self.current_trial = 0
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def _create_config_with_params(self, params: Dict[str, Any]) -> str:
        """パラメータを適用した一時設定ファイルを作成"""
        config_manager = ConfigManager(self.base_config_path)
        base_config = config_manager.get_config()

        config_dict = dict(base_config)
        config_dict.update(params)

        temp_config_path = os.path.join(
            self.results_dir,
            f"temp_config_trial_{self.current_trial}_{self.timestamp}.json"
        )

        with open(temp_config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

        return temp_config_path

    def _evaluate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """指定パラメータでモデルを訓練・評価"""
        print(f"\n{'='*60}")
        print(f"Trial {self.current_trial + 1}: Evaluating parameters:")
        for k, v in params.items():
            print(f"  {k}: {v}")
        print(f"{'='*60}")

        temp_config_path = self._create_config_with_params(params)

        try:
            config_manager = ConfigManager(temp_config_path)
            config = config_manager.get_config()
            dtypeFloat, dtypeLong = config_manager.get_dtypes()

            num_train_data = config.get('num_train_data', 3200)
            num_val_data = config.get('num_val_data', 320)

            train_loader = DatasetReader(num_train_data, 1, 'train', config)
            val_loader = DatasetReader(num_val_data, 1, 'val', config)

            trainer = GNNILSTrainer(config, dtypeFloat, dtypeLong)

            history = trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=config.get('max_epochs', 50),
            )

            # 結果収集
            result = {
                'trial': self.current_trial,
                'params': params.copy(),
                'best_val_load_factor': trainer.best_val_load_factor,
                'timestamp': datetime.now().isoformat(),
            }

            # 最終 train メトリクス
            if history['train_load_factor']:
                result['final_train_load_factor'] = history['train_load_factor'][-1]
                result['final_train_loss'] = history['train_loss'][-1]
                result['final_train_improvement'] = history['train_improvement'][-1]
                result['final_train_num_iterations'] = history['train_num_iterations'][-1]

            # train approx ratio
            valid_train_approx = [x for x in history['train_approx_ratio'] if x is not None]
            result['best_train_approx_ratio'] = max(valid_train_approx) if valid_train_approx else None
            result['final_train_approx_ratio'] = valid_train_approx[-1] if valid_train_approx else None

            # val メトリクス
            if history['val_load_factor']:
                result['final_val_load_factor'] = history['val_load_factor'][-1]
                result['final_val_improvement'] = history['val_improvement'][-1]
                result['final_val_num_iterations'] = history['val_num_iterations'][-1]

            valid_val_approx = [x for x in history['val_approx_ratio'] if x is not None]
            result['best_val_approx_ratio'] = max(valid_val_approx) if valid_val_approx else None
            result['final_val_approx_ratio'] = valid_val_approx[-1] if valid_val_approx else None

            # 時間
            if history['epoch_times']:
                result['total_time'] = sum(history['epoch_times'])
                result['avg_epoch_time'] = np.mean(history['epoch_times'])

            print(f"\nTrial {self.current_trial + 1} Results:")
            print(f"  Best Val Load Factor: {result['best_val_load_factor']:.4f}")
            if result.get('best_val_approx_ratio') is not None:
                print(f"  Best Val Approx Ratio: {result['best_val_approx_ratio']:.2f}%")

            return result

        except Exception as e:
            print(f"Trial {self.current_trial + 1} failed with error: {e}")
            import traceback
            traceback.print_exc()
            return {
                'trial': self.current_trial,
                'params': params.copy(),
                'best_val_load_factor': float('inf'),
                'best_val_approx_ratio': None,
                'final_train_load_factor': float('inf'),
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
            }
        finally:
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)

    def grid_search(self) -> Optional[Dict[str, Any]]:
        """グリッドサーチを実行"""
        print("Starting Grid Search (GNN-ILS)...")

        param_grid = self.tuning_config['grid_search']['parameters']
        param_names = list(param_grid.keys())
        param_values = [param_grid[name] for name in param_names]

        all_combinations = list(itertools.product(*param_values))
        total_trials = len(all_combinations)
        print(f"Total trials: {total_trials}")

        self.results = []

        for i, combination in enumerate(all_combinations):
            self.current_trial = i
            params = dict(zip(param_names, combination))

            result = self._evaluate_params(params)
            self.results.append(result)

            if self._is_better_result(result):
                self.best_result = result
                print(f"  >> New best! Val LF: {result['best_val_load_factor']:.4f}")

            self._save_intermediate_results()

        self._save_final_results("grid_search")
        return self.best_result

    def random_search(self, n_trials: int = None) -> Optional[Dict[str, Any]]:
        """ランダムサーチを実行"""
        print("Starting Random Search (GNN-ILS)...")

        if n_trials is None:
            n_trials = self.tuning_config['random_search']['n_trials']

        param_ranges = self.tuning_config['random_search']['parameters']
        print(f"Total trials: {n_trials}")

        self.results = []

        for i in range(n_trials):
            self.current_trial = i
            params = self._generate_random_params(param_ranges)

            result = self._evaluate_params(params)
            self.results.append(result)

            if self._is_better_result(result):
                self.best_result = result
                print(f"  >> New best! Val LF: {result['best_val_load_factor']:.4f}")

            self._save_intermediate_results()

        self._save_final_results("random_search")
        return self.best_result

    def _generate_random_params(self, param_ranges: Dict[str, Dict]) -> Dict[str, Any]:
        """ランダムパラメータを生成"""
        params = {}

        for param_name, config in param_ranges.items():
            param_type = config['type']

            if param_type == 'int':
                params[param_name] = random.randint(config['min'], config['max'])
            elif param_type == 'float':
                if config.get('log_scale', False):
                    log_min = np.log10(config['min'])
                    log_max = np.log10(config['max'])
                    params[param_name] = float(10 ** random.uniform(log_min, log_max))
                else:
                    params[param_name] = round(random.uniform(config['min'], config['max']), 6)
            elif param_type == 'choice':
                params[param_name] = random.choice(config['choices'])
            elif param_type == 'bool':
                params[param_name] = random.choice([True, False])

        return params

    def _is_better_result(self, result: Dict[str, Any]) -> bool:
        """結果がベストかどうかを判定"""
        if self.best_result is None:
            return True

        evaluation_config = self.tuning_config.get('evaluation', {})
        primary_metric = evaluation_config.get('primary_metric', 'best_val_load_factor')
        direction = evaluation_config.get('optimization_direction', 'minimize')

        current_value = result.get(primary_metric)
        best_value = self.best_result.get(primary_metric)

        if current_value is None:
            return False
        if best_value is None:
            return True

        if direction == 'minimize':
            return current_value < best_value
        else:
            return current_value > best_value

    def _save_intermediate_results(self):
        """中間結果を CSV で保存"""
        csv_path = os.path.join(self.results_dir, f"intermediate_results_{self.timestamp}.csv")

        if not self.results:
            return

        # フィールド名構築
        fieldnames = ['trial', 'timestamp']
        if 'params' in self.results[0]:
            fieldnames.extend(list(self.results[0]['params'].keys()))

        metric_names = [
            'best_val_load_factor', 'final_val_load_factor',
            'final_train_load_factor', 'final_train_loss',
            'best_val_approx_ratio', 'final_val_approx_ratio',
            'best_train_approx_ratio', 'final_train_approx_ratio',
            'final_train_improvement', 'final_val_improvement',
            'final_train_num_iterations', 'final_val_num_iterations',
            'total_time', 'avg_epoch_time',
        ]
        fieldnames.extend(metric_names)
        if any('error' in r for r in self.results):
            fieldnames.append('error')

        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for result in self.results:
                row = {'trial': result['trial'], 'timestamp': result['timestamp']}
                if 'params' in result:
                    row.update(result['params'])
                for metric in metric_names:
                    val = result.get(metric, '')
                    row[metric] = f"{val:.6f}" if isinstance(val, float) and val != float('inf') else val
                if 'error' in result:
                    row['error'] = result['error']
                writer.writerow(row)

    def _save_final_results(self, search_type: str):
        """最終結果を JSON で保存"""
        json_path = os.path.join(self.results_dir, f"{search_type}_results_{self.timestamp}.json")

        def sanitize(obj):
            if isinstance(obj, float) and (np.isinf(obj) or np.isnan(obj)):
                return str(obj)
            if hasattr(obj, 'item'):
                return obj.item()
            if isinstance(obj, dict):
                return {k: sanitize(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [sanitize(x) for x in obj]
            return obj

        final_results = {
            'search_type': search_type,
            'timestamp': self.timestamp,
            'total_trials': len(self.results),
            'best_result': sanitize(self.best_result),
            'all_results': sanitize(self.results),
            'tuning_config': self.tuning_config,
        }

        with open(json_path, 'w') as f:
            json.dump(final_results, f, indent=2)

        self._generate_summary_report(search_type)

        print(f"\nTuning completed!")
        print(f"Results saved to: {json_path}")
        if self.best_result:
            print(f"Best val load factor: {self.best_result['best_val_load_factor']:.4f}")

    def _generate_summary_report(self, search_type: str):
        """サマリーレポートを生成"""
        report_path = os.path.join(self.results_dir, f"{search_type}_summary_{self.timestamp}.txt")

        with open(report_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write(f"GNN-ILS HYPERPARAMETER TUNING SUMMARY - {search_type.upper()}\n")
            f.write("=" * 60 + "\n")
            f.write(f"Timestamp: {self.timestamp}\n")
            f.write(f"Total Trials: {len(self.results)}\n\n")

            if self.best_result:
                f.write("BEST RESULT:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Trial: {self.best_result['trial']}\n")
                f.write(f"Best Val Load Factor: {self.best_result['best_val_load_factor']:.4f}\n")

                if self.best_result.get('best_val_approx_ratio') is not None:
                    f.write(f"Best Val Approx Ratio: {self.best_result['best_val_approx_ratio']:.2f}%\n")
                if self.best_result.get('final_train_load_factor') is not None:
                    f.write(f"Final Train Load Factor: {self.best_result['final_train_load_factor']:.4f}\n")
                if self.best_result.get('total_time') is not None:
                    f.write(f"Total Time: {self.best_result['total_time']:.2f}s\n")

                f.write("\nBest Parameters:\n")
                for param, value in self.best_result['params'].items():
                    f.write(f"  {param}: {value}\n")

                f.write("\n" + "=" * 60 + "\n")

            # 統計
            successful = [r for r in self.results if 'error' not in r]
            if len(successful) > 1:
                val_lfs = [r['best_val_load_factor'] for r in successful
                           if r['best_val_load_factor'] != float('inf')]
                if val_lfs:
                    f.write("STATISTICS (Val Load Factor):\n")
                    f.write("-" * 30 + "\n")
                    f.write(f"Mean: {np.mean(val_lfs):.4f}\n")
                    f.write(f"Std:  {np.std(val_lfs):.4f}\n")
                    f.write(f"Min:  {np.min(val_lfs):.4f}\n")
                    f.write(f"Max:  {np.max(val_lfs):.4f}\n")

                val_approx = [r['best_val_approx_ratio'] for r in successful
                              if r.get('best_val_approx_ratio') is not None]
                if val_approx:
                    f.write("\nSTATISTICS (Val Approx Ratio):\n")
                    f.write("-" * 30 + "\n")
                    f.write(f"Mean: {np.mean(val_approx):.2f}%\n")
                    f.write(f"Std:  {np.std(val_approx):.2f}%\n")
                    f.write(f"Min:  {np.min(val_approx):.2f}%\n")
                    f.write(f"Max:  {np.max(val_approx):.2f}%\n")

            failed = [r for r in self.results if 'error' in r]
            if failed:
                f.write(f"\nFailed Trials: {len(failed)}\n")
                for r in failed:
                    f.write(f"  Trial {r['trial']}: {r['error']}\n")

        print(f"Summary report saved to: {report_path}")
