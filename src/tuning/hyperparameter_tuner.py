import os
import json
import csv
import itertools
import random
import copy
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
import numpy as np

from ..config.config_manager import ConfigManager
from ..train.trainer import Trainer
from ..train.evaluator import Evaluator
from ..train.metrics import MetricsLogger


class HyperparameterTuner:
    """ハイパーパラメータチューニングを担当するクラス"""

    def __init__(self, base_config_path: str, tuning_config_path: str, results_dir: str = "tuning_results"):
        """
        Args:
            base_config_path: ベース設定ファイルのパス
            tuning_config_path: チューニング設定ファイルのパス
            results_dir: 結果保存ディレクトリ
        """
        self.base_config_path = base_config_path
        self.tuning_config_path = tuning_config_path
        self.results_dir = results_dir

        # 結果保存ディレクトリの作成
        os.makedirs(self.results_dir, exist_ok=True)

        # チューニング設定の読み込み
        with open(tuning_config_path, 'r') as f:
            self.tuning_config = json.load(f)

        # 結果記録用
        self.results = []
        self.best_result = None
        self.current_trial = 0

        # タイムスタンプ
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def _create_config_with_params(self, params: Dict[str, Any]) -> str:
        """パラメータを適用した設定ファイルを作成"""
        # ベース設定を読み込み
        config_manager = ConfigManager(self.base_config_path)
        base_config = config_manager.get_config()

        # パラメータを適用
        config_dict = dict(base_config)
        config_dict.update(params)

        # 一時設定ファイルを作成
        temp_config_path = os.path.join(
            self.results_dir,
            f"temp_config_trial_{self.current_trial}_{self.timestamp}.json"
        )

        with open(temp_config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

        return temp_config_path

    def _evaluate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """指定されたパラメータでモデルを訓練・評価"""
        print(f"\nTrial {self.current_trial + 1}: Evaluating parameters: {params}")

        # 設定ファイルを作成
        temp_config_path = self._create_config_with_params(params)

        try:
            # 設定管理の初期化
            config_manager = ConfigManager(temp_config_path)
            config = config_manager.get_config()
            dtypeFloat, dtypeLong = config_manager.get_dtypes()

            # トレーナーと評価器の初期化
            trainer = Trainer(config, dtypeFloat, dtypeLong)
            evaluator = Evaluator(config, dtypeFloat, dtypeLong)
            metrics_logger = MetricsLogger(self.results_dir)

            # 訓練実行（詳細出力を無効にして高速化）
            best_net = trainer.train(evaluator, metrics_logger, verbose=False)

            # 最終評価
            from fastprogress import master_bar
            master_bar_obj = master_bar(range(1))

            # テスト評価
            test_time, test_loss, test_mean_load_factor, test_gt_load_factor, test_approximation_rate, test_infeasible_rate = evaluator.evaluate(
                best_net, master_bar_obj, mode='test'
            )

            # 検証評価
            val_time, val_loss, val_mean_load_factor, val_gt_load_factor, val_approximation_rate, val_infeasible_rate = evaluator.evaluate(
                best_net, master_bar_obj, mode='val'
            )

            # 結果をまとめる
            result = {
                'trial': self.current_trial,
                'params': params.copy(),
                'test_approximation_rate': test_approximation_rate,
                'val_approximation_rate': val_approximation_rate,
                'test_infeasible_rate': test_infeasible_rate,
                'val_infeasible_rate': val_infeasible_rate,
                'test_mean_load_factor': test_mean_load_factor,
                'val_mean_load_factor': val_mean_load_factor,
                'test_loss': test_loss,
                'val_loss': val_loss,
                'test_time': test_time,
                'val_time': val_time,
                'timestamp': datetime.now().isoformat()
            }

            print(f"Trial {self.current_trial + 1} Results:")
            print(f"  Test Approximation Rate: {test_approximation_rate:.2f}%")
            print(f"  Val Approximation Rate: {val_approximation_rate:.2f}%")
            print(f"  Test Infeasible Rate: {test_infeasible_rate:.2f}%")

            return result

        except Exception as e:
            print(f"Trial {self.current_trial + 1} failed with error: {e}")
            # エラーの場合は最低スコアを返す
            return {
                'trial': self.current_trial,
                'params': params.copy(),
                'test_approximation_rate': 0.0,
                'val_approximation_rate': 0.0,
                'test_infeasible_rate': 100.0,
                'val_infeasible_rate': 100.0,
                'test_mean_load_factor': 0.0,
                'val_mean_load_factor': 0.0,
                'test_loss': float('inf'),
                'val_loss': float('inf'),
                'test_time': 0.0,
                'val_time': 0.0,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
        finally:
            # 一時ファイルを削除
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)

    def grid_search(self) -> Dict[str, Any]:
        """グリッドサーチを実行"""
        print("Starting Grid Search...")

        # パラメータグリッドの生成
        param_grid = self.tuning_config['grid_search']['parameters']
        param_names = list(param_grid.keys())
        param_values = [param_grid[name] for name in param_names]

        # 全組み合わせを生成
        all_combinations = list(itertools.product(*param_values))
        total_trials = len(all_combinations)

        print(f"Total trials: {total_trials}")

        self.results = []

        for i, combination in enumerate(all_combinations):
            self.current_trial = i

            # パラメータ辞書を作成
            params = dict(zip(param_names, combination))

            # 評価実行
            result = self._evaluate_params(params)
            self.results.append(result)

            # ベスト結果の更新
            if self._is_better_result(result):
                self.best_result = result
                print(f"New best result found! Approximation Rate: {result['test_approximation_rate']:.2f}%")

            # 中間結果保存
            self._save_intermediate_results()

        # 最終結果保存
        self._save_final_results("grid_search")

        return self.best_result

    def random_search(self, n_trials: int = None) -> Dict[str, Any]:
        """ランダムサーチを実行"""
        print("Starting Random Search...")

        if n_trials is None:
            n_trials = self.tuning_config['random_search']['n_trials']

        param_ranges = self.tuning_config['random_search']['parameters']

        print(f"Total trials: {n_trials}")

        self.results = []

        for i in range(n_trials):
            self.current_trial = i

            # ランダムパラメータ生成
            params = self._generate_random_params(param_ranges)

            # 評価実行
            result = self._evaluate_params(params)
            self.results.append(result)

            # ベスト結果の更新
            if self._is_better_result(result):
                self.best_result = result
                print(f"New best result found! Approximation Rate: {result['test_approximation_rate']:.2f}%")

            # 中間結果保存
            self._save_intermediate_results()

        # 最終結果保存
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
                    # 対数スケール
                    log_min = np.log10(config['min'])
                    log_max = np.log10(config['max'])
                    params[param_name] = 10 ** random.uniform(log_min, log_max)
                else:
                    params[param_name] = random.uniform(config['min'], config['max'])
            elif param_type == 'choice':
                params[param_name] = random.choice(config['choices'])

        return params

    def _is_better_result(self, result: Dict[str, Any]) -> bool:
        """結果がベストかどうかを判定（設定ファイルベース）"""
        if self.best_result is None:
            return True

        # 設定ファイルから評価設定を取得
        evaluation_config = self.tuning_config.get('evaluation', {})
        primary_metric = evaluation_config.get('primary_metric', 'test_approximation_rate')
        optimization_direction = evaluation_config.get('optimization_direction', 'maximize')

        # プライマリメトリクスの値を取得
        current_value = result.get(primary_metric, 0.0)
        best_value = self.best_result.get(primary_metric, 0.0)

        # 最適化方向に応じて比較
        if optimization_direction == 'maximize':
            return current_value > best_value
        elif optimization_direction == 'minimize':
            return current_value < best_value
        else:
            # デフォルトは最大化
            return current_value > best_value

    def _save_intermediate_results(self):
        """中間結果を保存"""
        csv_path = os.path.join(self.results_dir, f"intermediate_results_{self.timestamp}.csv")

        if self.results:
            # CSVヘッダーの作成
            fieldnames = ['trial', 'timestamp']

            # パラメータ名を追加
            if 'params' in self.results[0]:
                param_names = list(self.results[0]['params'].keys())
                fieldnames.extend(param_names)

            # メトリクス名を追加
            metric_names = [
                'test_approximation_rate', 'val_approximation_rate',
                'test_infeasible_rate', 'val_infeasible_rate',
                'test_mean_load_factor', 'val_mean_load_factor',
                'test_loss', 'val_loss',
                'test_time', 'val_time'
            ]
            fieldnames.extend(metric_names)

            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for result in self.results:
                    row = {
                        'trial': result['trial'],
                        'timestamp': result['timestamp']
                    }

                    # パラメータを追加
                    if 'params' in result:
                        row.update(result['params'])

                    # メトリクスを追加
                    for metric in metric_names:
                        row[metric] = result.get(metric, '')

                    writer.writerow(row)

    def _save_final_results(self, search_type: str):
        """最終結果を保存"""
        # JSON形式で詳細結果を保存
        json_path = os.path.join(self.results_dir, f"{search_type}_results_{self.timestamp}.json")

        final_results = {
            'search_type': search_type,
            'timestamp': self.timestamp,
            'total_trials': len(self.results),
            'best_result': self.best_result,
            'all_results': self.results,
            'tuning_config': self.tuning_config
        }

        with open(json_path, 'w') as f:
            json.dump(final_results, f, indent=2)

        # サマリーレポート
        self._generate_summary_report(search_type)

        print(f"\nTuning completed!")
        print(f"Results saved to: {json_path}")
        print(f"Best approximation rate: {self.best_result['test_approximation_rate']:.2f}%")

    def _generate_summary_report(self, search_type: str):
        """サマリーレポートを生成"""
        report_path = os.path.join(self.results_dir, f"{search_type}_summary_{self.timestamp}.txt")

        with open(report_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write(f"HYPERPARAMETER TUNING SUMMARY - {search_type.upper()}\n")
            f.write("="*60 + "\n")
            f.write(f"Timestamp: {self.timestamp}\n")
            f.write(f"Total Trials: {len(self.results)}\n\n")

            if self.best_result:
                f.write("BEST RESULT:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Trial: {self.best_result['trial']}\n")
                f.write(f"Test Approximation Rate: {self.best_result['test_approximation_rate']:.2f}%\n")
                f.write(f"Val Approximation Rate: {self.best_result['val_approximation_rate']:.2f}%\n")
                f.write(f"Test Infeasible Rate: {self.best_result['test_infeasible_rate']:.2f}%\n")
                f.write(f"Val Infeasible Rate: {self.best_result['val_infeasible_rate']:.2f}%\n")

                f.write("\nBest Parameters:\n")
                for param, value in self.best_result['params'].items():
                    f.write(f"  {param}: {value}\n")

                f.write("\n" + "="*60 + "\n")

                # 統計情報
                if len(self.results) > 1:
                    approx_rates = [r['test_approximation_rate'] for r in self.results if 'error' not in r]
                    if approx_rates:
                        f.write("STATISTICS:\n")
                        f.write("-" * 30 + "\n")
                        f.write(f"Mean Approximation Rate: {np.mean(approx_rates):.2f}%\n")
                        f.write(f"Std Approximation Rate: {np.std(approx_rates):.2f}%\n")
                        f.write(f"Min Approximation Rate: {np.min(approx_rates):.2f}%\n")
                        f.write(f"Max Approximation Rate: {np.max(approx_rates):.2f}%\n")

        print(f"Summary report saved to: {report_path}")