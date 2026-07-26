import numpy as np
import pickle
import os
from typing import Optional

from src.common.train.base_metrics import BaseMetricsLogger


class RLKSPMetricsLogger(BaseMetricsLogger):
    """RL-KSP 用メトリクス記録クラス"""

    def __init__(self, save_dir: str = "logs"):
        super().__init__(save_dir)

        # エピソード単位の学習メトリクス
        self.train_loss_list = []
        self.episode_reward_list = []

        # テスト時のベースライン比較メトリクス
        self.test_avg_rl_load_list = []
        self.test_avg_exact_load_list = []
        self.test_avg_ksp_ilp_load_list = []
        self.test_ksp_ilp_approx_rate_list = []
        self.test_exact_approx_rate_list = []

    def log_train_metrics(self, loss: float, episode_reward: float,
                          train_time: Optional[float] = None):
        """エピソード単位の学習メトリクスを記録"""
        self.train_loss_list.append(loss)
        self.episode_reward_list.append(episode_reward)
        if train_time is not None:
            self.train_time_list.append(train_time)
            self.total_train_time += train_time

    def log_val_metrics(self, *args, **kwargs):
        """RL-KSP は検証フェーズを持たないため no-op"""

    def log_test_metrics(self, approximation_rate: float, test_time: Optional[float] = None,
                         epoch: Optional[int] = None, avg_rl_load: Optional[float] = None,
                         avg_exact_load: Optional[float] = None,
                         avg_ksp_ilp_load: Optional[float] = None,
                         ksp_ilp_approx_rate: Optional[float] = None,
                         exact_approx_rate: Optional[float] = None):
        """テストメトリクスを記録"""
        super().log_test_metrics(approximation_rate, test_time, epoch)
        if avg_rl_load is not None:
            self.test_avg_rl_load_list.append(avg_rl_load)
        if avg_exact_load is not None:
            self.test_avg_exact_load_list.append(avg_exact_load)
        if avg_ksp_ilp_load is not None:
            self.test_avg_ksp_ilp_load_list.append(avg_ksp_ilp_load)
        if ksp_ilp_approx_rate is not None:
            self.test_ksp_ilp_approx_rate_list.append(ksp_ilp_approx_rate)
        if exact_approx_rate is not None:
            self.test_exact_approx_rate_list.append(exact_approx_rate)

    def get_final_metrics(self) -> dict:
        """最終メトリクスを返す（Final/Best 命名なし）"""
        return {
            'avg_test_approx_rate': self.test_approximation_rate_list[-1] if self.test_approximation_rate_list else 0.0,
            'avg_rl_load': self.test_avg_rl_load_list[-1] if self.test_avg_rl_load_list else None,
            'avg_ksp_ilp_load': self.test_avg_ksp_ilp_load_list[-1] if self.test_avg_ksp_ilp_load_list else None,
            'avg_exact_load': self.test_avg_exact_load_list[-1] if self.test_avg_exact_load_list else None,
            'ksp_ilp_approx_rate': self.test_ksp_ilp_approx_rate_list[-1] if self.test_ksp_ilp_approx_rate_list else None,
            'exact_approx_rate': self.test_exact_approx_rate_list[-1] if self.test_exact_approx_rate_list else None,
            'total_train_time': self.total_train_time,
            'total_test_time': self.total_test_time,
            'avg_episode_time': np.mean(self.train_time_list) if self.train_time_list else 0.0,
            'avg_test_time': np.mean(self.test_time_list) if self.test_time_list else 0.0,
        }

    def save_results(self, config_info: Optional[dict] = None):
        """結果をファイルに保存"""
        num_train_data = config_info.get('num_train_data', 0) if config_info else 0
        num_test_data = config_info.get('num_test_data', 0) if config_info else 0
        time_per_data = self.calculate_time_per_data(num_train_data, num_test_data)

        results = {
            'timestamp': self.timestamp,
            'train_loss_list': self.train_loss_list,
            'episode_reward_list': self.episode_reward_list,
            'test_approximation_rate_list': self.test_approximation_rate_list,
            'test_avg_rl_load_list': self.test_avg_rl_load_list,
            'test_avg_exact_load_list': self.test_avg_exact_load_list,
            'test_avg_ksp_ilp_load_list': self.test_avg_ksp_ilp_load_list,
            'test_ksp_ilp_approx_rate_list': self.test_ksp_ilp_approx_rate_list,
            'test_exact_approx_rate_list': self.test_exact_approx_rate_list,
            'train_time_list': self.train_time_list,
            'test_time_list': self.test_time_list,
            'final_metrics': self.get_final_metrics(),
            'time_per_data': time_per_data,
            'config_info': config_info or {},
        }

        pickle_filename = os.path.join(self.save_dir, f"training_results_{self.timestamp}.pkl")
        try:
            with open(pickle_filename, 'wb') as f:
                pickle.dump(results, f)
            print(f"Results saved to Pickle: {pickle_filename}")
        except Exception as e:
            print(f"Pickle save error: {e}")
            simplified = {
                'timestamp': self.timestamp,
                'train_loss_list': self.train_loss_list,
                'test_approximation_rate_list': self.test_approximation_rate_list,
                'final_metrics': self.get_final_metrics(),
                'config_info': {'config_file': config_info.get('config_file', 'unknown') if config_info else 'unknown'},
            }
            with open(pickle_filename, 'wb') as f:
                pickle.dump(simplified, f)
            print(f"Simplified results saved to Pickle: {pickle_filename}")

        txt_filename = os.path.join(self.save_dir, f"training_results_{self.timestamp}.txt")
        metrics = self.get_final_metrics()
        with open(txt_filename, 'w', encoding='utf-8') as f:
            f.write("=" * 50 + "\n")
            f.write("RL-KSP TRAINING/EVALUATION RESULTS\n")
            f.write("=" * 50 + "\n")
            f.write(f"Timestamp: {self.timestamp}\n")
            f.write(f"Mode: {config_info.get('mode', 'unknown') if config_info else 'unknown'}\n")
            f.write(f"Config File: {config_info.get('config_file', 'unknown') if config_info else 'unknown'}\n\n")

            f.write("TEST METRICS:\n")
            if self.test_approximation_rate_list:
                f.write(f"  Avg Test Approx Rate (KSP-ILP base): {metrics['avg_test_approx_rate']:.2f}%\n")
            if metrics['exact_approx_rate'] is not None:
                f.write(f"  Avg Test Approx Rate (Exact base): {metrics['exact_approx_rate']:.2f}%\n")
            else:
                f.write(f"  Avg Test Approx Rate (Exact base): N/A\n")
            if metrics['avg_rl_load'] is not None:
                f.write(f"  Avg RL Load Factor: {metrics['avg_rl_load']:.6f}\n")
            if metrics['avg_ksp_ilp_load'] is not None:
                f.write(f"  Avg KSP-ILP Load Factor: {metrics['avg_ksp_ilp_load']:.6f}\n")
            if metrics['avg_exact_load'] is not None:
                f.write(f"  Avg Exact Solution Load Factor: {metrics['avg_exact_load']:.6f}\n")
            else:
                f.write(f"  Avg Exact Solution Load Factor: N/A\n")
            if metrics['ksp_ilp_approx_rate'] is not None:
                f.write(f"  KSP-ILP Approx Rate (Exact/KSP-ILP): {metrics['ksp_ilp_approx_rate']:.2f}%\n")
            else:
                f.write(f"  KSP-ILP Approx Rate (Exact/KSP-ILP): N/A\n")

            f.write("\nTIME METRICS:\n")
            f.write(f"  Total Training Time: {metrics['total_train_time']:.2f}s\n")
            f.write(f"  Total Test Time: {metrics['total_test_time']:.2f}s\n")
            f.write(f"  Average Time per Episode: {metrics['avg_episode_time']:.2f}s\n")
            f.write(f"  Test Time per Data: {time_per_data['test_time_per_data']:.4f}s\n")
            f.write(f"  Total Test Samples: {time_per_data['total_test_samples']}\n")

            f.write("\n" + "=" * 50 + "\n")
            f.write("EPISODE TRAINING SUMMARY\n")
            f.write("=" * 50 + "\n")
            f.write(f"Total Episodes: {len(self.train_loss_list)}\n")
            if self.train_loss_list:
                f.write(f"  Final Episode Loss: {self.train_loss_list[-1]:.6f}\n")
                f.write(f"  Avg Episode Loss: {np.mean(self.train_loss_list):.6f}\n")
            if self.episode_reward_list:
                f.write(f"  Final Episode Reward: {self.episode_reward_list[-1]:.6f}\n")
                f.write(f"  Avg Episode Reward: {np.mean(self.episode_reward_list):.6f}\n")

        print(f"Results saved to Text: {txt_filename}")
        return pickle_filename, txt_filename

    def print_summary(self, config_info: Optional[dict] = None):
        """RL-KSP 向けサマリーを表示"""
        metrics = self.get_final_metrics()
        print("\n" + "=" * 50)
        print("RL-KSP TEST SUMMARY")
        print("=" * 50)

        if self.test_approximation_rate_list:
            print(f"Avg Test Approx Rate (KSP-ILP base): {metrics['avg_test_approx_rate']:.2f}%")
        if metrics['exact_approx_rate'] is not None:
            print(f"Avg Test Approx Rate (Exact base): {metrics['exact_approx_rate']:.2f}%")
        else:
            print("Avg Test Approx Rate (Exact base): N/A")
        if metrics['avg_rl_load'] is not None:
            print(f"Avg RL Load Factor: {metrics['avg_rl_load']:.6f}")
        if metrics['avg_ksp_ilp_load'] is not None:
            print(f"Avg KSP-ILP Load Factor: {metrics['avg_ksp_ilp_load']:.6f}")

        print("\nTIME METRICS:")
        print(f"Total Training Time: {metrics['total_train_time']:.2f}s")
        print(f"Total Test Time: {metrics['total_test_time']:.2f}s")

        num_train_data = config_info.get('num_train_data', 0) if config_info else 0
        num_test_data = config_info.get('num_test_data', 0) if config_info else 0
        time_per_data = self.calculate_time_per_data(num_train_data, num_test_data)
        print(f"Test Time per Data: {time_per_data['test_time_per_data']:.4f}s")
        print(f"Total Test Samples: {time_per_data['total_test_samples']}")
