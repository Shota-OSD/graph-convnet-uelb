import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional


class BaseMetricsLogger(ABC):
    """GCN・RL-KSP 共通のメトリクス記録基底クラス"""

    def __init__(self, save_dir: str = "logs"):
        self.val_approximation_rate_list = []
        self.test_approximation_rate_list = []

        self.train_time_list = []
        self.val_time_list = []
        self.test_time_list = []
        self.total_train_time = 0.0
        self.total_test_time = 0.0

        self.val_epochs = []
        self.test_epochs = []

        self.save_dir = save_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        os.makedirs(self.save_dir, exist_ok=True)

    def log_val_metrics(self, approximation_rate: float, val_time: Optional[float] = None,
                        epoch: Optional[int] = None):
        """検証メトリクスの共通フィールドを記録"""
        self.val_approximation_rate_list.append(approximation_rate)
        if val_time is not None:
            self.val_time_list.append(val_time)
        if epoch is not None:
            self.val_epochs.append(epoch)

    def log_test_metrics(self, approximation_rate: float, test_time: Optional[float] = None,
                         epoch: Optional[int] = None):
        """テストメトリクスの共通フィールドを記録"""
        self.test_approximation_rate_list.append(approximation_rate)
        if test_time is not None:
            self.test_time_list.append(test_time)
            self.total_test_time += test_time
        if epoch is not None:
            self.test_epochs.append(epoch)

    def calculate_time_per_data(self, num_train_data: int, num_test_data: int) -> dict:
        """1データあたりの経過時間を計算"""
        total_train_samples = num_train_data * len(self.train_time_list) if self.train_time_list else 0
        total_test_samples = num_test_data * len(self.test_time_list) if self.test_time_list else 0
        return {
            'train_time_per_data': self.total_train_time / total_train_samples if total_train_samples > 0 else 0.0,
            'test_time_per_data': self.total_test_time / total_test_samples if total_test_samples > 0 else 0.0,
            'total_train_samples': total_train_samples,
            'total_test_samples': total_test_samples,
        }

    @abstractmethod
    def get_final_metrics(self) -> dict:
        """最終メトリクスを返す（サブクラスで実装）"""

    @abstractmethod
    def save_results(self, config_info: Optional[dict] = None):
        """結果をファイルに保存（サブクラスで実装）"""

    @abstractmethod
    def print_summary(self, config_info: Optional[dict] = None):
        """サマリーを表示（サブクラスで実装）"""
