import os
import shutil
from typing import Optional
from .create_data_files import create_data_files

class DatasetManager:
    """データセットの作成と管理を担当するクラス"""
    
    def __init__(self, config):
        self.config = config
    
    def remake_dataset(self) -> bool:
        """データセットの再作成を確認"""
        ans = input("トレーニング・検証・テスト用データを再作成しますか？ (y/n): ").strip().lower()
        return ans in ["y", "yes"]
    
    def create_all_datasets(self):
        """全てのデータセットを作成"""
        # 既存のデータを削除
        for mode in ["val", "test", "train"]:
            data_dir = f"./data/{mode}_data"
            if os.path.exists(data_dir):
                shutil.rmtree(data_dir)
                print(f"Removed existing {mode} data directory")
        
        # 新しいデータを生成
        for mode in ["val", "test", "train"]:
            create_data_files(self.config, data_mode=mode)
    
    def test_data_loading(self):
        """データローディングのテスト"""
        mode = "test"
        num_data = getattr(self.config, f'num_{mode}_data')
        batch_size = self.config.batch_size
        from .dataset_reader import DatasetReader
        dataset = DatasetReader(num_data, batch_size, mode)
        print(f"Number of batches of size {batch_size}: {dataset.max_iter}")
        batch = next(iter(dataset))
        print("edges shape:", batch.edges.shape)
        
        # プロット機能（必要に応じて）
        try:
            import matplotlib.pyplot as plt
            from ..visualization.plot_utils import plot_uelb
            f = plt.figure(figsize=(5, 5))
            a = f.add_subplot(111)
            plot_uelb(a, batch.edges[0], batch.edges_target[0])
            plt.close(f)
        except ImportError:
            print("matplotlib not available for plotting") 