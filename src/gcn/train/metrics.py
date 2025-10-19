import numpy as np
import pickle
import os
import torch
from datetime import datetime
from typing import List, Tuple, Any

class MetricsLogger:
    """メトリクスの記録と出力を管理するクラス"""
    
    def __init__(self, save_dir: str = "logs"):
        self.train_loss_list = []
        self.train_err_edges_list = []
        self.val_approximation_rate_list = []
        self.test_approximation_rate_list = []
        
        # 時間関連のメトリクスを追加
        self.train_time_list = []
        self.val_time_list = []
        self.test_time_list = []
        self.total_train_time = 0.0
        self.total_test_time = 0.0

        # RL-specific metrics (existing)
        self.rl_reward_list = []
        self.rl_advantage_list = []
        self.rl_entropy_list = []
        self.rl_load_factor_list = []
        self.rl_baseline_list = []

        # RL-specific metrics (new - path quality)
        self.rl_complete_paths_rate_list = []  # パス完成率
        self.rl_finite_solution_rate_list = []  # 有限解の割合
        self.rl_avg_finite_load_factor_list = []  # 有限解の平均負荷率
        self.rl_avg_path_length_list = []  # 平均パス長
        self.rl_commodity_success_rate_list = []  # コモディティ成功率
        self.rl_capacity_violation_rate_list = []  # 容量違反率

        self.save_dir = save_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存ディレクトリの作成
        os.makedirs(self.save_dir, exist_ok=True)
    
    def log_train_metrics(self, loss: float, edge_error: float, train_time: float = None):
        """トレーニングメトリクスを記録"""
        self.train_loss_list.append(loss)
        self.train_err_edges_list.append(edge_error)
        if train_time is not None:
            self.train_time_list.append(train_time)
            self.total_train_time += train_time
    
    def log_val_metrics(self, approximation_rate: float, val_time: float = None):
        """検証メトリクスを記録"""
        self.val_approximation_rate_list.append(approximation_rate)
        if val_time is not None:
            self.val_time_list.append(val_time)
    
    def log_test_metrics(self, approximation_rate: float, test_time: float = None):
        """テストメトリクスを記録"""
        self.test_approximation_rate_list.append(approximation_rate)
        if test_time is not None:
            self.test_time_list.append(test_time)
            self.total_test_time += test_time

    def log_rl_metrics(self, epoch: int, rl_metrics: dict):
        """RL特有のメトリクスを記録"""
        # Existing metrics
        self.rl_reward_list.append(rl_metrics.get('reward', 0.0))
        self.rl_advantage_list.append(rl_metrics.get('advantage', 0.0))
        self.rl_entropy_list.append(rl_metrics.get('entropy', 0.0))
        self.rl_load_factor_list.append(rl_metrics.get('load_factor', 0.0))
        self.rl_baseline_list.append(rl_metrics.get('baseline', 0.0))

        # New path quality metrics
        self.rl_complete_paths_rate_list.append(rl_metrics.get('complete_paths_rate', 0.0))
        self.rl_finite_solution_rate_list.append(rl_metrics.get('finite_solution_rate', 0.0))
        self.rl_avg_finite_load_factor_list.append(rl_metrics.get('avg_finite_load_factor', 0.0))
        self.rl_avg_path_length_list.append(rl_metrics.get('avg_path_length', 0.0))
        self.rl_commodity_success_rate_list.append(rl_metrics.get('commodity_success_rate', 0.0))
        self.rl_capacity_violation_rate_list.append(rl_metrics.get('capacity_violation_rate', 0.0))

        # CSV形式でも保存（拡張版）
        csv_filename = os.path.join(self.save_dir, f"rl_gcn_metrics_{self.timestamp}.csv")
        file_exists = os.path.exists(csv_filename)

        with open(csv_filename, 'a', encoding='utf-8') as f:
            if not file_exists:
                # ヘッダー行（拡張）
                f.write("epoch,reward,reward_std,advantage,advantage_std,entropy,load_factor,load_factor_std,baseline,")
                f.write("complete_paths_rate,finite_solution_rate,avg_finite_load_factor,avg_path_length,")
                f.write("commodity_success_rate,capacity_violation_rate\n")
            # データ行（拡張）
            f.write(f"{epoch},{rl_metrics.get('reward', 0.0):.6f},{rl_metrics.get('reward_std', 0.0):.6f},")
            f.write(f"{rl_metrics.get('advantage', 0.0):.6f},{rl_metrics.get('advantage_std', 0.0):.6f},")
            f.write(f"{rl_metrics.get('entropy', 0.0):.6f},{rl_metrics.get('load_factor', 0.0):.6f},")
            f.write(f"{rl_metrics.get('load_factor_std', 0.0):.6f},{rl_metrics.get('baseline', 0.0):.6f},")
            f.write(f"{rl_metrics.get('complete_paths_rate', 0.0):.6f},{rl_metrics.get('finite_solution_rate', 0.0):.6f},")
            f.write(f"{rl_metrics.get('avg_finite_load_factor', 0.0):.6f},{rl_metrics.get('avg_path_length', 0.0):.6f},")
            f.write(f"{rl_metrics.get('commodity_success_rate', 0.0):.6f},{rl_metrics.get('capacity_violation_rate', 0.0):.6f}\n")
    
    def get_final_metrics(self) -> dict:
        """最終メトリクスを取得"""
        return {
            'final_train_loss': self.train_loss_list[-1] if self.train_loss_list else 0.0,
            'final_edge_error': self.train_err_edges_list[-1] if self.train_err_edges_list else 0.0,
            'final_val_approximation_rate': self.val_approximation_rate_list[-1] if self.val_approximation_rate_list else 0.0,
            'final_test_approximation_rate': self.test_approximation_rate_list[-1] if self.test_approximation_rate_list else 0.0,
            'best_val_approximation_rate': max(self.val_approximation_rate_list) if self.val_approximation_rate_list else 0.0,
            'best_test_approximation_rate': max(self.test_approximation_rate_list) if self.test_approximation_rate_list else 0.0,
            'total_train_time': self.total_train_time,
            'total_test_time': self.total_test_time,
            'avg_train_time_per_epoch': np.mean(self.train_time_list) if self.train_time_list else 0.0,
            'avg_test_time_per_epoch': np.mean(self.test_time_list) if self.test_time_list else 0.0
        }
    
    def calculate_time_per_data(self, num_train_data: int, num_test_data: int) -> dict:
        """一つのデータあたりの経過時間を計算"""
        total_train_samples = num_train_data * len(self.train_time_list) if self.train_time_list else 0
        total_test_samples = num_test_data * len(self.test_time_list) if self.test_time_list else 0
        
        time_per_data = {
            'train_time_per_data': self.total_train_time / total_train_samples if total_train_samples > 0 else 0.0,
            'test_time_per_data': self.total_test_time / total_test_samples if total_test_samples > 0 else 0.0,
            'total_train_samples': total_train_samples,
            'total_test_samples': total_test_samples
        }
        return time_per_data
    
    def save_results(self, config_info: dict = None):
        """結果をファイルに保存"""
        # 時間関連の計算
        num_train_data = config_info.get('num_train_data', 0) if config_info else 0
        num_test_data = config_info.get('num_test_data', 0) if config_info else 0
        time_per_data = self.calculate_time_per_data(num_train_data, num_test_data)
        
        # 結果データの準備
        results = {
            'timestamp': self.timestamp,
            'train_loss_list': self.train_loss_list,
            'train_err_edges_list': self.train_err_edges_list,
            'val_approximation_rate_list': self.val_approximation_rate_list,
            'test_approximation_rate_list': self.test_approximation_rate_list,
            'train_time_list': self.train_time_list,
            'val_time_list': self.val_time_list,
            'test_time_list': self.test_time_list,
            'final_metrics': self.get_final_metrics(),
            'time_per_data': time_per_data,
            'config_info': config_info or {}
        }
        
        # Pickleファイルとして保存（Tensorオブジェクトも含めて保存可能）
        pickle_filename = os.path.join(self.save_dir, f"training_results_{self.timestamp}.pkl")
        try:
            with open(pickle_filename, 'wb') as f:
                pickle.dump(results, f)
            print(f"Results saved to Pickle: {pickle_filename}")
        except Exception as e:
            print(f"Pickle save error: {e}")
            # 簡略化されたデータで再試行
            simplified_results = {
                'timestamp': self.timestamp,
                'train_loss_list': self.train_loss_list,
                'train_err_edges_list': self.train_err_edges_list,
                'val_approximation_rate_list': self.val_approximation_rate_list,
                'test_approximation_rate_list': self.test_approximation_rate_list,
                'final_metrics': self.get_final_metrics(),
                'config_info': {'config_file': config_info.get('config_file', 'unknown') if config_info else 'unknown'}
            }
            with open(pickle_filename, 'wb') as f:
                pickle.dump(simplified_results, f)
            print(f"Simplified results saved to Pickle: {pickle_filename}")
        
        # テキストファイルとして保存（読みやすい形式）
        txt_filename = os.path.join(self.save_dir, f"training_results_{self.timestamp}.txt")
        with open(txt_filename, 'w', encoding='utf-8') as f:
            f.write("="*50 + "\n")
            f.write("TRAINING/EVALUATION RESULTS\n")
            f.write("="*50 + "\n")
            f.write(f"Timestamp: {self.timestamp}\n")
            f.write(f"Mode: {config_info.get('mode', 'unknown') if config_info else 'unknown'}\n")
            f.write(f"Config File: {config_info.get('config_file', 'unknown') if config_info else 'unknown'}\n")
            f.write(f"Model Hash: {config_info.get('model_hash', 'unknown') if config_info else 'unknown'}\n\n")
            
            # 最終メトリクス
            metrics = self.get_final_metrics()
            f.write("FINAL METRICS:\n")
            f.write(f"  Final Train Loss: {metrics['final_train_loss']:.4f}\n")
            f.write(f"  Final Train Edge Error: {metrics['final_edge_error']:.2f}%\n")
            if self.val_approximation_rate_list:
                f.write(f"  Final Val Approximation Rate: {metrics['final_val_approximation_rate']:.2f}%\n")
                f.write(f"  Best Val Approximation Rate: {metrics['best_val_approximation_rate']:.2f}%\n")
            if self.test_approximation_rate_list:
                f.write(f"  Final Test Approximation Rate: {metrics['final_test_approximation_rate']:.2f}%\n")
                f.write(f"  Best Test Approximation Rate: {metrics['best_test_approximation_rate']:.2f}%\n")
            
            # 時間関連のメトリクス
            f.write("\nTIME METRICS:\n")
            f.write(f"  Total Training Time: {metrics['total_train_time']:.2f}s\n")
            f.write(f"  Total Test Time: {metrics['total_test_time']:.2f}s\n")
            f.write(f"  Average Training Time per Epoch: {metrics['avg_train_time_per_epoch']:.2f}s\n")
            f.write(f"  Average Test Time per Epoch: {metrics['avg_test_time_per_epoch']:.2f}s\n")
            
            # 一つのデータあたりの時間
            time_per_data = self.calculate_time_per_data(
                config_info.get('num_train_data', 0) if config_info else 0,
                config_info.get('num_test_data', 0) if config_info else 0
            )
            f.write(f"  Training Time per Data: {time_per_data['train_time_per_data']:.4f}s\n")
            f.write(f"  Test Time per Data: {time_per_data['test_time_per_data']:.4f}s\n")
            f.write(f"  Total Training Samples: {time_per_data['total_train_samples']}\n")
            f.write(f"  Total Test Samples: {time_per_data['total_test_samples']}\n")
            
            f.write("\n" + "="*50 + "\n")
            f.write("DETAILED RESULTS\n")
            f.write("="*50 + "\n")
            
            # エポックごとの詳細結果
            f.write(f"{'Epoch':<6} {'Train Loss':<12} {'Edge Error':<12} {'Train Time':<12} {'Val Approx':<12} {'Test Approx':<12}\n")
            f.write("-" * 72 + "\n")
            
            max_epochs = len(self.train_loss_list)
            for epoch in range(max_epochs):
                train_loss = self.train_loss_list[epoch]
                edge_error = self.train_err_edges_list[epoch]
                train_time = self.train_time_list[epoch] if epoch < len(self.train_time_list) else 0.0
                
                # 検証とテストの近似率（該当するエポックのみ）
                val_approx = "N/A"
                test_approx = "N/A"
                
                # 検証近似率の取得（簡略化）
                if epoch < len(self.val_approximation_rate_list):
                    val_approx = f"{self.val_approximation_rate_list[epoch]:.2f}%"
                
                # テスト近似率の取得（簡略化）
                if epoch < len(self.test_approximation_rate_list):
                    test_approx = f"{self.test_approximation_rate_list[epoch]:.2f}%"
                
                f.write(f"{epoch+1:<6} {train_loss:<12.4f} {edge_error:<12.2f} {train_time:<12.2f} {val_approx:<12} {test_approx:<12}\n")
            
            f.write("-" * 72 + "\n")
        
        print(f"Results saved to Text: {txt_filename}")
        
        return pickle_filename, txt_filename
    
    def print_summary(self, config_info: dict = None):
        """最終結果のサマリーを表示"""
        metrics = self.get_final_metrics()
        print("\n" + "="*50)
        print("TRAINING SUMMARY")
        print("="*50)
        print(f"Final Train Loss: {metrics['final_train_loss']:.4f}")
        print(f"Final Train Edge Error: {metrics['final_edge_error']:.2f}%")
        if self.val_approximation_rate_list:
            print(f"Final Val Approximation Rate: {metrics['final_val_approximation_rate']:.2f}%")
        if self.test_approximation_rate_list:
            print(f"Final Test Approximation Rate: {metrics['final_test_approximation_rate']:.2f}%")
        
        # 時間関連のメトリクス
        print("\nTIME METRICS:")
        print(f"Total Training Time: {metrics['total_train_time']:.2f}s")
        print(f"Total Test Time: {metrics['total_test_time']:.2f}s")
        print(f"Average Training Time per Epoch: {metrics['avg_train_time_per_epoch']:.2f}s")
        print(f"Average Test Time per Epoch: {metrics['avg_test_time_per_epoch']:.2f}s")
        
        # 一つのデータあたりの時間
        num_train_data = config_info.get('num_train_data', 0) if config_info else 0
        num_test_data = config_info.get('num_test_data', 0) if config_info else 0
        time_per_data = self.calculate_time_per_data(num_train_data, num_test_data)
        print(f"Training Time per Data: {time_per_data['train_time_per_data']:.4f}s")
        print(f"Test Time per Data: {time_per_data['test_time_per_data']:.4f}s")
        print(f"Total Training Samples: {time_per_data['total_train_samples']}")
        print(f"Total Test Samples: {time_per_data['total_test_samples']}")
    
    def print_evaluation_summary(self, config_info: dict = None):
        """評価のみモード用のサマリー表示"""
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        
        # 検証結果
        if self.val_approximation_rate_list:
            val_approx = self.val_approximation_rate_list[-1]
            val_time = self.val_time_list[-1] if self.val_time_list else 0.0
            print(f"VALIDATION:")
            print(f"  Approximation Rate: {val_approx:.2f}%")
            print(f"  Execution Time: {val_time:.2f}s")
        
        # テスト結果
        if self.test_approximation_rate_list:
            test_approx = self.test_approximation_rate_list[-1]
            test_time = self.test_time_list[-1] if self.test_time_list else 0.0
            print(f"TEST:")
            print(f"  Approximation Rate: {test_approx:.2f}%")
            print(f"  Execution Time: {test_time:.2f}s")
        
        # 総合結果
        if self.val_approximation_rate_list and self.test_approximation_rate_list:
            avg_approx = (self.val_approximation_rate_list[-1] + self.test_approximation_rate_list[-1]) / 2
            total_time = (self.val_time_list[-1] if self.val_time_list else 0.0) + (self.test_time_list[-1] if self.test_time_list else 0.0)
            print(f"OVERALL:")
            print(f"  Average Approximation Rate: {avg_approx:.2f}%")
            print(f"  Total Evaluation Time: {total_time:.2f}s")
        
        # 設定情報（シンプル）
        if config_info:
            print(f"\nUsed Config: {config_info.get('config_file', 'Unknown')}")
            print(f"Model Hash: {config_info.get('model_hash', 'Unknown')}")

def metrics_to_str(epoch: int, time: float, learning_rate: float, loss: float, 
                  mean_maximum_load_factor: float, gt_load_factor: float, 
                  approximation_rate: float, infeasible_rate: float) -> str:
    """メトリクスを文字列形式で返す関数"""
    return (f'epoch:{epoch:02d}\t'
            f'execution time:{time:.8f}s\t'
            f'lr:{learning_rate:.2e}\t'
            f'loss:{loss:.4f}\t'
            f'mean_maximum_load_factor:{mean_maximum_load_factor:.3f}\t'
            f'gt_load_factor:{gt_load_factor:.3f}\t'
            f'approximation_rate:{approximation_rate:.3f}\t'
            f'infeasible_rate:{infeasible_rate:.3f}')

def load_training_results(pickle_filepath: str) -> dict:
    """保存されたトレーニング結果を読み込む関数"""
    try:
        with open(pickle_filepath, 'rb') as f:
            results = pickle.load(f)
        return results
    except Exception as e:
        print(f"Error loading results from {pickle_filepath}: {e}")
        return None

def print_training_summary(results: dict):
    """保存された結果のサマリーを表示する関数"""
    if results is None:
        print("No results to display")
        return
    
    print("\n" + "="*50)
    print("LOADED TRAINING RESULTS")
    print("="*50)
    print(f"Timestamp: {results.get('timestamp', 'Unknown')}")
    
    final_metrics = results.get('final_metrics', {})
    print(f"Final Train Loss: {final_metrics.get('final_train_loss', 0.0):.4f}")
    print(f"Final Train Edge Error: {final_metrics.get('final_edge_error', 0.0):.2f}%")
    print(f"Final Val Approximation Rate: {final_metrics.get('final_val_approximation_rate', 0.0):.2f}%")
    print(f"Final Test Approximation Rate: {final_metrics.get('final_test_approximation_rate', 0.0):.2f}%")
    print(f"Best Val Approximation Rate: {final_metrics.get('best_val_approximation_rate', 0.0):.2f}%")
    print(f"Best Test Approximation Rate: {final_metrics.get('best_test_approximation_rate', 0.0):.2f}%")
    
    # 設定情報の表示
    config_info = results.get('config_info', {})
    if config_info:
        print(f"\nConfig File: {config_info.get('config_file', 'Unknown')}")
        print(f"Max Epochs: {config_info.get('max_epochs', 'Unknown')}")
        print(f"Learning Rate: {config_info.get('initial_learning_rate', 'Unknown')}")

# 使用例:
# results = load_training_results("logs/training_results_20241201_143022.pkl")
# print_training_summary(results) 