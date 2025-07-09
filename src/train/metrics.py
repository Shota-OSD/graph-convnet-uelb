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
        self.save_dir = save_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存ディレクトリの作成
        os.makedirs(self.save_dir, exist_ok=True)
    
    def log_train_metrics(self, loss: float, edge_error: float):
        """トレーニングメトリクスを記録"""
        self.train_loss_list.append(loss)
        self.train_err_edges_list.append(edge_error)
    
    def log_val_metrics(self, approximation_rate: float):
        """検証メトリクスを記録"""
        self.val_approximation_rate_list.append(approximation_rate)
    
    def log_test_metrics(self, approximation_rate: float):
        """テストメトリクスを記録"""
        self.test_approximation_rate_list.append(approximation_rate)
    
    def get_final_metrics(self) -> dict:
        """最終メトリクスを取得"""
        return {
            'final_train_loss': self.train_loss_list[-1] if self.train_loss_list else 0.0,
            'final_edge_error': self.train_err_edges_list[-1] if self.train_err_edges_list else 0.0,
            'final_val_approximation_rate': self.val_approximation_rate_list[-1] if self.val_approximation_rate_list else 0.0,
            'final_test_approximation_rate': self.test_approximation_rate_list[-1] if self.test_approximation_rate_list else 0.0,
            'best_val_approximation_rate': max(self.val_approximation_rate_list) if self.val_approximation_rate_list else 0.0,
            'best_test_approximation_rate': max(self.test_approximation_rate_list) if self.test_approximation_rate_list else 0.0
        }
    
    def save_results(self, config_info: dict = None):
        """結果をファイルに保存"""
        # 結果データの準備
        results = {
            'timestamp': self.timestamp,
            'train_loss_list': self.train_loss_list,
            'train_err_edges_list': self.train_err_edges_list,
            'val_approximation_rate_list': self.val_approximation_rate_list,
            'test_approximation_rate_list': self.test_approximation_rate_list,
            'final_metrics': self.get_final_metrics(),
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
            f.write("TRAINING RESULTS\n")
            f.write("="*50 + "\n")
            f.write(f"Timestamp: {self.timestamp}\n\n")
            
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
            
            f.write("\n" + "="*50 + "\n")
            f.write("DETAILED RESULTS\n")
            f.write("="*50 + "\n")
            
            # エポックごとの詳細結果
            f.write(f"{'Epoch':<6} {'Train Loss':<12} {'Edge Error':<12} {'Val Approx':<12} {'Test Approx':<12}\n")
            f.write("-" * 60 + "\n")
            
            max_epochs = len(self.train_loss_list)
            for epoch in range(max_epochs):
                train_loss = self.train_loss_list[epoch]
                edge_error = self.train_err_edges_list[epoch]
                
                # 検証とテストの近似率（該当するエポックのみ）
                val_approx = "N/A"
                test_approx = "N/A"
                
                # 検証近似率の取得（簡略化）
                if epoch < len(self.val_approximation_rate_list):
                    val_approx = f"{self.val_approximation_rate_list[epoch]:.2f}%"
                
                # テスト近似率の取得（簡略化）
                if epoch < len(self.test_approximation_rate_list):
                    test_approx = f"{self.test_approximation_rate_list[epoch]:.2f}%"
                
                f.write(f"{epoch+1:<6} {train_loss:<12.4f} {edge_error:<12.2f} {val_approx:<12} {test_approx:<12}\n")
            
            f.write("-" * 60 + "\n")
        
        print(f"Results saved to Text: {txt_filename}")
        
        return pickle_filename, txt_filename
    
    def print_summary(self):
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